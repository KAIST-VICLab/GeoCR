"""Stage 1: stem adaptation around the frozen FLUX.2 autoencoder trunks (Section 3.1).

Each route reconstructs its own observation under an l1 loss in the preprocessed input
domain. Only the selected stems train, with AdamW at a constant learning rate and no
warmup or gradient clipping.

    # ten-band TOA stem "10" and SAR stem "2" on AllClear, global batch 64 on two GPUs
    torchrun --nproc_per_node=2 -m geocr.stems toa --root <transcoded AllClear> \\
        --ae <ae.safetensors> --out <toa_dir>
    # ten-band BOA stem "10b" on WHUS2-CRv, initialised from "10"
    python -m geocr.stems boa --root <WHUS2-CRv/extracted> --ae <ae.safetensors> \\
        --init <toa_dir>/stems.safetensors --out <boa_dir>

The single-band NIR stem "1" keeps its closed-form initialisation and is never trained.
<boa_dir>/stems.safetensors holds all four stems in the released format. On one GPU,
``toa --batch 64`` draws the same samples per update as two GPUs at 32.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from geocr.config import DataCfg
from geocr.data.benchmarks import WHUS2CRvProvider
from geocr.data.dataset import AllClearObsDataset
from geocr.data.index import AllClearIndex
from geocr.models.flux2_ae import Flux2AE
from geocr.models.multistem_ae import MultiStemAE
from geocr.normalize import MS_IDX, Norms

NORMS = Norms()
KIND_S2, KIND_S1 = 0, 1
SPECKLE_BLOB, L_AUG_LO, L_AUG_HI, P_SPECKLE = 3, 8, 32, 0.5
DB_NODATA_GUARD = -45.0
CKPT_EVERY = 2000


def stable_u64(*parts) -> int:
    """Process-stable 64-bit hash (the builtin ``hash`` is salted per process)."""
    h = hashlib.blake2b("\x1f".join(str(p) for p in parts).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little")


def dihedral(x: torch.Tensor, k: int) -> torch.Tensor:
    """View ``k`` in 0..7 of a square tile: rot90 x (k mod 4), after an hflip if k >= 4."""
    if k >= 4:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, k % 4, dims=(-2, -1))


def speckle_db(db: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """With probability ``P_SPECKLE``, multiply the power of a raw-dB SAR scene ``[2,H,W]``
    by one unit-mean Gamma field shared by VV and VH, applied before the model-range clip.
    Values at or below ``DB_NODATA_GUARD`` are no-data and stay unchanged."""
    if rng.random() >= P_SPECKLE:
        return db
    looks = float(rng.integers(L_AUG_LO, L_AUG_HI + 1))
    pad = SPECKLE_BLOB - 1
    micro = rng.gamma(shape=looks / SPECKLE_BLOB**2, scale=1.0 / looks,
                      size=(1, 1, db.shape[-2] + pad, db.shape[-1] + pad))
    # A box-sum of SPECKLE_BLOB^2 draws of Gamma(L / SPECKLE_BLOB^2, rate L) is exactly
    # Gamma(L, rate L): the box correlates neighbours and leaves the marginal unchanged.
    g = F.conv2d(torch.from_numpy(micro).float(), torch.ones(1, 1, SPECKLE_BLOB, SPECKLE_BLOB))[0]
    noise = (10.0 / math.log(10.0)) * torch.log(g.clamp_min(1e-12))
    return torch.where(db > DB_NODATA_GUARD, db + noise, db)


def build_schedule(n_s2: int, n_s1: int, visits: int, batch: int, world: int,
                   seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``(kind, within-kind step)`` per update: ``visits`` passes over both scene pools
    at ``batch * world`` samples per update, the two kinds interleaved by a seeded shuffle."""
    per = batch * world
    kinds = np.array([KIND_S2] * math.ceil(visits * n_s2 / per)
                     + [KIND_S1] * math.ceil(visits * n_s1 / per), dtype=np.int8)
    np.random.default_rng([seed, 7]).shuffle(kinds)
    s1 = kinds == KIND_S1
    return kinds, np.where(s1, np.cumsum(s1), np.cumsum(~s1)) - 1


def _scene_lists(index: AllClearIndex, rois: list[str]):
    """``(roi id, scene)`` arrays of the S2 pool (at most 1 % no-data) and the S1 pool."""
    r2, i2, r1, i1 = [], [], [], []
    for rid, roi in enumerate(rois):
        t = index.tables[roi]
        keep = np.flatnonzero(np.asarray(t.s2_nan) <= 1.0)
        r2.append(np.full(len(keep), rid, dtype=np.int32))
        i2.append(keep.astype(np.int32))
        if len(t.s1_date):
            r1.append(np.full(len(t.s1_date), rid, dtype=np.int32))
            i1.append(np.arange(len(t.s1_date), dtype=np.int32))
    return ((np.concatenate(r2), np.concatenate(i2)),
            (np.concatenate(r1), np.concatenate(i1)))


def load_batch(dataset, rois, lists, perms, kind: int, j: int, batch: int, world: int,
               rank: int, visits: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One rank's ``(input, target)`` for within-kind step ``j``, a pure function of its
    arguments, so resuming needs only the step. Visit ``e`` of a scene uses its ``e``-th
    dihedral view; SAR inputs get speckle while the target stays the observation."""
    roi_ids, scene_is = lists[kind]
    n = len(roi_ids)
    xs_in, xs_tg = [], []
    for b in range(batch):
        pos = (j * batch * world + rank * batch + b) % (visits * n)   # the last update wraps
        e, flat = pos // n, int(perms[kind][pos // n][pos % n])
        roi, i = rois[int(roi_ids[flat])], int(scene_is[flat])
        if kind == KIND_S2:
            x = tg = NORMS.s2.ms_to_model(dataset.s2_refl(roi, i)[list(MS_IDX)][None])[0]
        else:
            db = dataset.s1_db(roi, i)
            tg = NORMS.s1.to_model(db)
            x = NORMS.s1.to_model(speckle_db(
                db, np.random.default_rng([seed, stable_u64(roi), i, e, 5])))
        k = (stable_u64(roi, i) + e) % 8
        xs_in.append(dihedral(x, k))
        xs_tg.append(xs_in[-1] if x is tg else dihedral(tg, k))
    return torch.stack(xs_in), torch.stack(xs_tg)


def _whus2_crop(whus2: WHUS2CRvProvider, rng: np.random.Generator) -> torch.Tensor:
    """Ten non-RGB bands of a random 256 px crop of a clear or cloudy 384 px clip."""
    sid = whus2.ids[int(rng.integers(len(whus2.ids)))]
    x13 = whus2._to13(sid, "clear" if rng.random() < 0.5 else "cloud")
    y, x0 = (int(v) for v in rng.integers(0, 384 - 256 + 1, size=2))
    return NORMS.s2.ms_to_model(x13[list(MS_IDX), y:y + 256, x0:x0 + 256][None])[0]


def _device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    return device


def _build(ae_path: str, keys: tuple[str, ...], device) -> MultiStemAE:
    """Stems ``keys`` in their closed-form initialisation around the frozen trunks."""
    return MultiStemAE(Flux2AE.from_pretrained(ae_path, device=device), keys).to(device)


def _stem_state(stems: MultiStemAE, keys) -> dict:
    return {n: t for n, t in stems.state_dict().items()
            if n.split(".")[0] in ("stem_in", "stem_out") and n.split(".")[1] in keys}


def _load_stems(stems: MultiStemAE, path: str, keys) -> None:
    sd = load_file(path)
    stems.load_state_dict({n: sd[n] for n in _stem_state(stems, keys)}, strict=False)


def _save_stems(stems: MultiStemAE, keys, path: Path) -> None:
    save_file({n: t.detach().cpu().contiguous() for n, t in _stem_state(stems, keys).items()},
              str(path), metadata={"channel_counts": ",".join(keys)})


def _trainable(stems: MultiStemAE, keys) -> list[torch.nn.Parameter]:
    stems.requires_grad_(False)
    params = [p for side in (stems.stem_in, stems.stem_out) for k in keys
              for p in side[k].parameters()]
    for p in params:
        p.requires_grad_(True)
    return params


def train_toa(a: argparse.Namespace) -> None:
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = _device()
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = ("10", "2")
    stems = _build(a.ae, keys, device)
    if a.init:
        _load_stems(stems, a.init, keys)
    index = AllClearIndex(a.root)
    dataset = AllClearObsDataset(index)
    rois = index.split_rois("train")
    lists = _scene_lists(index, rois)
    pools = [len(lists[KIND_S2][0]), len(lists[KIND_S1][0])]
    kinds, prefix = build_schedule(*pools, a.visits, a.batch, world, a.seed)
    perms = [[np.random.default_rng([a.seed, kind, e]).permutation(n) for e in range(a.visits)]
             for kind, n in enumerate(pools)]
    total = min(a.steps, len(kinds))
    params = _trainable(stems, keys)
    opt = torch.optim.AdamW(params, lr=a.lr)
    start, ckpt = 0, out / "ckpt.pt"
    if ckpt.exists():
        ck = torch.load(ckpt, map_location=device)
        stems.load_state_dict(ck["stems"], strict=False)
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
    if rank == 0:
        print(f"pools s2={pools[0]:,} s1={pools[1]:,}, schedule {len(kinds):,} updates, "
              f"running {start:,} -> {total:,}", flush=True)
        log = (out / "train.jsonl").open("a")

    def fetch(step):
        return load_batch(dataset, rois, lists, perms, int(kinds[step]), int(prefix[step]),
                          a.batch, world, rank, a.visits, a.seed)

    pool = ThreadPoolExecutor(max_workers=1)
    nxt = pool.submit(fetch, start)
    for step in range(start, total):
        x_in, x_tg = nxt.result()
        if step + 1 < total:
            nxt = pool.submit(fetch, step + 1)
        key = "10" if kinds[step] == KIND_S2 else "2"
        x_in, x_tg = x_in.to(device, non_blocking=True), x_tg.to(device, non_blocking=True)
        loss = (stems.decode(stems.encode(x_in, key), key) - x_tg).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if world > 1:
            for p in params:
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad /= world
        opt.step()
        done = step + 1
        if rank == 0 and step % 100 == 0:
            log.write(json.dumps({"step": step, "stem": key, "l1": loss.item(),
                                  "t": time.time()}) + "\n")
            log.flush()
        if rank == 0 and (done % CKPT_EVERY == 0 or done == total):
            torch.save({"step": done, "stems": _stem_state(stems, keys),
                        "opt": opt.state_dict()}, ckpt.with_suffix(".tmp"))
            os.replace(ckpt.with_suffix(".tmp"), ckpt)
    if rank == 0:
        _save_stems(stems, keys, out / "stems.safetensors")
        print(f"stems -> {out / 'stems.safetensors'}", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def train_boa(a: argparse.Namespace) -> None:
    device = _device()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = ("10", "10b", "1", "2")
    stems = _build(a.ae, keys, device)
    _load_stems(stems, a.init, ("10", "2"))
    with torch.no_grad():
        for side in (stems.stem_in, stems.stem_out):
            side["10b"].weight.copy_(side["10"].weight)
            side["10b"].bias.copy_(side["10"].bias)
    opt = torch.optim.AdamW(_trainable(stems, ("10b",)), lr=a.lr)
    whus2 = WHUS2CRvProvider(a.root, "train", 1, 0, DataCfg(root=a.root))
    rng = np.random.default_rng(a.seed)
    log = (out / "train.jsonl").open("a")
    for step in range(a.steps):
        x = torch.stack([_whus2_crop(whus2, rng) for _ in range(a.batch)]).to(device)
        loss = (stems.decode(stems.encode(x, "10b"), "10b") - x).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            log.write(json.dumps({"step": step, "l1": loss.item(), "t": time.time()}) + "\n")
            log.flush()
    _save_stems(stems, keys, out / "stems.safetensors")
    print(f"stems -> {out / 'stems.safetensors'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m geocr.stems", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    toa = sub.add_parser("toa", help='train "10" and "2" on AllClear')
    toa.add_argument("--root", required=True, help="transcoded AllClear root")
    toa.add_argument("--init", help='stems.safetensors whose "10" and "2" start training '
                                    "(default: closed-form initialisation)")
    toa.add_argument("--visits", type=int, default=10, help="passes over every training scene")
    toa.add_argument("--batch", type=int, default=32, help="per GPU")
    toa.add_argument("--steps", type=int, default=400_000,
                     help="maximum number of updates; the run also ends with the visit schedule")
    boa = sub.add_parser("boa", help='train "10b" on WHUS2-CRv, starting from "10"')
    boa.add_argument("--root", default=os.path.join(os.environ.get("CR_DATASETS", "."),
                                                    "WHUS2-CRv", "extracted"),
                     help="WHUS2-CRv root with splits/train.txt "
                          "(default: $CR_DATASETS/WHUS2-CRv/extracted)")
    boa.add_argument("--init", required=True, help='stems.safetensors with "10" and "2"')
    boa.add_argument("--batch", type=int, default=8)
    boa.add_argument("--steps", type=int, default=20_000)
    for p in (toa, boa):
        p.add_argument("--ae", required=True, help="FLUX.2 autoencoder ae.safetensors")
        p.add_argument("--out", required=True)
        p.add_argument("--lr", type=float, default=1e-4)
        p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    (train_toa if a.stage == "toa" else train_boa)(a)


if __name__ == "__main__":
    main()
