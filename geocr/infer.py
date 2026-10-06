"""GeoCR inference: predictions for one split of one dataset.

    python -m geocr.infer --dataset t-cloud --out outputs/t-cloud/wo-ft
    python -m geocr.infer --dataset t-cloud --lora t-cloud --out outputs/t-cloud/lora

The learned flow is integrated from Gaussian noise with four Euler steps on a uniform time grid,
without classifier-free guidance; the noise of each sample is seeded by its split-list id, so the
output does not depend on the batch size. ``--out`` receives ``preds/<id>.npy`` (CHW float in the
dataset's domain, unclipped; bands the model does not predict are NaN) and ``manifest.json``.
Weights come from the Hugging Face repo (downloaded on first use) or a local copy of it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from geocr.config import DataCfg, ModelCfg, RunCfg
from geocr.data.benchmarks import PROVIDERS
from geocr.data.cr_corpus import FAMILY, NIR_POS, STEM_KEY, apply_family_slice
from geocr.flows.rectified import euler_sample
from geocr.lora import inject_lora
from geocr.normalize import MS_IDX, RGB_IDX, S2_BANDS, Norms
from geocr.obs_pack import assemble_obs, encode_target
from geocr.train import _to_device, build_model, load_weights

REPO = "JeonghyeokDo/GeoCR"
NFE = 4
EVAL_SEED = 0
MS_STEMS = ("10", "10b", "1")       # non-RGB stems of stems/stems.safetensors (TOA, BOA, NIR)
#: Recorded as the training split of GeoCR (w/o FT), which was trained on the pretraining
#: mixture rather than on the split list of the dataset it is evaluated on.
PRETRAIN_SPLIT = "pretrain:pretrain_multi"
CR_DATASETS = Path(os.environ.get("CR_DATASETS", "CR_datasets"))

_RGB, _RGBN = ["R", "G", "B"], ["R", "G", "B", "NIR"]
#: Dataset -> Hub slug, provider kind, pretraining-mixture member (band family and stem),
#: processing level, output bands, resolution, and directory under $CR_DATASETS.
REGISTRY: dict[str, dict] = {
    "sen12mscr": dict(slug="sen12ms-cr", kind="sen12mscr", member="sen12mscr", level="l1c_toa",
                      bands=list(S2_BANDS), resolution=256, root="SEN12MS-CR"),
    "sen12mscr_rgb": dict(slug="sen12ms-cr-rgb", kind="sen12mscr_rgb", member=None,
                          level="display8", bands=_RGB, resolution=256, root="SEN12MS-CR"),
    "sen2mtc_new": dict(slug="sen2-mtc-new", kind="sen2mtc_new", member="sen2mtc_new",
                        level="boa", bands=_RGBN, resolution=256, root="Sen2_MTC_New"),
    "sen2mtc_new_rgb": dict(slug="sen2-mtc-new-rgb", kind="sen2mtc_new_rgb", member=None,
                            level="display8", bands=_RGB, resolution=256, root="Sen2_MTC_New"),
    "whus2crv": dict(slug="whus2-crv", kind="whus2crv", member="whus2crv", level="boa",
                     bands=list(S2_BANDS), resolution=384, root="WHUS2-CRv/extracted"),
    "whus2crv_rgb": dict(slug="whus2-crv-rgb", kind="whus2crv_rgb", member=None,
                         level="display8", bands=_RGB, resolution=256, root="WHUS2-CRv/extracted"),
    "tcloud": dict(slug="t-cloud", kind="tcloud", member="tcloud", level="display8",
                   bands=_RGB, resolution=256, root="T-CLOUD"),
    "cuhk_cr1": dict(slug="cuhk-cr1", kind="cuhk_cr", member="cuhk_cr", level="display8",
                     bands=_RGBN, resolution=256, root="C-CUHK/CUHK-CR1"),
    "cuhk_cr2": dict(slug="cuhk-cr2", kind="cuhk_cr", member="cuhk_cr2", level="display8",
                     bands=_RGB, resolution=256, root="C-CUHK/CUHK-CR2"),
}
#: Accepted dataset names (Hub slug or registry key) -> registry key.
NAMES = {e["slug"]: k for k, e in REGISTRY.items()} | {k: k for k in REGISTRY}


def sid_stem(sid: str) -> str:
    """Split-list id -> prediction file stem: ``/`` becomes ``__``, the extension is dropped."""
    return Path(sid.replace("/", "__")).stem


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _noise_seed(eval_seed: int, sid: str) -> int:
    """Per-sample RNG seed. blake2b, not the builtin ``hash()``, which is salted per process."""
    d = hashlib.blake2b(f"{eval_seed}|{sid}".encode(), digest_size=8).digest()
    return int.from_bytes(d, "little") & ((1 << 63) - 1)


def _sample_noise(sids: list[str], shape: tuple, eval_seed: int) -> Tensor:
    """Starting noise for one batch, drawn per sample on the CPU from a generator keyed by its
    id, so it depends neither on the batching nor on the device."""
    out = []
    for sid in sids:
        g = torch.Generator().manual_seed(_noise_seed(eval_seed, sid))
        out.append(torch.randn(shape[1:], generator=g))
    return torch.stack(out)


def _nir_plane(ms_refl: Tensor) -> Tensor:
    return ms_refl[:, 0] if ms_refl.shape[1] == 1 else ms_refl[:, NIR_POS]


def to_dump_array(entry: dict, rgb_refl: Tensor,
                  ms_refl: Tensor | None) -> tuple[Tensor, list[str]]:
    """Decoded reflectance -> the dataset's output array (NaN where a band is not predicted) and
    the list of predicted bands. 8-bit datasets are returned as png/255 (``refl * rgb_gain``)."""
    b, _, h, w = rgb_refl.shape
    n_bands = len(entry["bands"])
    out = torch.full((b, n_bands, h, w), float("nan"),
                     dtype=torch.float32, device=rgb_refl.device)
    if entry["level"] == "display8":
        g = torch.tensor(Norms().s2.rgb_gain, dtype=rgb_refl.dtype,
                         device=rgb_refl.device).view(1, 3, 1, 1)
        out[:, :3] = rgb_refl * g
        predicted = list(entry["bands"][:3])
        if n_bands == 4 and ms_refl is not None:
            out[:, 3] = _nir_plane(ms_refl)
            predicted.append("NIR")
    elif n_bands == 4:
        out[:, :3] = rgb_refl
        predicted = list(entry["bands"][:3])
        if ms_refl is not None:
            out[:, 3] = _nir_plane(ms_refl)
            predicted.append("NIR")
    else:
        for k, i in enumerate(RGB_IDX):
            out[:, i] = rgb_refl[:, k]
        predicted = [S2_BANDS[i] for i in RGB_IDX]
        if ms_refl is not None and ms_refl.shape[1] == len(MS_IDX):
            for k, i in enumerate(MS_IDX):
                out[:, i] = ms_refl[:, k]
            predicted = list(entry["bands"])
    return out, predicted


def dump_corpus(model, codec, provider, sids: list[str], entry: dict, out_dir: Path, *,
                batch_size: int, device) -> dict:
    """Write ``<out_dir>/preds/<sid_stem>.npy`` for every id. Each batch is cut to the band family
    and encoded with the stem the pretraining used for this dataset."""
    preds = out_dir / "preds"
    preds.mkdir(parents=True, exist_ok=True)
    norms = Norms()
    member = entry["member"]
    fam, stem = (FAMILY[member], STEM_KEY[member]) if member else ("none", None)
    dtype = np.float16 if len(entry["bands"]) == 13 and len(sids) >= 5000 else np.float32
    rng = np.random.default_rng(0)        # feeds only the provider's SAR-drop and null draws (off)
    k, sar_used, predicted = None, False, None
    for i in range(0, len(sids), batch_size):
        chunk = sids[i:i + batch_size]
        samples = [provider._sample(s, rng) for s in chunk]
        batch = apply_family_slice({kk: torch.stack([s[kk] for s in samples])
                                    for kk in samples[0]}, fam)
        if stem:
            batch["ms_stem"] = stem
        batch = _to_device(batch, device)
        obs = assemble_obs(codec, batch)
        shape = encode_target(codec, batch).shape        # only the latent shape is used
        fn = lambda z, t: model(z, t, obs["obs_lat"], obs["obs_mod"])
        z_hat = euler_sample(fn, shape, NFE, device, z0=_sample_noise(chunk, shape, EVAL_SEED))
        rgb, ms = codec.decode_s2(z_hat, stem=stem or 10)
        ms_refl = norms.s2.ms_to_refl(ms) if fam in ("ms10", "nir1") else None
        arr, predicted = to_dump_array(entry, norms.s2.rgb_to_refl(rgb), ms_refl)
        k = int(batch["frames_rgb"].shape[1])
        sar_used = sar_used or bool(batch["s1_valid"].any())
        for bi, s in enumerate(chunk):
            np.save(preds / f"{sid_stem(s)}.npy", arr[bi].cpu().numpy().astype(dtype))
        if (i // batch_size) % 25 == 0:
            print(f"[infer] {i + len(chunk)}/{len(sids)}", flush=True)
    return dict(n_written=len(sids), k=k, sar_used=sar_used, bands_predicted=predicted,
                dtype=np.dtype(dtype).name)


def _weights_dir(spec: str, lora_slug: str | None) -> Path:
    """A local directory as given, or a snapshot of the Hub repo with only the needed files."""
    if Path(spec).is_dir():
        return Path(spec)
    from huggingface_hub import snapshot_download
    patterns = ["transformer/*", "vae/*", "stems/*"]
    if lora_slug:
        patterns.append(f"lora/{lora_slug}/*")
    return Path(snapshot_download(spec, allow_patterns=patterns))


def _read_adapter(path: Path) -> dict:
    """LoRA tensors, rank, alpha, training data and training split list of an adapter directory
    (``adapter_model.safetensors`` + ``adapter_config.json``) or of a ``geocr.train`` checkpoint."""
    if path.is_dir():
        from safetensors.torch import load_file
        meta = json.loads((path / "adapter_config.json").read_text())
        return dict(state=load_file(path / "adapter_model.safetensors"), rank=meta["r"],
                    alpha=meta["lora_alpha"], trained_on=NAMES[meta["dataset"]],
                    train_split="trainval")
    ck = torch.load(path, map_location="cpu", mmap=True)
    cfg = ck["config"]
    return dict(state={k: v for k, v in ck["model"].items() if k.endswith(("lora_A", "lora_B"))},
                rank=cfg["model"]["lora_rank"], alpha=cfg["model"]["lora_alpha"],
                trained_on=cfg["data"]["source"], train_split=cfg["data"]["train_split"])


def load_model(weights: Path, adapter: dict | None, device):
    """GeoCR (w/o FT) from ``weights``; with ``adapter``, GeoCR (LoRA). The adapter must supply
    exactly the LoRA tensors of the wrapped model."""
    cfg = RunCfg(model=ModelCfg(
        ae_path=str(weights / "vae" / "ae.safetensors"),
        stems_path=str(weights / "stems" / "stems.safetensors"), ms_stems=list(MS_STEMS)))
    model, codec = build_model(cfg, device)
    model.load_state_dict(load_weights(weights / "transformer"))
    if adapter is not None:
        inject_lora(model, adapter["rank"], adapter["alpha"])
        want = {k for k in model.state_dict() if k.endswith(("lora_A", "lora_B"))}
        if set(adapter["state"]) != want:
            raise SystemExit(f"adapter holds {len(adapter['state'])} LoRA tensors; the model "
                             f"expects these {len(want)}: {sorted(want)[:2]}...")
        model.load_state_dict(adapter["state"], strict=False)
    return model.eval(), codec


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m geocr.infer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(NAMES), metavar="DATASET",
                    help="one of " + ", ".join(e["slug"] for e in REGISTRY.values()))
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", default=REPO,
                    help=f"Hugging Face repo id (default {REPO}) or a local directory with its "
                         "layout: transformer/, vae/, stems/, lora/<dataset>/")
    ap.add_argument("--lora", default="none",
                    help="none (GeoCR w/o FT), a dataset name (its adapter under --weights), an "
                         "adapter directory, or a LoRA checkpoint written by geocr.train")
    ap.add_argument("--out", required=True, type=Path,
                    help="output directory for preds/*.npy and manifest.json")
    ap.add_argument("--root", type=Path, help="dataset root (default: $CR_DATASETS/<dataset dir>)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    key = NAMES[args.dataset]
    entry = REGISTRY[key]
    root = args.root or CR_DATASETS / entry["root"]
    if (args.out / "manifest.json").exists():
        raise SystemExit(f"{args.out} already holds a finished run")
    data = DataCfg(source=entry["kind"], root=str(root), resolution=entry["resolution"],
                   s1_drop=0.0, p_uncond=0.0)
    provider = PROVIDERS[entry["kind"]](root, args.split, batch_size=1, seed=0, cfg=data)
    lora_slug = REGISTRY[NAMES[args.lora]]["slug"] if args.lora in NAMES else None
    weights = _weights_dir(args.weights, lora_slug)
    adapter = None
    if args.lora != "none":
        adapter = _read_adapter(weights / "lora" / lora_slug if lora_slug else Path(args.lora))
        if adapter["trained_on"] not in (key, entry["kind"]):
            raise SystemExit(f"the adapter was trained on {adapter['trained_on']!r}, "
                             f"not on {args.dataset!r}")
    device = torch.device(args.device)
    model, codec = load_model(weights, adapter, device)
    row = "no-ft" if adapter is None else "lora"
    print(f"[infer] {key} geocr-{row} n={len(provider.ids)} device={device}", flush=True)
    stats = dump_corpus(model, codec, provider, provider.ids, entry, args.out,
                        batch_size=args.batch_size, device=device)

    split_list = root / data.split_dir / f"{args.split}.txt"
    train_sha, n_train = PRETRAIN_SPLIT, None
    if adapter is not None:
        train_list = root / data.split_dir / f"{adapter['train_split']}.txt"
        train_sha = sha256_file(train_list)
        n_train = sum(1 for ln in train_list.read_text().splitlines()
                      if ln.strip() and not ln.lstrip().startswith("#"))
    res = entry["resolution"]
    man = {
        "corpus": key, "split": args.split, "split_list": str(split_list),
        "split_list_sha256": sha256_file(split_list), "n_items": stats["n_written"],
        "resolution": [res, res], "crop": "none", "bands": list(entry["bands"]),
        "domain": "unit_float" if entry["level"] == "display8" else "reflectance",
        "dtype": stats["dtype"], "processing_level": entry["level"],
        "method": f"geocr-{row}", "run_id": args.out.name, "ckpt_selection": "final",
        "train_split_sha256": train_sha, "n_train": n_train,
        "sampler": {"type": "euler", "nfe": NFE},
        "eval_noise": {"seed": EVAL_SEED, "keying": "blake2b(seed|sid) per sample"},
        "frames_used": stats["k"], "sar_used": stats["sar_used"],
        "bands_predicted": stats["bands_predicted"],
        "weights": str(weights), "lora": args.lora,
        "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "params_total": sum(p.numel() for p in model.parameters()),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "code_rev": f"infer:{sha256_file(Path(__file__))[:12]}",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.out / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    print(f"[done] {stats['n_written']} predictions -> {args.out}\n"
          f"       score: python -m geocr.eval --dataset {entry['slug']} --pred {args.out}")


if __name__ == "__main__":
    main()
