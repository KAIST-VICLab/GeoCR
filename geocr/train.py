"""GeoCR training: Stage 2 pretraining and LoRA adaptation.

    torchrun --nproc_per_node=4 -m geocr.train configs/pretrain.yaml [key=value ...]
    python -m geocr.train configs/lora/t-cloud.yaml [key=value ...]

Rank ``r`` of ``world`` reads sampler step ``(step * grad_accum + micro) * world + r``, so every
batch is a pure function of (seed, step, rank) and ``train.resume`` continues exactly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from geocr.config import RunCfg, load_config, resolve_and_freeze
from geocr.data.benchmarks import PROVIDERS
from geocr.data.cr_corpus import (NIR_POS, CRCorpus, apply_family_slice, family_for_source,
                                  stem_key_for_source)
from geocr.flows.rectified import euler_sample, flow_loss, interpolate, sample_t
from geocr.lora import inject_lora
from geocr.metrics.lineages import tcloud_metrics
from geocr.metrics.masked import masked_metrics, strip_ms_scope
from geocr.metrics.sen2mtc import sen2mtc_metrics
from geocr.models.flux2_ae import Flux2AE
from geocr.models.geocr_dit import GeoCRDiT, GeoCRParams
from geocr.models.s2_codec import DualS2Codec
from geocr.normalize import RGB_IDX, Norms
from geocr.obs_pack import assemble_obs, encode_target


def _dist_env() -> tuple[int, int]:
    if "RANK" in os.environ:
        torch.distributed.init_process_group("nccl")
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return rank, world
    return 0, 1


def _lr_at(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    p = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * p))


class Ema:
    """fp32 shadow of the trainable tensors (and float buffers); the decay warms in as
    ``min(decay, (1 + step) / (10 + step))``."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        params = dict(model.named_parameters())
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point
                       and (k not in params or params[k].requires_grad)}

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].lerp_(v.detach().float(), 1.0 - d)


def build_model(cfg: RunCfg, device) -> tuple[GeoCRDiT, DualS2Codec]:
    """The DiT (Klein-initialised when ``model.klein_path`` is set) and the frozen codec."""
    m = cfg.model
    ae = Flux2AE.from_pretrained(m.ae_path, device=device)
    codec = DualS2Codec(ae, ms_enabled=m.ms_enabled, ms_stems=tuple(m.ms_stems)).to(device)
    if m.stems_path:
        codec.load_stems(m.stems_path)
    codec.requires_grad_(False)
    model = GeoCRDiT(GeoCRParams(
        target_channels=m.target_channels, cond_channels=m.cond_channels,
        hidden_size=m.hidden_size, num_heads=m.num_heads, depth=m.depth,
        depth_single_blocks=m.depth_single_blocks, axes_dim=list(m.axes_dim),
        theta=m.theta, mlp_ratio=m.mlp_ratio)).to(device)
    if m.klein_path:
        from safetensors.torch import load_file
        report = model.load_klein(load_file(m.klein_path))
        report.pop("randomly_initialised")
        print(f"[load_klein] {json.dumps(report)}")
    return model, codec


def load_weights(path: str | Path) -> dict:
    """DiT state dict from a transformer directory (sharded safetensors with
    ``model.safetensors.index.json``) or from a training checkpoint, whose EMA tensors replace
    the raw tensors they shadow."""
    path = Path(path)
    if path.is_dir():
        from safetensors.torch import load_file
        shards = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        state = {}
        for shard in sorted(set(shards.values())):
            state.update(load_file(path / shard))
        return state
    ck = torch.load(path, map_location="cpu", mmap=True)
    return {**ck["model"], **(ck.get("ema") or {})}


def _to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def _gen_states(gen: torch.Generator, world: int) -> list:
    """Per-rank noise-generator states for the checkpoint; collective when world > 1."""
    if world == 1:
        return [gen.get_state()]
    states: list = [None] * world
    torch.distributed.all_gather_object(states, gen.get_state())
    return states


def _prune_ckpts(out_dir: Path, keep: int) -> list[str]:
    """Delete all but the ``keep`` newest ``ckpt_<step>.pt``; ``ckpt_final.pt`` never matches."""
    if keep <= 0:
        return []
    doomed = [p for p in sorted(out_dir.glob("ckpt_[0-9]*.pt"))
              if p.stem[len("ckpt_"):].isdigit()][:-keep]
    for p in doomed:
        p.unlink()
    return [p.name for p in doomed]


def _save(path: Path, raw, ema, opt, step: int, gen_states: list, cfg: RunCfg) -> None:
    torch.save({"model": raw.state_dict(),
                "ema": ema.shadow if ema is not None else None,
                "opt": opt.state_dict(), "step": step, "gen": gen_states,
                "config": resolve_and_freeze(cfg)}, path)


def _step_loss(model, codec, batch, cfg: RunCfg, device, gen):
    z1 = encode_target(codec, batch)
    obs = assemble_obs(codec, batch)
    t = sample_t(z1.shape[0], device, generator=gen,
                 scheme=cfg.flow.time_scheme, shift=cfg.flow.shift)
    eps = torch.randn(z1.shape, device=device, generator=gen)
    if not codec.ms_enabled or batch["target_ms"].shape[1] == 0:
        # z1's z_ms half is zero here; keep it zero at every t instead of feeding pure noise
        eps[:, 128:] = 0
    z_t, u_star = interpolate(z1, eps, t)
    v = model(z_t, t, obs["obs_lat"], obs["obs_mod"], uncond=batch["uncond"])
    # ms_loss_w = 0 for mixture members without a multispectral target
    loss, parts = flow_loss(v, u_star,
                            cfg.loss.ms_weight * float(batch.get("ms_loss_w", 1.0)))
    parts.update({"z_rgb_std": z1[:, :128].std(), "z_ms_std": z1[:, 128:].std(),
                  "ustar_rgb_rms": u_star[:, :128].pow(2).mean().sqrt(),
                  "ustar_ms_rms": u_star[:, 128:].pow(2).mean().sqrt()})
    return loss, parts


def _pad_ms10(ms_refl):
    """A one-band (NIR) decode placed at B8 of a zero 10-band block; the flag marks it partial."""
    if ms_refl.shape[1] == 10:
        return ms_refl, False
    out = ms_refl.new_zeros(ms_refl.shape[0], 10, *ms_refl.shape[2:])
    out[:, NIR_POS] = ms_refl[:, 0]
    return out, True


@torch.no_grad()
def validate(model, codec, load_val, cfg: RunCfg, device, load_metric_batch=None) -> dict:
    """Flow loss on ``val_batches`` fixed batches (per member for the mixture), and metrics of
    NFE-8 samples of one fixed batch averaged over four fixed noise seeds."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(cfg.train.seed)
    losses, extras = [], []
    by_ds: dict[str, list[float]] = {}
    for i in range(cfg.train.val_batches):
        batch = _to_device(load_val(10_000_000 + i), device)
        loss, parts = _step_loss(model, codec, batch, cfg, device, gen)
        losses.append(loss.item())
        extras.append({k: v.item() for k, v in parts.items()})
        if "ds_name" in batch:
            by_ds.setdefault(batch["ds_name"], []).append(loss.item())
    out = {"val_loss": sum(losses) / len(losses)}
    for k in extras[0]:
        out[f"val_{k}"] = sum(e[k] for e in extras) / len(extras)
    for name, ls in sorted(by_ds.items()):
        out[f"val_loss_{name}"] = sum(ls) / len(ls)
        out[f"val_n_{name}"] = len(ls)

    batch = _to_device((load_metric_batch or load_val)(10_000_000), device)
    obs = assemble_obs(codec, batch)
    z1 = encode_target(codec, batch)
    norms = Norms()
    src = cfg.data.source
    stem = 10
    if cfg.model.ms_enabled and src != "cr_corpus":
        stem = stem_key_for_source(src) or 10
    fn = lambda z, t: model(z, t, obs["obs_lat"], obs["obs_mod"])
    acc: dict = {}
    for s in (0, 1, 2, 3):
        z_hat = euler_sample(fn, z1.shape, 8, device,
                             generator=torch.Generator(device=device).manual_seed(s))
        rgb, ms = codec.decode_s2(z_hat, stem=stem)
        ms_refl, partial = _pad_ms10(norms.s2.ms_to_refl(ms))
        pred = norms.s2.merge_refl(norms.s2.rgb_to_refl(rgb), ms_refl)
        m = masked_metrics(pred, batch["target_refl"], batch["eval_mask"][:, None])
        if not codec.ms_enabled or partial:
            m = strip_ms_scope(m)
        m = {k: v for k, v in m.items() if k in ("psnr_rgb", "psnr_nonrgb", "psnr_all", "sam_all")}
        if src in ("sen2mtc_new", "tcloud"):
            pr = pred[:, list(RGB_IDX)]
            gr = batch["target_refl"][:, list(RGB_IDX)]
            if src == "sen2mtc_new":
                m.update(sen2mtc_metrics(pr, gr))
            else:
                g = torch.tensor(norms.s2.rgb_gain, device=pr.device).view(1, 3, 1, 1)
                m.update(tcloud_metrics((pr * g).clamp(0, 1), (gr * g).clamp(0, 1)))
        for k, v in m.items():
            acc[k] = acc.get(k, 0.0) + v / 4.0
    out.update({f"nfe8_stable_{k}": v for k, v in acc.items()})
    model.train()
    return out


def _loaders(cfg: RunCfg, world: int):
    """``step -> batch`` loaders for training, validation and the decoded-metrics batch."""
    d, t = cfg.data, cfg.train
    if d.source == "cr_corpus":
        roots, weights = dict(d.corpus_roots), dict(d.corpus_weights)
        train = CRCorpus(roots, weights, d.train_split, t.batch_size, t.seed, d,
                         draws_per_opt_step=t.grad_accum * world)
        val = CRCorpus(roots, weights, "val", t.batch_size, t.seed + 1, d)
        # one fixed AllClear batch keeps the decoded metrics comparable across the run
        metric = val.loaders["allclear"].load if "allclear" in val.loaders else None
        return train.load, val.load, metric
    provider = PROVIDERS[d.source]
    fam, stem = family_for_source(d.source), stem_key_for_source(d.source)

    def keyed(load):
        def _load(step):
            b = dict(load(step))
            if cfg.model.ms_enabled:        # band slice and stem exactly as in pretraining
                b = apply_family_slice(b, fam)
                b["ms_stem"] = stem
            return b
        return _load

    return (keyed(provider(d.root, d.train_split, t.batch_size, t.seed, d).load),
            keyed(provider(d.root, "val", t.batch_size, t.seed + 1, d).load), None)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m geocr.train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="yaml config, e.g. configs/lora/t-cloud.yaml")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.output_dir=runs/x")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    rank, world = _dist_env()
    device = torch.device("cuda", torch.cuda.current_device()) \
        if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(cfg.train.seed + rank)

    out_dir = Path(cfg.train.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(resolve_and_freeze(cfg), indent=2))
    load_train, load_val, load_metric_batch = _loaders(cfg, world)

    model, codec = build_model(cfg, device)
    if cfg.train.init_from:
        model.load_state_dict(load_weights(cfg.train.init_from))
        print(f"[init_from] {cfg.train.init_from}")
    if cfg.model.lora_rank > 0:             # lora_A is initialised from the seeded global RNG
        n = inject_lora(model, cfg.model.lora_rank, cfg.model.lora_alpha)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[lora] wrapped {n} linears, rank {cfg.model.lora_rank}, trainable {trainable:,}")
    raw = model
    ddp = None
    if world > 1:
        model = ddp = nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    if cfg.train.compile:
        model = torch.compile(model)        # training forward only; raw stays eager
    params = [p for p in raw.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    ema = Ema(raw, cfg.train.ema_decay) if cfg.train.ema_decay > 0 else None
    gen = torch.Generator(device=device).manual_seed(cfg.train.seed * world + rank)
    autocast = torch.autocast("cuda", torch.bfloat16,
                              enabled=cfg.train.mixed_precision == "bf16"
                              and device.type == "cuda")

    start = 0
    if cfg.train.resume:
        ck = torch.load(cfg.train.resume, map_location="cpu")
        raw.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if ema is not None and ck.get("ema") is not None:
            ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
        if len(ck["gen"]) != world:
            raise ValueError(f"resume: checkpoint holds {len(ck['gen'])} noise-generator "
                             f"states but world={world}")
        gen.set_state(ck["gen"][rank].cpu())
        start = ck["step"] + 1
        del ck

    accum = cfg.train.grad_accum
    log = (out_dir / "train.jsonl").open("a") if rank == 0 else None
    t0 = time.time()
    nonfinite_streak = nonfinite_total = 0
    for step in range(start, cfg.train.steps):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, cfg.train)
        opt.zero_grad(set_to_none=True)
        parts_acc: dict = {}
        for micro in range(accum):
            batch = _to_device(load_train((step * accum + micro) * world + rank), device)
            # gradients are all-reduced once per optimizer step
            sync = nullcontext() if (ddp is None or micro == accum - 1) else ddp.no_sync()
            with sync:
                with autocast:
                    loss, parts = _step_loss(model, codec, batch, cfg, device, gen)
                (loss / accum).backward()
            parts_acc = {k: parts_acc.get(k, 0.0) + v.item() / accum
                         for k, v in {**parts, "loss": loss.detach()}.items()}
        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.train.max_grad_norm)
        # a non-finite gradient skips the update; 25 in a row stop the run
        if bool(torch.isfinite(gnorm)):
            opt.step()
            if ema is not None:
                ema.update(raw, step)
            nonfinite_streak = 0
        else:
            nonfinite_streak += 1
            nonfinite_total += 1
            if nonfinite_streak >= 25:
                raise RuntimeError(f"non-finite grad norm for {nonfinite_streak} consecutive "
                                   f"optimizer steps (step {step})")

        if rank == 0 and (step % cfg.train.log_every == 0 or step == cfg.train.steps - 1):
            rec = {"step": step, "lr": opt.param_groups[0]["lr"],
                   "grad_norm": float(gnorm), "sec": time.time() - t0, **parts_acc,
                   **({"nonfinite_skips": nonfinite_total} if nonfinite_total else {})}
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if rank == 0 and step and step % cfg.train.eval_every == 0:
            rec = {"step": step, **validate(raw, codec, load_val, cfg, device,
                                            load_metric_batch)}
            with (out_dir / "eval.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
        if step and step % cfg.train.ckpt_every == 0:
            gen_states = _gen_states(gen, world)
            if rank == 0:
                _save(out_dir / f"ckpt_{step:07d}.pt", raw, ema, opt, step, gen_states, cfg)
                gone = _prune_ckpts(out_dir, cfg.train.ckpt_keep)
                if gone:
                    print(f"[ckpt] pruned {len(gone)}: {', '.join(gone)}", flush=True)
    gen_states = _gen_states(gen, world)
    if rank == 0:
        _save(out_dir / "ckpt_final.pt", raw, ema, opt, cfg.train.steps - 1, gen_states, cfg)
        log.close()
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
