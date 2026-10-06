"""Closed OmegaConf schema: unknown keys are errors and missing required values fail loudly.

A run is one flat yaml plus dotted CLI overrides (``data.root=...``). Paths may be written as
``${oc.env:CR_DATASETS}/...`` interpolations; they are resolved when the config is loaded.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from omegaconf import MISSING, DictConfig, OmegaConf

#: ``cr_corpus`` is the pretraining mixture; the others are single-benchmark provider kinds.
SOURCES = ("cr_corpus", "sen12mscr", "sen2mtc_new", "whus2crv", "cuhk_cr", "tcloud",
           "sen12mscr_rgb", "whus2crv_rgb", "sen2mtc_new_rgb")
#: Sources with no multispectral observation: they train with ``model.ms_enabled=false``.
RGB_ONLY_SOURCES = ("tcloud", "sen12mscr_rgb", "whus2crv_rgb", "sen2mtc_new_rgb")


@dataclass
class DataCfg:
    source: str = MISSING
    root: Optional[str] = None           # benchmark root; unused by cr_corpus
    #: cr_corpus only: member name -> root, and member name -> sampling weight (normalised).
    corpus_roots: Dict[str, str] = field(default_factory=dict)
    corpus_weights: Dict[str, float] = field(default_factory=dict)
    split_dir: str = "splits"            # {train,val,trainval,test}.txt, relative to each root
    train_split: str = "train"           # list to train on; finetunes use "trainval"
    resolution: int = 256
    k_choices: List[int] = field(default_factory=lambda: [1, 2, 3])
    k_probs: List[float] = field(default_factory=lambda: [0.34, 0.33, 0.33])
    #: During the first k_ramp_steps optimizer steps only a prefix of k_choices is drawn,
    #: unlocking linearly until the full list is active. 0 disables.
    k_ramp_steps: int = 0
    window_days: int = 40                # max |dt| of a condition frame to the target
    target_cloud_max: float = 10.0       # target eligibility, percent of pixels
    target_shadow_max: float = 10.0
    target_nan_max: float = 1.0
    s1_max: int = 1                      # SAR slots per sample
    s1_drop: float = 0.3                 # drop SAR even when available
    sar_only_frac: float = 0.0           # fraction of batches with SAR-only conditions
    p_uncond: float = 0.1                # all conditions replaced by the null token


@dataclass
class ModelCfg:
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 5
    depth_single_blocks: int = 20
    axes_dim: List[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0
    target_channels: int = 256           # [z_rgb | z_ms]
    cond_channels: int = 256
    ae_path: str = MISSING               # FLUX.2 ae.safetensors
    klein_path: Optional[str] = None     # FLUX.2 [klein] 4B base weights (pretraining init)
    stems_path: Optional[str] = None     # Stage 1 stems; None -> closed-form initial stems
    ms_enabled: bool = True              # False: z_ms is fed as zeros
    #: Multispectral stem keys "<width>[letter]" ("10" TOA, "10b" BOA, "1" NIR); must match
    #: the stems file. The SAR stem is always built.
    ms_stems: List[str] = field(default_factory=lambda: ["10"])
    #: LoRA on every nn.Linear inside double_blocks/single_blocks (rank 0 = off). The base,
    #: the projections, modulations, time_in, obs_embed and null_token stay frozen.
    lora_rank: int = 0
    lora_alpha: float = 16.0


@dataclass
class FlowCfg:
    time_scheme: str = "logit_normal"    # t = sigmoid(N(0,1) - log(shift)), training draw only
    shift: float = 1.0                   # odds multiplier; > 1 pushes t toward noise


@dataclass
class LossCfg:
    ms_weight: float = 1.0               # weight of the z_ms half of the flow loss


@dataclass
class TrainCfg:
    seed: int = 2026
    output_dir: str = MISSING
    steps: int = 100000
    batch_size: int = 8                  # per process
    grad_accum: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 2000
    min_lr: float = 1e-6
    max_grad_norm: float = 1.0
    ema_decay: float = 0.9999
    mixed_precision: str = "bf16"        # bf16 | no
    compile: bool = False                # after the DDP wrap; EMA and checkpoints stay eager
    log_every: int = 50
    ckpt_every: int = 5000
    ckpt_keep: int = 0                   # keep N most recent ckpt_*.pt (0 = all); not final
    eval_every: int = 2500
    val_batches: int = 32
    resume: Optional[str] = None         # full resume: weights + optimizer + step
    init_from: Optional[str] = None      # weights-only initialisation (finetuning)


@dataclass
class RunCfg:
    name: str = MISSING
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    flow: FlowCfg = field(default_factory=FlowCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    train: TrainCfg = field(default_factory=TrainCfg)


def load_config(path: str, overrides: Optional[Sequence[str]] = None) -> RunCfg:
    try:
        cfg = OmegaConf.merge(OmegaConf.structured(RunCfg), OmegaConf.load(path))
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
        run: RunCfg = OmegaConf.to_object(cfg)
    except Exception as exc:
        raise ValueError(
            f"invalid configuration from {path}: {type(exc).__name__}: {exc}\n"
            "  Every key must exist in geocr.config.RunCfg."
        ) from exc
    _validate(run, str(path))
    return run


def _validate(cfg: RunCfg, origin: str) -> None:
    def bad(msg: str) -> None:
        raise ValueError(f"{origin}: {msg}")

    d, m, f, t = cfg.data, cfg.model, cfg.flow, cfg.train
    if d.source not in SOURCES:
        bad(f"data.source={d.source!r} not in {SOURCES}")
    if d.source != "cr_corpus" and not d.root:
        bad(f"data.root is required for data.source={d.source!r}")
    if d.source in RGB_ONLY_SOURCES and m.ms_enabled:
        bad(f"data.source={d.source} has no multispectral observation; "
            "set model.ms_enabled=false")
    if len(d.k_choices) != len(d.k_probs):
        bad(f"k_choices ({len(d.k_choices)}) and k_probs ({len(d.k_probs)}) differ in length")
    if abs(sum(d.k_probs) - 1.0) > 1e-6:
        bad(f"k_probs must sum to 1, got {sum(d.k_probs)}")
    if any(k < 1 for k in d.k_choices):
        bad("k_choices entries must be >= 1")
    if d.k_ramp_steps < 0:
        bad("data.k_ramp_steps must be >= 0")
    if d.k_ramp_steps and list(d.k_choices) != sorted(d.k_choices):
        bad("data.k_ramp_steps requires k_choices sorted ascending")
    for name in ("s1_drop", "sar_only_frac", "p_uncond"):
        v = getattr(d, name)
        if not 0.0 <= v <= 1.0:
            bad(f"data.{name}={v} must be in [0,1]")
    if d.resolution % 16 != 0:
        bad(f"data.resolution={d.resolution} must be a multiple of 16")

    if m.hidden_size % m.num_heads != 0:
        bad("model.hidden_size must be divisible by model.num_heads")
    if sum(m.axes_dim) != m.hidden_size // m.num_heads:
        bad(f"sum(model.axes_dim)={sum(m.axes_dim)} must equal head_dim="
            f"{m.hidden_size // m.num_heads}")
    if m.target_channels != 256 or m.cond_channels != 256:
        bad("model.target_channels and model.cond_channels must be 256 ([z_rgb|z_ms])")
    if m.klein_path is not None and (m.hidden_size, m.num_heads, m.depth,
                                     m.depth_single_blocks) != (3072, 24, 5, 20):
        bad("model.klein_path requires the Klein-4B geometry")
    if m.lora_rank < 0:
        bad("model.lora_rank must be >= 0 (0 disables LoRA)")
    if m.lora_rank > 0 and m.lora_alpha <= 0:
        bad("model.lora_alpha must be > 0 when lora_rank > 0")
    if m.lora_rank > 0 and not (t.init_from or m.klein_path):
        bad("model.lora_rank > 0 needs train.init_from or model.klein_path")

    if f.time_scheme != "logit_normal":
        bad(f"flow.time_scheme={f.time_scheme!r}; only 'logit_normal' is supported")
    if f.shift < 1.0:
        bad("flow.shift must be >= 1")

    if t.warmup_steps >= t.steps:
        bad("train.warmup_steps must be < train.steps")
    if t.mixed_precision not in ("bf16", "no"):
        bad(f"train.mixed_precision={t.mixed_precision!r} not in ('bf16','no')")
    if t.ckpt_keep < 0:
        bad("train.ckpt_keep must be >= 0 (0 = keep every checkpoint)")
    if not 0.0 <= t.ema_decay < 1.0:
        bad(f"train.ema_decay={t.ema_decay} must be in [0, 1)")
    if cfg.loss.ms_weight < 0:
        bad("loss.ms_weight must be >= 0")
    if not m.ms_enabled and cfg.loss.ms_weight != 0.0:
        bad("model.ms_enabled=false requires loss.ms_weight=0 (the z_ms half is zeros)")
    keys = [str(k) for k in m.ms_stems]
    if not keys or len(keys) != len(set(keys)) \
            or any(not re.match(r"\d+[a-z]?$", k) for k in keys):
        bad(f"model.ms_stems={keys} must be unique '<width>[letter]' keys")

    if d.source == "cr_corpus":
        from geocr.data.cr_corpus import FAMILY, STEM_KEY
        roots, weights = dict(d.corpus_roots), dict(d.corpus_weights)
        if not roots:
            bad("source=cr_corpus requires data.corpus_roots")
        if set(roots) != set(weights):
            bad(f"corpus_roots keys {sorted(roots)} != corpus_weights keys {sorted(weights)}")
        unknown = sorted(set(roots) - set(FAMILY))
        if unknown:
            bad(f"unknown corpus dataset(s) {unknown}; known: {sorted(FAMILY)}")
        if any(w <= 0 for w in weights.values()):
            bad("corpus_weights must all be > 0")
        missing = sorted({STEM_KEY[n] for n in roots if STEM_KEY.get(n)} - set(keys))
        if missing:
            bad(f"corpus members need ms stem key(s) {missing} but model.ms_stems={keys}")
    elif d.corpus_roots or d.corpus_weights:
        bad(f"data.corpus_* is only used by source=cr_corpus, not {d.source!r}")
    elif m.ms_enabled:
        # A finetune encodes its corpus with the stem the pretraining used for it.
        from geocr.data.cr_corpus import stem_key_for_source
        need = stem_key_for_source(d.source)
        if need and need not in keys:
            bad(f"data.source={d.source!r} needs ms stem key {need!r} "
                f"but model.ms_stems={keys}")


def resolve_and_freeze(cfg: Any) -> dict:
    node = OmegaConf.structured(cfg) if not isinstance(cfg, DictConfig) else cfg
    out = OmegaConf.to_container(node, resolve=True, throw_on_missing=True)
    json.dumps(out)
    return out
