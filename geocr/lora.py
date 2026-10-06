"""LoRA on every ``nn.Linear`` inside ``double_blocks`` and ``single_blocks`` (80 layers at the release
geometry). Input projections, modulations, ``time_in`` and ``final_layer`` are not wrapped.

``LoRALinear`` adopts the wrapped module's Parameter objects, so the base state-dict keys are unchanged
and only ``lora_A``/``lora_B`` are added. ``lora_B`` starts at zero, so injection preserves the function.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LoRALinear(nn.Linear):
    """``y = x W^T (+ b) + scale * (x A^T) B^T`` with A kaiming, B zero, ``scale = alpha / rank``."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        nn.Linear.__init__(self, base.in_features, base.out_features,
                           bias=base.bias is not None)
        self.weight = base.weight
        if base.bias is not None:
            self.bias = base.bias
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features,
                                               device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank,
                                               device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_scale = alpha / rank

    def forward(self, x: Tensor) -> Tensor:
        y = F.linear(x, self.weight, self.bias)
        return y + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scale


def inject_lora(model: nn.Module, rank: int, alpha: float) -> int:
    """Wrap every nn.Linear under double_blocks/single_blocks and freeze everything else.

    Returns the number of wrapped Linears; afterwards only ``lora_A``/``lora_B`` require grad.
    """
    if rank <= 0:
        raise ValueError("inject_lora requires rank > 0")
    targets = []
    for blocks in (model.double_blocks, model.single_blocks):
        for blk in blocks:
            for _, parent in blk.named_modules():
                for child_name, child in parent.named_children():
                    if type(child) is nn.Linear:
                        targets.append((parent, child_name, child))
    for parent, child_name, child in targets:
        setattr(parent, child_name, LoRALinear(child, rank, alpha))
    for name, p in model.named_parameters():
        p.requires_grad_("lora_A" in name or "lora_B" in name)
    return len(targets)
