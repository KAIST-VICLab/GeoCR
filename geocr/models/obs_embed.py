"""Per-observation modality embedding, added to every token of an observation block after its
input projection. Modality ids: 0 = null/padded, 1 = S2, 2 = S1.
"""
from __future__ import annotations

from torch import Tensor, nn

MOD_NULL, MOD_S2, MOD_S1 = 0, 1, 2


class ObsEmbedder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.mod_emb = nn.Embedding(3, hidden)
        nn.init.normal_(self.mod_emb.weight, std=0.02)

    def forward(self, mod: Tensor) -> Tensor:
        """``mod [B,S]`` -> ``[B,S,hidden]``."""
        return self.mod_emb(mod)
