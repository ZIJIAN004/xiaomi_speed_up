"""Xiaomi-EAGLE draft modules.

The draft network is independent of the frozen Xiaomi target decoder. The
Qwen3.5-specific attention/cache adapter is deliberately isolated behind
`PendingQwen35AttentionAdapter`; its exact kernel/cache wiring is left for the
separate target-interface validation pass.
"""
from __future__ import annotations

import torch
from torch import nn


class Qwen35RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * (1.0 + self.weight.float())).to(dtype=x.dtype)


class FeatureFusion(nn.Module):
    def __init__(self, hidden_size: int, low: int = 1, mid: int = 12, high: int = 24):
        super().__init__()
        self.layer_indices = (low, mid, high)
        self.proj = nn.Linear(3 * hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        selected = [hidden_states[i] for i in self.layer_indices]
        return self.proj(torch.cat(selected, dim=-1))


class PendingQwen35AttentionAdapter(nn.Module):
    """Placeholder for the validated Qwen3.5-compatible 2H attention.

    Required contract:
      input:  [B, T, 2H]
      output: [B, T, H]
      cache:  draft KV/cache object, implementation pending validation

    Do not silently replace this with ordinary Llama attention: Qwen3.5 has
    q/k norms, gated queries, RoPE, hybrid cache semantics, and backend-specific
    attention behavior.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "Qwen3.5 draft-attention/cache adapter awaits target-interface validation"
        )


class DraftMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class DraftBlock(nn.Module):
    """Draft block shell; attention wiring remains an explicit validation gate."""
    def __init__(self, hidden_size: int, intermediate_size: int, **attention_kwargs):
        super().__init__()
        self.hidden_norm = Qwen35RMSNorm(hidden_size)
        self.embedding_norm = Qwen35RMSNorm(hidden_size)
        self.attention = PendingQwen35AttentionAdapter(
            hidden_size=hidden_size, input_size=2 * hidden_size, **attention_kwargs
        )
        self.post_attention_norm = Qwen35RMSNorm(hidden_size)
        self.mlp = DraftMLP(hidden_size, intermediate_size)

    def forward(self, feature: torch.Tensor, embedding: torch.Tensor, **kwargs):
        residual = feature
        attention_input = torch.cat(
            [self.embedding_norm(embedding), self.hidden_norm(feature)], dim=-1
        )
        feature, cache = self.attention(attention_input, **kwargs)
        feature = residual + feature
        residual = feature
        feature = self.mlp(self.post_attention_norm(feature))
        return residual + feature, cache


class XiaomiEagleDraft(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, intermediate_size: int,
                 low_layer: int = 1, mid_layer: int = 12, high_layer: int = 24,
                 num_blocks: int = 1, **attention_kwargs):
        super().__init__()
        self.fusion = FeatureFusion(hidden_size, low_layer, mid_layer, high_layer)
        self.blocks = nn.ModuleList([
            DraftBlock(hidden_size, intermediate_size, **attention_kwargs)
            for _ in range(num_blocks)
        ])
        self.output_norm = Qwen35RMSNorm(hidden_size)
        self.draft_lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def initial_feature(self, target_hidden_states):
        return self.fusion(target_hidden_states)

    def forward_step(self, feature, embedding, cache=None, **kwargs):
        for block in self.blocks:
            feature, cache = block(feature, embedding, cache=cache, **kwargs)
        logits = self.draft_lm_head(self.output_norm(feature))
        return feature, logits, cache
