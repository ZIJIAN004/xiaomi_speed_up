"""Trainable EAGLE draft network for Xiaomi-OCR-0.

The target model is kept outside this module and frozen. Draft parameters are
new parameters. The block follows the official EAGLE-3 contract: feature H and
embedding H are normalized, concatenated to 2H, projected by new attention
parameters, and returned to H before residual/MLP.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class Qwen35RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * (1.0 + self.weight.float())).to(x.dtype)


class FeatureFusion(nn.Module):
    def __init__(self, hidden_size: int, low: int, mid: int, high: int):
        super().__init__()
        self.layer_indices = (low, mid, high)
        self.proj = nn.Linear(3 * hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        selected = [hidden_states[i] for i in self.layer_indices]
        return self.proj(torch.cat(selected, dim=-1))


class DraftAttention(nn.Module):
    """New 2H-input causal attention with an explicit draft K/V cache."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, eps: float):
        super().__init__()
        if num_heads % num_kv_heads:
            raise ValueError("num_attention_heads must divide num_key_value_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        input_size = 2 * hidden_size
        self.q_proj = nn.Linear(input_size, 2 * num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(input_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(input_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = Qwen35RMSNorm(head_dim, eps)
        self.k_norm = Qwen35RMSNorm(head_dim, eps)

    def forward(self, x: torch.Tensor, cache: Optional[dict[str, torch.Tensor]] = None):
        bsz, q_len, _ = x.shape
        q_gate = self.q_proj(x).view(bsz, q_len, self.num_heads, 2 * self.head_dim)
        q, gate = q_gate.chunk(2, dim=-1)
        k = self.k_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        past_k = cache["k"] if cache is not None else None
        past_v = cache["v"] if cache is not None else None
        past_len = 0 if past_k is None else past_k.shape[-2]
        raw_k = torch.cat([past_k, k], dim=-2) if past_k is not None else k
        raw_v = torch.cat([past_v, v], dim=-2) if past_v is not None else v
        k_all = raw_k.repeat_interleave(self.num_groups, dim=1)
        v_all = raw_v.repeat_interleave(self.num_groups, dim=1)
        scores = torch.matmul(q, k_all.transpose(-2, -1)) * self.scale
        causal = torch.triu(
            torch.ones(q_len, q_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        if past_len:
            mask = torch.cat([torch.zeros(q_len, past_len, device=x.device, dtype=torch.bool), causal], dim=-1)
        else:
            mask = causal
        scores = scores.masked_fill(mask[None, None], torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        out = torch.matmul(probs, v_all).transpose(1, 2).contiguous()
        out = out.reshape(bsz, q_len, self.num_heads * self.head_dim)
        out = out * torch.sigmoid(gate.reshape(bsz, q_len, -1))
        return self.o_proj(out), {"k": raw_k, "v": raw_v}


class DraftBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int,
                 num_kv_heads: int, head_dim: int, eps: float):
        super().__init__()
        self.feature_norm = Qwen35RMSNorm(hidden_size, eps)
        self.embedding_norm = Qwen35RMSNorm(hidden_size, eps)
        self.attn = DraftAttention(hidden_size, num_heads, num_kv_heads, head_dim, eps)
        self.post_norm = Qwen35RMSNorm(hidden_size, eps)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, feature, embedding, cache=None):
        residual = feature
        x = torch.cat([self.embedding_norm(embedding), self.feature_norm(feature)], dim=-1)
        x, cache = self.attn(x, cache)
        x = residual + x
        residual = x
        normalized = self.post_norm(x)
        x = self.down_proj(F.silu(self.gate_proj(normalized)) * self.up_proj(normalized))
        return residual + x, cache


class XiaomiEagleDraft(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, intermediate_size: int,
                 num_heads: int, num_kv_heads: int, head_dim: int, rms_eps: float,
                 low_layer: int = 1, mid_layer: int = 12, high_layer: int = 24,
                 rollout_steps: int = 7):
        super().__init__()
        self.rollout_steps = rollout_steps
        self.fusion = FeatureFusion(hidden_size, low_layer, mid_layer, high_layer)
        self.block = DraftBlock(hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, rms_eps)
        self.output_norm = Qwen35RMSNorm(hidden_size, rms_eps)
        self.draft_lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def initial_feature(self, target_hidden_states):
        return self.fusion(tuple(target_hidden_states))

    def step(self, feature, embedding, cache=None):
        feature, cache = self.block(feature, embedding, cache)
        logits = self.draft_lm_head(self.output_norm(feature))
        return feature, logits, cache
