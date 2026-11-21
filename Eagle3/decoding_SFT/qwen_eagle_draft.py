"""Trainable EAGLE draft network for Xiaomi-OCR-0.

The target model is kept outside this module and frozen. Draft parameters are
new parameters. The block follows the official EAGLE-3 contract: feature H and
embedding H are normalized, concatenated to 2H, projected by new attention
parameters, and returned to H before residual/MLP.
"""
from __future__ import annotations

import copy
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
                 head_dim: int, eps: float, rotary_emb: nn.Module):
        super().__init__()
        if num_heads % num_kv_heads:
            raise ValueError("num_attention_heads must divide num_key_value_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        # Copy the loaded checkpoint's native rotary implementation/configuration.
        self.rotary_emb = copy.deepcopy(rotary_emb)
        self.rotary_emb.requires_grad_(False)
        input_size = 2 * hidden_size
        self.q_proj = nn.Linear(input_size, 2 * num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(input_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(input_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = Qwen35RMSNorm(head_dim, eps)
        self.k_norm = Qwen35RMSNorm(head_dim, eps)

    def _apply(self, fn, recurse=True):
        # Module.to(dtype=...) must not quantize native FP32 RoPE frequencies.
        saved = [(module, name, value.clone())
                 for module in self.rotary_emb.modules()
                 for name, value in module._buffers.items()
                 if value is not None and value.is_floating_point()]
        result = super()._apply(fn, recurse=recurse)
        for module, name, original in saved:
            module._buffers[name] = original.to(device=module._buffers[name].device)
        return result

    def forward(self, x: torch.Tensor, cache: Optional[list[dict[str, torch.Tensor]]] = None,
                position_ids: Optional[torch.Tensor] = None):
        bsz, q_len, _ = x.shape
        q_gate = self.q_proj(x).view(bsz, q_len, self.num_heads, 2 * self.head_dim)
        q, gate = q_gate.chunk(2, dim=-1)
        k = self.k_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        if position_ids is None:
            position_ids = torch.arange(q_len, device=x.device).view(1, -1).expand(bsz, -1)
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]  # native text plane is not a rotary axis
        if position_ids.shape != (3, bsz, q_len):
            raise ValueError("draft position_ids must have shape [3, batch, sequence]")
        history = list(cache or [])
        for previous in history:
            if previous["k"].shape != k.shape or previous["v"].shape != v.shape:
                raise ValueError("training rollout cache requires unchanged batch/sequence shapes")
        # Official TTT uses base positions + rollout depth, not token-cache length.
        cos, sin = self.rotary_emb(x, position_ids + len(history))
        rotary_dim = cos.shape[-1]
        if rotary_dim > self.head_dim or rotary_dim % 2:
            raise ValueError("native rotary dimension must be even and <= head_dim")
        cos, sin = cos[:, None], sin[:, None]
        def rotate(tensor):
            rotated, passthrough = tensor[..., :rotary_dim], tensor[..., rotary_dim:]
            half = rotary_dim // 2
            turned = torch.cat((-rotated[..., half:], rotated[..., :half]), dim=-1)
            return torch.cat((rotated * cos + turned * sin, passthrough), dim=-1)
        q, k = rotate(q), rotate(k)
        history.append({"k": k, "v": v})
        keys = [entry["k"].repeat_interleave(self.num_groups, dim=1) for entry in history]
        values = [entry["v"].repeat_interleave(self.num_groups, dim=1) for entry in history]
        # First round: dense causal anchor. Later rounds, INCLUDING current:
        # one same-column K/V each. This is a training cache, not a token cache.
        scores = torch.matmul(q, keys[0].transpose(-2, -1)) * self.scale
        causal = torch.triu(torch.ones(q_len, q_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal[None, None], torch.finfo(scores.dtype).min)
        for pk in keys[1:]:
            scores = torch.cat([scores, (q * pk).sum(-1, keepdim=True) * self.scale], dim=-1)
        probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        out = torch.matmul(probs[..., :q_len], values[0])
        offset = q_len
        for pv in values[1:]:
            out = out + probs[..., offset:offset + 1] * pv
            offset += 1
        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads * self.head_dim)
        out = out * torch.sigmoid(gate.reshape(bsz, q_len, -1))
        return self.o_proj(out), history


class DraftBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int,
                 num_kv_heads: int, head_dim: int, eps: float,
                 rotary_emb: nn.Module):
        super().__init__()
        self.feature_norm = Qwen35RMSNorm(hidden_size, eps)
        self.embedding_norm = Qwen35RMSNorm(hidden_size, eps)
        self.attn = DraftAttention(hidden_size, num_heads, num_kv_heads, head_dim, eps, rotary_emb)
        self.post_norm = Qwen35RMSNorm(hidden_size, eps)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, feature, embedding, cache=None, position_ids=None):
        residual = feature
        x = torch.cat([self.embedding_norm(embedding), self.feature_norm(feature)], dim=-1)
        x, cache = self.attn(x, cache, position_ids=position_ids)
        x = residual + x
        residual = x
        normalized = self.post_norm(x)
        x = self.down_proj(F.silu(self.gate_proj(normalized)) * self.up_proj(normalized))
        return residual + x, cache


class XiaomiEagleDraft(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, intermediate_size: int,
                 num_heads: int, num_kv_heads: int, head_dim: int, rms_eps: float,
                 low_layer: int = 1, mid_layer: int = 12, high_layer: int = 24,
                 rollout_steps: int = 7, *, rotary_emb: nn.Module):
        super().__init__()
        self.rollout_steps = rollout_steps
        self.fusion = FeatureFusion(hidden_size, low_layer, mid_layer, high_layer)
        self.block = DraftBlock(hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, rms_eps, rotary_emb)
        self.output_norm = Qwen35RMSNorm(hidden_size, rms_eps)
        self.draft_lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def initial_feature(self, target_hidden_states):
        return self.fusion(tuple(target_hidden_states))

    def step(self, feature, embedding, cache=None, position_ids=None):
        feature, cache = self.block(feature, embedding, cache, position_ids=position_ids)
        logits = self.draft_lm_head(self.output_norm(feature))
        return feature, logits, cache
