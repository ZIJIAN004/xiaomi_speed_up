"""Size-capped spherical k-means (grouping A of the draft lm_head rows).

Rows are L2-normalised, so a row's similarity to a centroid is the cosine: two
tokens whose head rows point the same way get similar logits from any hidden
state, i.e. they compete for the same slots. Every group holds at most
``cap = ceil(N / K * (1 + slack))`` rows, so a shortlist of m groups has a
bounded size and the gathered head slice has a regular shape on the GPU.

Seeding is k-means++ with distance 1 - cos (K sequential passes over the rows;
a few minutes on CPU for K = 4096 and ~110k rows).

Assignment per iteration: each row lists its ``topk`` most similar centroids; rows
are placed most-confident first (highest best similarity) into the first listed
group with room left. The few rows whose listed groups are all full take the most
similar group that still has room, searched over all centroids.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def capacity(n: int, k: int, slack: float) -> int:
    if k < 1 or n < k:
        raise ValueError(f"need 1 <= num_groups <= rows, got {k} groups for {n} rows")
    if slack < 0:
        raise ValueError("slack must be >= 0")
    return max(1, math.ceil(n / k * (1 + slack)))


def top_similar(x: torch.Tensor, centroids: torch.Tensor, topk: int, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    values, indices = [], []
    for start in range(0, x.shape[0], chunk):
        sims = x[start:start + chunk] @ centroids.T
        v, i = sims.topk(min(topk, centroids.shape[0]), dim=1)
        values.append(v.cpu())
        indices.append(i.cpu())
    return torch.cat(values).numpy(), torch.cat(indices).numpy()


def assign_with_capacity(values: np.ndarray, indices: np.ndarray, cap: int, k: int,
                         full_similarity) -> np.ndarray:
    """Greedy capped assignment; ``full_similarity(rows) -> [len(rows), k]`` for leftovers."""
    n = values.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    room = np.full(k, cap, dtype=np.int64)
    order = np.argsort(-values[:, 0], kind="stable")
    leftovers = []
    for row in order:
        for group in indices[row]:
            if room[group] > 0:
                labels[row] = group
                room[group] -= 1
                break
        else:
            leftovers.append(row)
    if leftovers:
        rows = np.asarray(leftovers)
        sims = full_similarity(rows)
        for offset in np.argsort(-sims.max(axis=1), kind="stable"):
            masked = np.where(room > 0, sims[offset], -np.inf)
            group = int(masked.argmax())
            if not np.isfinite(masked[group]):
                raise RuntimeError("capacity exhausted (cap * num_groups < rows)")
            labels[rows[offset]] = group
            room[group] -= 1
    return labels


@torch.no_grad()
def kmeanspp_init(x: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
    """k-means++ seeding with cosine distance 1 - cos (rows of x are unit vectors)."""
    n = x.shape[0]
    chosen = [int(torch.randint(n, (1,), generator=generator))]
    best = x @ x[chosen[0]]
    for _ in range(1, k):
        weight = (1 - best).clamp_min(0).cpu().double()
        if float(weight.sum()) <= 0:
            pick = int(torch.randint(n, (1,), generator=generator))
        else:
            pick = int(torch.multinomial(weight, 1, generator=generator))
        chosen.append(pick)
        best = torch.maximum(best, x @ x[pick])
    return x[torch.as_tensor(chosen, device=x.device)].clone()


@torch.no_grad()
def balanced_spherical_kmeans(rows: torch.Tensor, k: int, iters: int = 30, slack: float = 0.1, seed: int = 0,
                              topk: int = 16, tol: float = 1e-3, device: str = "cpu",
                              chunk: int = 8192) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Labels in [0, k) for every row (each group non-empty, size <= cap) and per-iteration stats."""
    x = torch.nn.functional.normalize(rows.float(), dim=1).to(device)
    n = x.shape[0]
    cap = capacity(n, k, slack)
    centroids = kmeanspp_init(x, k, torch.Generator().manual_seed(seed))
    labels = np.full(n, -1, dtype=np.int64)
    history: list[dict[str, float]] = []

    def full_similarity(subset: np.ndarray) -> np.ndarray:
        return (x[torch.as_tensor(subset, device=device)] @ centroids.T).cpu().numpy()

    for iteration in range(iters):
        values, indices = top_similar(x, centroids, topk, chunk)
        new_labels = assign_with_capacity(values, indices, cap, k, full_similarity)
        changed = float((new_labels != labels).mean())
        labels = new_labels
        index = torch.as_tensor(labels, device=device)
        sums = torch.zeros_like(centroids).index_add_(0, index, x)
        sizes = torch.bincount(index, minlength=k)
        empty = (sizes == 0).nonzero().squeeze(-1)
        if empty.numel():
            # Re-seed an empty group with the rows least similar to their own centroid.
            own = (x * centroids[index]).sum(1)
            worst = own.argsort()[:empty.numel()]
            sums[empty] = x[worst]
        centroids = torch.nn.functional.normalize(sums, dim=1)
        cosine = float((x * centroids[index]).sum(1).mean())
        history.append({"iter": iteration, "changed": changed, "mean_cosine": cosine,
                        "empty_reseeded": int(empty.numel())})
        print(f"kmeans iter {iteration}: changed={changed:.4f} mean_cos={cosine:.4f} "
              f"empty={int(empty.numel())}", flush=True)
        if iteration > 0 and changed < tol and not empty.numel():
            break
    # Final capped assignment against the final centroids; no group may stay empty.
    values, indices = top_similar(x, centroids, topk, chunk)
    labels = fill_empty(assign_with_capacity(values, indices, cap, k, full_similarity), x, centroids)
    return labels, history


@torch.no_grad()
def fill_empty(labels: np.ndarray, x: torch.Tensor, centroids: torch.Tensor) -> np.ndarray:
    """Give each empty group the row most similar to its centroid among groups of size > 1."""
    labels = labels.copy()
    sizes = np.bincount(labels, minlength=centroids.shape[0])
    for group in np.flatnonzero(sizes == 0):
        order = (x @ centroids[group]).argsort(descending=True).cpu().numpy()
        for row in order:
            if sizes[labels[row]] > 1:
                sizes[labels[row]] -= 1
                labels[row] = group
                sizes[group] = 1
                break
        else:
            raise RuntimeError("cannot fill an empty group: every group holds a single row")
    return labels
