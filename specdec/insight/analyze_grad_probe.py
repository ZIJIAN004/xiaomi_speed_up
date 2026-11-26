#!/usr/bin/env python3
"""Tables and figures from grad_probe.py outputs (``<run>/probe/step_*.pt``).

For every probe step, parameter group (all, backbone, each module) and
geometry (raw, adam):

* decomposition.csv  rho_k = 1 + earlier_k + later_k + conf_k per slot, in the
  signal view (cross-group, noise-free) and the batch view (same-group, what a
  training batch sees); the signal view also carries the min/max of later_k
  over the half splits of the sample groups (stability).
* whatif.csv         the same decomposition under other decay gammas, from the
  same undecayed gradients, with each slot's weight.
* noise.csv          per term: signal |E g|^2, batch |mean g|^2 and the share of
  the latter that is noise.
* checks.csv         reconstruction cosine and the real-step validation.
* report.txt         per step, later_k of the backbone (signal view) and the
  slots whose later_k <= -threshold in every half split.
* fig_*.png          if matplotlib is installed.

    python specdec/insight/analyze_grad_probe.py runs/insight/<run>/probe [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grad_probe import SIDE_HEADS, batch_view, cosine, decompose, signal_view, split_signals, what_if  # noqa: E402

DEFAULT_GAMMAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, math.inf)


def load_probes(probe_dir: Path):
    for path in sorted(probe_dir.glob("step_*.pt")):
        summary = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        yield summary, torch.load(path, map_location="cpu", weights_only=False)


def with_coarse(fine: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {"all": sum(fine.values()),
           "backbone": sum(v for k, v in fine.items() if k not in SIDE_HEADS)}
    out.update(fine)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("probe_dir", type=Path)
    p.add_argument("--out", type=Path, default=None, help="default: <probe_dir>/analysis")
    p.add_argument("--gammas", default=",".join(str(g) for g in DEFAULT_GAMMAS))
    p.add_argument("--threshold", type=float, default=0.1,
                   help="report slots whose later_k <= -threshold in every half split")
    args = p.parse_args(argv)
    out = args.out or args.probe_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    gammas = [float(g) for g in args.gammas.split(",")]

    rows_dec, rows_whatif, rows_noise, rows_checks, report, series = [], [], [], [], [], []
    for summary, tensors in load_probes(args.probe_dir):
        step, meta = summary["opt_step"], tensors["meta"]
        slot_ids, labels = meta["slot_ids"], meta["labels"]
        coef = torch.tensor(meta["coef"], dtype=torch.float64)
        n_slots = len(slot_ids)
        alpha = meta["alpha"] if len(labels) > n_slots else None
        recon = summary["checks"]["reconstruction"]
        step_checks = summary["checks"]["step"] or {}
        rows_checks.append({"step": step, "recon_cos": recon["cos"], "recon_rel_err": recon["rel_err"],
                            **{f"{k}_{f}": (v or {}).get(f) for k, v in step_checks.items()
                               for f in ("corr", "ratio")}})
        main_kind = "adam" if tensors["pair_grams"].get("adam") else "raw"
        for kind, fine in tensors["pair_grams"].items():
            if fine is None:
                continue
            for group, pair in with_coarse(fine).items():
                signal, batch = signal_view(pair), batch_view(pair)
                splits = [decompose(g, coef, n_slots)["later"] for g in split_signals(pair)]
                for view, gram in (("signal", signal), ("batch", batch)):
                    dec = decompose(gram, coef, n_slots)
                    for i, k in enumerate(slot_ids):
                        row = {"step": step, "kind": kind, "group": group, "view": view, "slot": k,
                               **{key: dec[key][i] for key in ("rho", "earlier", "later", "conf")}}
                        if view == "signal" and splits:
                            row["later_split_min"] = min(s[i] for s in splits)
                            row["later_split_max"] = max(s[i] for s in splits)
                        rows_dec.append(row)
                for gamma, dec in what_if(signal, slot_ids, gammas, alpha, meta["sample_from_anchor"]).items():
                    for i, k in enumerate(slot_ids):
                        rows_whatif.append({"step": step, "kind": kind, "group": group, "gamma": gamma, "slot": k,
                                            "weight": dec["weight"][i], "later": dec["later"][i],
                                            "rho": dec["rho"][i]})
                for v, label in enumerate(labels):
                    s_sq, b_sq = float(signal[v, v]), float(batch[v, v])
                    rows_noise.append({"step": step, "kind": kind, "group": group, "term": label,
                                       "signal_sq": s_sq, "batch_sq": b_sq,
                                       "noise_share": 1.0 - s_sq / b_sq if b_sq > 0 else float("nan")})
                if kind == main_kind and group == "backbone":
                    later = decompose(signal, coef, n_slots)["later"]
                    stable = [k for i, k in enumerate(slot_ids)
                              if splits and max(s[i] for s in splits) <= -args.threshold]
                    series.append((step, later, cosine(signal)[:n_slots, :n_slots]))
                    report.append(f"step {step:6d} [{kind}, backbone, signal] later_k: "
                                  + " ".join(f"{k}:{x:+.3f}" for k, x in zip(slot_ids, later, strict=True))
                                  + f"\n             later_k <= -{args.threshold} in every split: {stable or 'none'}")

    for name, rows in (("decomposition", rows_dec), ("whatif", rows_whatif), ("noise", rows_noise),
                       ("checks", rows_checks)):
        if rows:
            keys = list(dict.fromkeys(k for row in rows for k in row))
            with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
    (out / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    plot(out, series, rows_whatif)
    print(f"wrote {out}")


def plot(out: Path, series, rows_whatif) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; tables only")
        return
    if not series:
        return
    steps = [s for s, _, _ in series]
    n_slots = len(series[0][1])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.get_cmap("viridis")
    for k in range(n_slots - 1):  # the last slot has no later slots
        ax.plot(steps, [later[k] for _, later, _ in series], color=cmap(k / max(n_slots - 2, 1)), label=f"slot {k}")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("later_k (backbone, signal)")
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_later_vs_step.png", dpi=150)
    plt.close(fig)
    for step, _, cos in series:
        fig, ax = plt.subplots(figsize=(5, 4.3))
        im = ax.imshow(cos.numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(f"slot gradient cosine, step {step} (backbone, signal)")
        ax.set_xlabel("slot")
        ax.set_ylabel("slot")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out / f"fig_cos_step{step:06d}.png", dpi=150)
        plt.close(fig)
    main_rows = [r for r in rows_whatif if r["group"] == "backbone"]
    kinds = {r["kind"] for r in main_rows}
    kind = "adam" if "adam" in kinds else "raw"
    for step in steps:
        rows = [r for r in main_rows if r["step"] == step and r["kind"] == kind]
        gammas = sorted({r["gamma"] for r in rows}, key=float)
        finite = [g for g in gammas if math.isfinite(float(g))]
        if not finite:
            continue
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        slots = sorted({r["slot"] for r in rows})
        for k in slots[:-1]:
            ys = [next(r["later"] for r in rows if r["gamma"] == g and r["slot"] == k) for g in finite]
            ws = [next(r["weight"] for r in rows if r["gamma"] == g and r["slot"] == k) for g in finite]
            color = cmap(k / max(len(slots) - 2, 1))
            top.plot([float(g) for g in finite], ys, color=color, label=f"slot {k}")
            bottom.plot([float(g) for g in finite], ws, color=color)
        top.axhline(0, color="grey", lw=0.8)
        top.set_ylabel("later_k under gamma")
        bottom.set_ylabel("slot weight w_k")
        bottom.set_xscale("log")
        bottom.set_xlabel("decay gamma")
        top.legend(ncol=3, fontsize=7)
        top.set_title(f"what if: step {step} ({kind}, backbone, signal)")
        fig.tight_layout()
        fig.savefig(out / f"fig_whatif_step{step:06d}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
