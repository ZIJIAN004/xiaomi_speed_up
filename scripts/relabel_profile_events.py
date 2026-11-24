"""Per-page stage CUDA Event spans from representative profiles, relabeling pre-fix decode steps.

Runs profiled before the V2-runner classification fix labeled every decode step
`runner_mixed_or_unknown_step`. Representative profiles run one request at a time
(benchmark/run_benchmark.py profile loop), so after the single prefill step every
runner step is decode. This is checked per page: exactly one runner_prefill step
that precedes all other runner steps, and no overlapping runner steps.

Usage: python scripts/relabel_profile_events.py <run>/inference --output <dir>
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

DECODE_LABELS = {"runner_decode", "runner_mixed_or_unknown_step"}


def page_row(path, window_steps):
    events = [e for r in json.load(open(path))["results"] for e in r["events"]]
    runner = sorted((e for e in events if e["stage"].startswith("runner_")), key=lambda e: e["start_ms"])
    prefill = [e for e in runner if e["stage"] == "runner_prefill"]
    others = {e["stage"] for e in runner} - {"runner_prefill"} - DECODE_LABELS
    if len(prefill) != 1 or runner[0] is not prefill[0] or others:
        raise ValueError(f"{path}: expected one leading runner_prefill and decode steps only, got "
                         f"{len(prefill)} prefill, other labels {sorted(others)}")
    if any(b["start_ms"] < a["end_ms"] for a, b in zip(runner, runner[1:])):
        raise ValueError(f"{path}: overlapping runner steps; not a single-request profile")
    decode = runner[1:]
    span = lambda xs: sum(e["end_ms"] - e["start_ms"] for e in xs)
    stage_span = lambda name: span([e for e in events if e["stage"] == name]) or None
    gaps = [b["start_ms"] - a["end_ms"] for a, b in zip(runner, runner[1:])]
    # The profiler stopping/exporting at the end of its window stalls the engine once; report it apart.
    stall = max(range(len(gaps)), key=gaps.__getitem__) if gaps else None
    first = decode[:window_steps]
    return {
        "sample_id": path.parent.name,
        "vision_encoder_ms": stage_span("vision_encoder"),
        "projector_ms": stage_span("projector"),
        "prefill_step_ms": span(prefill),
        "decode_steps": len(decode),
        "decode_span_ms": span(decode),
        "decode_span_per_step_ms": span(decode) / len(decode) if decode else None,
        "decode_window_steps": len(first),
        "decode_window_span_per_step_ms": span(first) / len(first) if first else None,
        "inter_step_gap_median_ms": statistics.median(gaps) if gaps else None,
        "max_inter_step_gap_ms": gaps[stall] if gaps else None,
        "max_gap_after_runner_step": stall,
        "request_gpu_window_ms": max(e["end_ms"] for e in events) - min(e["start_ms"] for e in events),
        "request_gpu_window_minus_max_gap_ms": (max(e["end_ms"] for e in events) - min(e["start_ms"] for e in events)
                                                - (gaps[stall] if gaps else 0)),
        "relabeled_from_mixed": sum(e["stage"] == "runner_mixed_or_unknown_step" for e in decode),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inference_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-max-iterations", type=int, default=64,
                        help="profiler window in engine steps; window = this minus the prefill step")
    args = parser.parse_args()
    rows = [page_row(p, args.profile_max_iterations - 1)
            for p in sorted(args.inference_dir.glob("profiles/*/cuda_events.json"))]
    if not rows:
        raise SystemExit(f"no profiles/*/cuda_events.json under {args.inference_dir}")
    args.output.mkdir(parents=True, exist_ok=False)
    with open(args.output / "stage_events_per_page.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"pages": len(rows), "source": str(args.inference_dir),
               "semantics": "per-stream CUDA Event spans (include GPU idle inside a range); not kernel work",
               "stages": {}}
    for field in rows[0]:
        values = [r[field] for r in rows if isinstance(r[field], (int, float))]
        if field != "sample_id" and values:
            summary["stages"][field] = {"median": statistics.median(values), "mean": statistics.mean(values),
                                        "min": min(values), "max": max(values), "count": len(values)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    for field, s in summary["stages"].items():
        print(f"{field:32s} median={s['median']:10.2f} mean={s['mean']:10.2f} n={s['count']}")


if __name__ == "__main__":
    main()
