"""Experiment tracker for train_dspark.py: SwanLab, W&B, or none.

Both backends get the same metric dicts. Files go under <output-dir>/<backend>/ and the
run id is kept in <output-dir>/<backend>_run_id.txt, so --resume continues the same run.
Per-epoch metrics (val/*, train_epoch/*) also carry "epoch" and are plotted against it.
"""
from __future__ import annotations

import os
import secrets
import string
from pathlib import Path

EPOCH_KEYS = ("val/*", "train_epoch/*")


def add_tracker_args(p):
    p.add_argument("--tracker", choices=("none", "swanlab", "wandb"), default="none")
    p.add_argument("--tracker-project", default="minicpm-ocr-dspark")
    p.add_argument("--tracker-run-name", default=None, help="defaults to the output dir name")
    p.add_argument("--tracker-mode", choices=("online", "offline"), default="online",
                   help="offline: write locally, upload later (swanlab sync / wandb sync)")


def _run_id(path: Path) -> str:
    if path.is_file():
        return path.read_text().strip()
    run_id = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(12))
    path.write_text(run_id + "\n")
    return run_id


class Tracker:
    def __init__(self, args, config: dict):
        self.backend = args.tracker
        self.lib = None
        if self.backend == "none":
            return
        out = Path(args.output_dir)
        run_id = _run_id(out / f"{self.backend}_run_id.txt")
        name = args.tracker_run_name or out.name
        log_dir = out / self.backend
        log_dir.mkdir(parents=True, exist_ok=True)
        if self.backend == "swanlab":
            # Without a key, swanlab.init asks for one on the terminal and a tmux run waits there.
            if args.tracker_mode == "online" and not os.environ.get("SWANLAB_API_KEY"):
                raise SystemExit("--tracker swanlab: set SWANLAB_API_KEY (or use --tracker-mode offline)")
            import swanlab
            self.lib = swanlab
            swanlab.init(project=args.tracker_project, name=name, id=run_id, resume="allow",
                         mode="online" if args.tracker_mode == "online" else "offline",
                         log_dir=str(log_dir), config=config)
            for key in EPOCH_KEYS:
                swanlab.define_metric(key, x_axis="epoch")
        else:
            import wandb
            self.lib = wandb
            wandb.init(project=args.tracker_project, name=name, id=run_id, resume="allow",
                       mode=args.tracker_mode, dir=str(log_dir), config=config)
            wandb.define_metric("epoch")
            for key in EPOCH_KEYS:
                wandb.define_metric(key, step_metric="epoch")

    def log(self, metrics: dict, step: int, epoch: int | None = None):
        if self.lib is None:
            return
        if epoch is not None:
            metrics = {"epoch": epoch, **metrics}
        self.lib.log(metrics, step=step)

    def finish(self):
        if self.lib is not None:
            self.lib.finish()
