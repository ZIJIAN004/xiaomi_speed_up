"""Compatibility shim for OmniDocBench v1.5 pages whose prediction contains a LaTeX table.

When a page's prediction has more LaTeX tables than HTML tables, end2end_dataset matches the GT tables with
line_type='latex_table', and utils.match.get_gt_pred_lines reads item['latex'] from every GT table. The v1.5 GT
stores tables as HTML only (no table carries a 'latex' key), so a single LaTeX-table prediction aborts the whole
evaluation with KeyError: 'latex'.

The value is never scored: for 'latex_table' get_gt_pred_lines replaces gt_lines with the GT HTML, and
end2end_dataset then blanks the prediction and sets edit=1 for every LaTeX-table match. So supplying an empty
'latex' for GT tables that lack one only lets the evaluator reach its own rule (the table scores zero); it does
not change any score. GT items are copied, never modified in place.
"""
from __future__ import annotations

from functools import wraps
import sys


def _wrap(current):
    if getattr(current, "_omnidocbench_latex_table_compat", False):
        return current

    @wraps(current)
    def get_gt_pred_lines_compat(gt_mix, pred_dataset_mix, line_type=None, *args, **kwargs):
        if line_type == "latex_table":
            gt_mix = [item if "latex" in item else {**item, "latex": ""} for item in gt_mix]
        return current(gt_mix, pred_dataset_mix, line_type, *args, **kwargs)

    get_gt_pred_lines_compat._omnidocbench_latex_table_compat = True
    return get_gt_pred_lines_compat


def install() -> None:
    import utils.match as match  # the official evaluator root is on PYTHONPATH

    wrapped = _wrap(match.get_gt_pred_lines)
    match.get_gt_pred_lines = wrapped
    # match_quick imports get_gt_pred_lines by value; patch that binding if it was already imported.
    match_quick = sys.modules.get("utils.match_quick")
    if match_quick is not None and hasattr(match_quick, "get_gt_pred_lines"):
        match_quick.get_gt_pred_lines = wrapped


install()
