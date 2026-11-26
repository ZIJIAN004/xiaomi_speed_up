"""SwanLab ECharts panels for one probe (interactive; hover shows the values).

All panels use the backbone (markov / confidence heads excluded), the signal
view (cross-group, noise-free) and the AdamW geometry when the optimizer has
state (raw gradients at step 0):

* probe_chart/cos_heatmap         15 x 15 cosine between slot gradients
* probe_chart/slot_decomposition  earlier_k, later_k, conf_k and rho_k per slot
* probe_chart/whatif_later        later_k of every slot under other decay gammas
"""
from __future__ import annotations

import math
from typing import Any

from grad_probe import SIDE_HEADS, signal_view, what_if

WHATIF_GAMMAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, math.inf)


def main_kind(summary: dict[str, Any]) -> str:
    return "adam" if summary["views"].get("adam") else "raw"


def _clean(x: float, digits: int = 4):
    return round(float(x), digits) if x is not None and math.isfinite(x) else None


def swanlab_charts(summary: dict[str, Any], tensors: dict[str, Any]) -> dict[str, Any]:
    import swanlab
    from pyecharts import options as opts

    charts = swanlab.echarts
    kind = main_kind(summary)
    step = summary.get("opt_step")
    view = summary["views"][kind]["backbone"]["signal"]
    slots = [str(k) for k in summary["slot_ids"]]
    n = len(slots)
    tag = f"step {step}, {kind}, backbone, signal"

    heat = charts.HeatMap()
    heat.add_xaxis(slots)
    cos = view["cos"]
    heat.add_yaxis("cos", slots, [[j, i, _clean(cos[i][j], 3)] for i in range(n) for j in range(n)],
                   label_opts=opts.LabelOpts(is_show=True, font_size=8))
    heat.set_global_opts(title_opts=opts.TitleOpts(title="slot gradient cosine", subtitle=tag),
                         visualmap_opts=opts.VisualMapOpts(min_=-1, max_=1, range_color=["#2166ac", "#f7f7f7", "#b2182b"]),
                         xaxis_opts=opts.AxisOpts(name="slot"), yaxis_opts=opts.AxisOpts(name="slot"))

    bar = charts.Bar()
    bar.add_xaxis(slots)
    for key in ("earlier", "later", "conf", "rho"):
        bar.add_yaxis(key, [_clean(v) for v in view[key]], label_opts=opts.LabelOpts(is_show=False))
    bar.set_global_opts(title_opts=opts.TitleOpts(title="rho_k = 1 + earlier_k + later_k + conf_k", subtitle=tag),
                        xaxis_opts=opts.AxisOpts(name="slot"),
                        tooltip_opts=opts.TooltipOpts(trigger="axis"))

    fine = tensors["pair_grams"][kind]
    backbone = signal_view(sum(v for k, v in fine.items() if k not in SIDE_HEADS))
    alpha = summary["alpha"] if len(summary["labels"]) > n else None
    result = what_if(backbone, summary["slot_ids"], WHATIF_GAMMAS, alpha, summary["sample_from_anchor"])
    line = charts.Line()
    line.add_xaxis(["inf" if math.isinf(g) else f"{g:g}" for g in WHATIF_GAMMAS])
    for i, k in enumerate(summary["slot_ids"][:-1]):  # the last slot has no later slots
        line.add_yaxis(f"slot {k}", [_clean(result[str(g)]["later"][i]) for g in WHATIF_GAMMAS],
                       label_opts=opts.LabelOpts(is_show=False))
    line.set_global_opts(title_opts=opts.TitleOpts(title=f"later_k under other gamma (trained with {summary['gamma']:g})",
                                                   subtitle=tag),
                         xaxis_opts=opts.AxisOpts(name="gamma"), yaxis_opts=opts.AxisOpts(name="later_k"),
                         tooltip_opts=opts.TooltipOpts(trigger="axis"),
                         legend_opts=opts.LegendOpts(type_="scroll"))

    return {"probe_chart/cos_heatmap": swanlab.ECharts(heat),
            "probe_chart/slot_decomposition": swanlab.ECharts(bar),
            "probe_chart/whatif_later": swanlab.ECharts(line)}
