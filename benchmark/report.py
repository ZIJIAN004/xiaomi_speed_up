"""Evidence-only bottleneck report; missing stages block a stage-dominance conclusion."""
from __future__ import annotations

import math
from collections import defaultdict


def correlation(records, x, y):
    pairs = [(r.get(x), r.get(y)) for r in records if isinstance(r.get(x), (int, float)) and isinstance(r.get(y), (int, float))]
    if len(pairs) < 3:
        return {"sample_count": len(pairs), "pearson_r": None}
    mx = sum(a for a, _ in pairs) / len(pairs)
    my = sum(b for _, b in pairs) / len(pairs)
    xx = sum((a - mx) ** 2 for a, _ in pairs)
    yy = sum((b - my) ** 2 for _, b in pairs)
    value = sum((a - mx) * (b - my) for a, b in pairs) / math.sqrt(xx * yy) if xx and yy else None
    return {"sample_count": len(pairs), "pearson_r": value}


def write_report(path, summary, records, traces, acceptance):
    lines = ["# 瓶颈证据报告", "", "## 采集验收", "",
             f"正式样本 {summary['sample_count']}，成功 {summary['successful_samples']}，失败 {summary['failed_samples']}。",
             f"最低验收全部满足：{acceptance['all_minimum_criteria_met']}。", "",
             "本报告区分实测、trace 归因和不可观测项。相关性不等于因果，代表请求 profiler 不代表整个稳态窗口。", ""]
    for row in acceptance["criteria"]:
        lines.append(f"- {row['id']}: {row['status']} — {row['requirement']}")
    lines.extend(["", "## 请求与客户端阶段", "", "| 指标 | 样本数 | Mean ms | P50 ms | P95 ms | P99 ms |", "|---|---:|---:|---:|---:|---:|"])
    for name in ("latency_ms", "request_latency_ms", "ttft_ms", "itl_ms"):
        s = summary[name]
        lines.append(f"| {name} | {s['count']} | {s['mean']} | {s['p50']} | {s['p95']} | {s['p99']} |")
    processor = summary.get("server_processor", {})
    lines.append("服务端 Processor 采集状态：" + processor.get("status", "unavailable") + "。实际调用成本见 server_processor_statistics.csv。")
    if processor.get("unavailable_reason"):
        lines.append(processor["unavailable_reason"])
    lines.append("multimodal_apply 与内部 hf_processor 是嵌套口径，不相加；调用wall时间在8并发下可能重叠，不直接作为任务耗时占比。")
    lines.extend(["", "客户端文件读取、图像解码、resize、序列化、后处理和写盘分布见 stage_latency.csv。CPU thread time 与 wall time 分开。", "",
                  "## 模型阶段与 GPU work", ""])
    groups = defaultdict(dict)
    for t in traces:
        for stage, value in t.get("stage_kernel_time_ms", {}).items():
            if value is not None:
                groups[t["sample_id"]][stage] = groups[t["sample_id"]].get(stage, 0) + value
    if not groups:
        lines.append("没有可归因的模型阶段 GPU 数据，不能判断 Vision/Prefill/Decode 主导关系。先检查 profile_ranges.json、profile_metrics.json 和 worker trace。")
    for sample_id, stages in sorted(groups.items()):
        complete = all(t.get("trace_covers_full_request", False) for t in traces if t.get("sample_id") == sample_id)
        lines.append(f"- {sample_id} trace 范围：" + ("完整请求" if complete else "有界局部窗口，不代表完整输出"))
        ranking = sorted(stages.items(), key=lambda item: item[1], reverse=True)
        lines.append(f"- {sample_id}: " + ", ".join(f"{name}={value:.4f} ms kernel work" for name, value in ranking))
        if not complete or not all(s in stages for s in ("vision_encoder", "prefill", "decode")):
            lines.append("  阶段覆盖不完整，不作全流水线主导结论。")
    lines.append("完整代表请求的 CUDA Event 阶段跨度与 GPU TTFT 见 stage_latency.csv/profile_metrics.json；同 stream 嵌套范围独占计时，与局部 kernel work 分开，不混入稳态吞吐。")
    lines.extend(["", "kernel work 是关联 kernel 时长之和，不是阶段 wall latency；深层阶段独占归因。代表请求已尝试清缓存，因此不能把其阶段比例直接外推到存在缓存命中的稳态服务。", "",
                  "## 长度与图像规模关联", ""])
    good = [r for r in records if r.get("success")]
    for x, y in (("generated_tokens", "end_to_end_latency_ms"), ("image_megapixels", "ttft_ms"),
                 ("visual_tokens", "ttft_ms"), ("total_input_tokens", "ttft_ms")):
        item = correlation(good, x, y)
        lines.append(f"- {x} / {y}: n={item['sample_count']}, Pearson r={item['pearson_r']}。")
    lines.extend(["", "这些数值只描述此工作负载中的关联。需结合长度分桶、真实阶段归因和单因素对照，才能判断输出长度或图像规模是否造成瓶颈。", "",
                  "## 调度、内存与底层热点", "",
                  "- 调度/排队/运行/等待/preemption、KV cache 与 prefix-cache 指标以 vllm_metrics.csv 的原指标名、labels 和窗口差值为证据。",
                  "- CPU/GPU 利用率、功耗、温度与时钟保存在 monitor/scalars.zip，分阶段在线摘要见 monitor/monitor_summary.json；没有 SM/Tensor Core 计数器时不作相关硬件饱和判断。",
                  "- 显存峰值为稳态窗口级，代表请求的 profiled 峰值单独保存；不把整个窗口峰值复制成各请求峰值。",
                  "- operator_top50.csv 和 kernel_top50.csv 分别定位算子与 kernel 热点；注意 kernel_attribution_ratio 与未归因 work。",
                  "- cpu_gpu_transfer.csv 给出传输和同步证据。GPU 活跃/空闲与 CPU/GPU overlap 的范围仅为实际 trace 事件窗口。",
                  "- 如存在明显 GPU 空洞、同步或传输异常，再用 Nsight Systems 验证；不能仅凭低 nvidia-smi 利用率认定 CPU 瓶颈。", "",
                  "## 质量与剩余证据", "",
                  f"质量状态：{summary['quality'].get('status')}；缺失指标：{summary['quality'].get('missing_metrics')}。",
                  "complete 仅表示质量指标和样本覆盖完整，不代表达到业务基线阈值。",
                  f"显式基线阈值验证：{summary['quality'].get('baseline_evaluation', {}).get('status')}。", "",
                  "缺失项及原因以 metric_coverage.csv 为准。任何 missing_evidence 项补齐之前，不应宣称完整通过设计文档验收。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
