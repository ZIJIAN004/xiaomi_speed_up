# OmniDocBench v1.5 vLLM 瓶颈分析流程

## 目标

在固定 GPU、模型和采样配置下，使用 Xiaomi-OCR-0 官方文档解析路径，比较 vLLM `concurrency=8` 时的吞吐、延迟、资源占用、服务端调度和最终解析质量，并定位瓶颈。

本轮只做 **OmniDocBench v1.5**，不混用 v1.6、MIDV-500 或 MIDV-LAIT。

## 关键特殊性

### 整页端到端文档解析

本实验直接将每张页面图片交给 Xiaomi-OCR-0，生成一份页面级 Markdown：

```text
整页图片
  → Xiaomi-OCR-0 整页推理
  → 页面级 Markdown
  → OmniDocBench v1.5 evaluator
```

当前正式评测不执行 PaddleX 版面检测、区域裁剪、区域 OCR 或页面拼接。最终评分对象是每页 Markdown。

### 版本必须固定

```text
数据：OmniDocBench v1.5
GT：OmniDocBench.json（与 v1.5 图片严格对应）
evaluator：OmniDocBench v1.5 evaluator
```

不能使用项目当前 README 中的 v1.6 配置或结果替代 v1.5。结果的 `config.json`、`environment.txt` 和 `quality.json` 必须显式记录 `benchmark_version=1.5`。

### Batch 和并发不是同一个量

```text
page workers=8：最多 8 个页面请求同时在途
vLLM scheduler batch：由 vLLM 根据 running/waiting 请求、KV cache 和 token 调度实际形成，不能由 workers=8 推断为 batch=8
```

客户端 `workers=8` 是本轮并发窗口，实际 scheduler batch、padding 和 KV cache 状态必须来自服务端证据。

## 实验步骤

### 1. 固定并记录环境

记录：

- 物理 GPU（本轮 GPU 1）
- 模型路径、模型 revision 和 served model name
- vLLM、PyTorch、Transformers、CUDA、驱动版本
- dtype、`gpu_memory_utilization`、`max_model_len`
- attention backend、CUDA Graph 实际状态、chunked prefill/prefix cache 实际配置
- `temperature=0`、`top_p=0.8`、presence/repetition penalty
- page workers、max tokens

### 2. 准备 v1.5 数据

确认目录包含：

```text
OmniDocBench_v1.5/
├── images/
└── OmniDocBench.json
```

定量检查图片数、GT 记录数、图片与 `image_path` 的覆盖关系。输入 manifest 只包含 v1.5 评测页面，不把 PDF、示例图或其他版本文件混入。

### 3. 整页端到端推理

当前正式流程直接将每张页面图片交给 Xiaomi-OCR-0，生成一份页面级 Markdown，不执行 PaddleX 版面检测、区域裁切或区域拼接。

运行时记录：

- 每个 page 请求的开始/结束、成功/失败和 finish reason
- prompt/input、generated/total token（优先使用 vLLM usage，禁止用字符数冒充 token）
- TTFT、ITL/TPOT（只有 streaming 证据充分时才填写）
- 图片预处理、processor、HTTP、后处理和写盘耗时
- vLLM `/metrics` 中的 running/waiting、KV cache、token throughput、request latency 和 preemption
- GPU 利用率、显存、功耗、温度、时钟、CPU/RSS

`workers=8` 是客户端并发窗口，不是实际 vLLM batch。实际 scheduler batch、padding 和 KV cache 状态必须来自服务端证据。

### 4. 预热、稳态和 profile 分开

- 服务启动/模型加载/compile/graph capture：单独记录 cold start
- 每个数据集预热 3–5 次：不进入稳态统计
- 正式样本：记录全量请求级 metrics
- 代表样本：选择不同图像面积、输出长度和文档类型，独立启用有限 worker profiler
- profiler trace 不与普通稳态 latency、throughput 混合

### 5. 质量评分

将整页推理生成的页面 Markdown 交给 **OmniDocBench v1.5 end-to-end evaluator**，报告：

- Text Edit Distance
- Formula Edit Distance
- Formula CDM
- Table TEDS
- Table Edit Distance
- Reading Order Edit Distance
- Overall

质量结果必须记录 evaluator 版本、GT 版本和实际评测样本覆盖数。没有 v1.5 GT/evaluator 时，只能报告性能采集，不能宣称完成质量评测。

## 瓶颈分析输出

至少按数据集生成：

```text
config.json
environment.txt
manifest.jsonl
page_manifest.jsonl
per_sample_metrics.jsonl
vllm_metrics.csv
gpu_metrics.csv
latency_statistics.csv
token_statistics.csv
throughput_statistics.csv
memory_statistics.csv
bucket_by_image_area.csv
bucket_by_output_tokens.csv
profile_metrics.json
operator_top50.csv
kernel_statistics.csv
quality.json
bottleneck_report.md
```

分析重点：

```text
CPU 高、GPU 低       → 文件读取/解码/processor/调度可能是瓶颈
waiting 持续增长     → concurrency=8 超过服务调度或 KV cache 能力
GPU 高、decode 占比高 → 输出长度/LLM decode 可能主导
vision work 占比高   → 图像面积/视觉编码可能主导
P95/P99 远高于 P50   → 长图、长输出、区域数量或 scheduler 长尾
OOM/preemption 增加  → 显存、KV cache 或 max_model_len 受限
```

所有延迟和吞吐同时报告 `mean/std/min/P50/P90/P95/P99/max`。平均值用于总体水平，P95/P99 和按图像面积、区域数量、输出长度分桶用于定位长尾瓶颈。
