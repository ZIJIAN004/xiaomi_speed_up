# vLLM 文档/证件 OCR 瓶颈采集

本目录按《文档解析与证件识别_推理瓶颈分析方案_基线.md》组织采集。客户端固定默认 **workers=8**，每个请求一张图像/一页。客户端 worker 数、请求图像数、vLLM 实际调度 batch 是三个不同概念，不再输出虚构的 `batch_size=16`。

## 运行

在已有 GPU/vLLM 环境中，从任何目录执行以下脚本。不会安装依赖，不下载数据，不修改集群配置。结果路径必须不存在。

```bash
bash benchmark/run_vllm_concurrency8.sh \
  /data/OmniDocBench/images /data/MIDV-500/images /data/MIDV-LAIT/images \
  /data/models/Xiaomi-OCR-0 /data/results/xiaomi-profile-new
```

默认 GPU=1、上下文=16384；显存比例按设计文档示例为 **0.90**，可用 `GPU_MEM_UTIL` 显式覆盖。`DTYPE=auto` 保留模型自身精度选择，并记录 worker 实际 dtype；也可显式固定 `DTYPE=bfloat16`。不传 `--enforce-eager`，不改变 attention/compile/chunked-prefill/prefix-cache 默认值。`--generation-config vllm` 排除模型 generation_config 中隐藏的小输出上限。默认不传请求 `max_tokens`，由服务端按真实输入和上下文边界确定；仅显式设置 `MAX_TOKENS` 才增加输出上限。

```bash
GPU_ID=1 PORT=26000 GPU_MEM_UTIL=0.90 DTYPE=auto \
WARMUP=5 REPRESENTATIVES=5 PROFILE_MAX_ITERATIONS=64 RETRY_REPETITIVE=1 \
  bash benchmark/run_vllm_concurrency8.sh ...
```

`TRUST_REMOTE_CODE=1` 是显式选择，默认不开启。不要对不可信模型启用。

脚本检查安装版本的 `--worker-extension-cls`，保存版本与完整 help，按版本能力选择 JSON profiler-config 或 legacy profiler 环境变量。所有服务绑定 127.0.0.1。worker RPC/profiler 属于开发接口，**不可暴露到公网或生产服务**。启动前检查端口，退出只结束本脚本记录的服务 PID。

## 采集顺序与计时边界

1. 生成唯一 sample_id manifest，保留相对路径和显式注解；同名文件不覆盖。
2. 启动一次 vLLM，记录启动到 HTTP readiness 的冷启动时间。该时间包括加载/初始化/compile/graph capture，不冒称纯模型加载时间。
3. 独立运行 processor 检查。记录 tokenizer、padding_side、输入 ID 长度、视觉网格/patch、tensor shape、预处理画布等。该检查不计入服务稳态阶段耗时。
4. 每个数据集预热默认 5 次，预热失败则停止，避免把失败流量当成基线。
5. workers=8 全量 streaming 请求；每样本日志包含成功与失败。分别测文件读取、图像解码、客户端 resize、PNG/base64/JSON 序列化、HTTP 请求、后处理和写盘。E2E 起点为线程开始处理样本，终点为结果写出，**不含线程池排队**；整个任务吞吐的 wall time 包含队列处理和客户端开销。
6. 同时采集原始 `/metrics`、nvidia-smi、服务端进程树与客户端进程 CPU/RSS。实际 worker 采集 host 已就绪的输入/输出 token ID、视觉 placeholder span、真实 scheduler batch 直方图。不会为了观测异步输出调用 `.cpu()`、`.item()` 或同步 GPU。
7. 稳态后单独重试重复输出；保留首轮、重试和最终采用版。稳态吞吐不包含重试，业务耗时单独记录。
8. 按图像面积、输出长度、视觉长度、表格/公式/类型/困难标签的可用覆盖贪心选择 3–5 个代表样本，**独立启停 worker profiler**。每个代表样本前尝试 reset prefix/mm/encoder cache，记录各 endpoint 的响应，避免复用视觉缓存导致 Vision Encoder 消失。它们是冷缓存诊断重跑，不冒称正常稳态 cache 行为。
9. 正式质量 evaluator（或显式字段注解适配器）与性能数据一并汇总。
10. 输出 `metric_coverage.csv` 和设计文档 §19 的 13 条 `acceptance_report.json`。有字段、有文件不等于完成采集；缺少实际证据会显示 `missing_evidence`。

## 指标定义与证据边界

### Token、停止原因与 padding

- 输入/生成/总处理 token 来自服务端 usage，不再按空白或字符估计。
- processor 独立记录 patch 数，不把 patch 无条件当作 LLM visual token。
- worker 请求 ID 与 SSE server_request_id 唯一匹配后，使用实际多模态 placeholder spans 计 visual token，并对比 worker input IDs 长度与 usage。文本/特殊 token 可由同模型 tokenizer 分类；找不到对应证据则保留 null。
- 输出可见/特殊 token 仅在 worker 已返回真实 Python token IDs 且长度与 usage 一致时分类。异步 worker 输出不为采集强制同步，缺失会明确记录。
- `stop` 不自动等于 EOS。只有真实最后一个 token 与 tokenizer EOS 相符且序列完整才有 EOS 证据。
- 有 finish_reason 的请求作为停止原因/截断率的完成分母，HTTP/处理失败保留在请求错误率分母。
- useful-output token 是**同模型 tokenizer 对接受版后处理文本的计数**，与 raw generated token 数分开。
- 不根据并发 worker 数推导 batch padding。processor attention padding 独立记录；实际 scheduler batch、视觉对齐、graph padding 没有内部证据时为 null，不擅填 0。跨样本 max/mean 和 P95/median 是长度不均衡，不是实际 padding。

### TTFT、ITL、TPOT

- `ttft_ms`：HTTP 提交到第一个可观测输出事件。`ttft_e2e_ms`：文件读取阶段起点到同一事件。纯 role/空 delta 不算首 token。
- 保存所有 chunk 间隔。只有每个输出 SSE 事件均有且仅有一个 logprob token 时，才把相邻事件间隔报告为 host-observed ITL。
- 多 token 合并 flush、无 logprobs 的文本块不伪造 ITL；只报告 chunk interval。即使 logprob 每块一条，这仍是 host 观测，不是 GPU token 完成时间。
- `host_tpot_after_first_ms=(末次输出事件−首次输出事件)/(generated_tokens−1)`，明确不是精确 GPU Decode 时间。
- 模型 TPOT 需要实际模型 Decode 证据，缺失则 null。`time_per_generated_token_ms` 使用 HTTP 请求时延/实际生成数，名称与模型 TPOT 分开。
- 所有延迟统计输出 count、mean、总体标准差、min、P50/P90/P95/P99、max。

### 模型阶段、算子、kernel 与传输

- worker profiler 在真实模型执行位置运行，而不是给 API client 包 torch.profiler。
- Vision/Projector 通过实际模型子模块 forward 范围标记。默认寻找 visual/vision_model/vision_tower 与 projector/merger 等常见真实模块路径，记录匹配路径和完整模块清单。特殊模型可显式指定：

```bash
MODULE_MAP='{"vision_encoder":"visual","projector":"visual.merger"}' \
  bash benchmark/run_vllm_concurrency8.sh ...
```

- V1 Prefill/Decode 使用 scheduler 的 prompt 长度与 computed position 分类，不用“调度一个 token 就是 Decode”的猜测。混合或未知 step 单独标记。V0 有明确 attention metadata 时使用相应字段。
- runner 范围包在真实 execute_model 外，覆盖 CUDA Graph replay。图内 kernel 到 CPU 范围的关联仍依赖 profiler 实际 external/correlation ID；无法归因就 null，不把 CPU 范围持续时间当 GPU 时间。
- 阶段结果为**关联 kernel 持续时间之和**，不是阶段 wall latency。最深阶段独占归因，避免将嵌套 Projector 重复算进 Vision。并行 kernel 的总 work time 可大于 wall time。
- `operator_top50.csv` 按关联 CUDA self time 排序，包含 CPU inclusive/self、CUDA inclusive/self、调用数、形状以及 trace 可提供的 FLOPs/内存字节。
- kernel 次数、时长统计、小 kernel 比例的默认阈值 **10 μs**，可在重新汇总时改 `--small-kernel-us`。
- kernel launch host time 是实际 CUDA runtime launch API 时间，不冒称所有 GPU launch overhead。
- H2D/D2H 来自 memcpy trace 的方向/字节数/时间；CUDA synchronize、MemcpyAsync、aten::to/cpu/item/copy_ 分别计数和计时。
- GPU active/idle 使用 GPU 事件时间区间并集，只覆盖首个到最后一个实际 GPU 事件之间；不把事件总时间直接当活跃 wall time。CPU/GPU overlap 同样使用区间并集。
- vLLM request-phase histogram 差值单独输出。tokens / 请求阶段时长总和不是并发设备 wall throughput，scope 会明确标注。

### 代表请求的计时与有界 trace

- 默认 `PROFILE_MAX_ITERATIONS=64`，只限制 profiler 的 engine iteration 窗口，**不改变请求输出长度**。优先使用已安装 vLLM 的 `ProfilerConfig.max_iterations`；旧版仅对实际 torch profiler 使用有限 schedule。未知接口拒绝无界 profiling，并保留缺失原因。
- 支持时设置 `ignore_frontend=true`，避免 frontend profiler 跟随长请求无界记录。旧版接口是否实际适配须在目标 vLLM 环境验证。
- `profile_trace_window` 记录实际 model step 数与配置上限；达到上限的 trace 按局部窗口处理。局部 Decode kernel work 不能代替完整长输出 Decode 成本，也不能单独通过全请求阶段主导性验收。
- 独立代表请求全程启用 CUDA Event，按实际 stream 记录阶段起止，不逐 token synchronize。在 `/stop_profile` 成功后做一次最终同步，再读取事件时间；只要停止失败，就不把相关事件统计当作有效阶段证据。
- CUDA Event 阶段跨度包含该 stream 上范围内空洞，**不是纯 kernel work**。同 stream 嵌套范围做独占计时；不比较不同 stream 的独立时间锚点。GPU TTFT 需要可验证的 Prefill 和末段 Prefill Sampling 事件。
- graph replay 可能绕过 Python 子模块 forward；缺阶段就保持 null，不能用整个 runner 或 host TTFT 冒充 Vision/LLM 阶段。

### 资源、显存与可选硬件计数器

- nvidia-smi 采集 GPU busy、memory utilization、用量、功耗、温度、SM/memory clock；`memory utilization` 不等于显存占用率。
- worker reset/read allocator peak；workers=8 时只报告**整个稳态窗口**的峰值，不冒称每请求峰值。代表请求可单独 reset/read，仍标为 profiled。
- 权重和 KV cache 从实际 CUDA tensor storage 去重计字节。allocated-minus-weights 不当作 activation memory，reserved-minus-allocated 不当作精确碎片。
- 保存 inactive_split_bytes、CPU RSS、HTTP OOM 次数与 worker allocator OOM 差值。显存输入/activation/pinned 归因缺证据时 null。
- 若已有本机 DCGM exporter，可设置 `DCGM_URL=http://127.0.0.1:9400/metrics`，采集匹配物理 GPU 标签的 `DCGM_FI_PROF_SM_ACTIVE` 和 `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`。不自动安装或改变 GPU profiling 配置。没有 DCGM 不伪造 SM/Tensor Core 指标。
- WSL 下若能调用 `powershell.exe`，每10秒通过 Windows CIM 采集主机总/空闲/已用内存，并记录真实观测时间；失败显式记错。不用 Linux RSS 代替 Windows 主机内存，没有主机遥测时明确不可用。Nsight Systems/Compute 仍按设计在 trace 发现空洞、传输/同步异常后补充，不默认录整个数据集。

## 数据标签与质量评价

图像根目录必须是正确的页面/帧集合。多帧 TIFF/GIF 等输入会拒绝静默只评第一页，应先显式拆页。ID 与输出文件的对应关系保存在 `prediction_mapping.json`，外部 evaluator 必须使用这个映射，不能假定原 stem 与输出名一致。

可通过 `OMNIDOC_ANNOTATIONS`、`MIDV500_ANNOTATIONS`、`MIDV_LAIT_ANNOTATIONS` 提供 JSON sidecar，key 为数据集内 POSIX 相对路径：

```json
{
  "document_a/page01.png": {
    "document_id": "document_a",
    "document_type": "identity_card",
    "has_table": false,
    "has_formula": false,
    "difficulty": ["blur"],
    "ground_truth_fields": {"name": "EXAMPLE NAME", "number": "1234"},
    "field_patterns": {"name": "^Name:\\s*(.+)$", "number": "^Number:\\s*(.+)$"}
  }
}
```

MIDV 内置显式字段适配器支持完整 JSON 输出，或用户提供的字段正则，不会用 GT 猜测提取规则。报告 micro exact-match 字段准确率、Precision/Recall/F1、按参考字符/单词总数归一化的 CER/WER，默认不修改大小写或空白。没有注解或无法提取时验收不通过，不把性能请求成功率当字段准确率。

OmniDocBench 的 Text Edit/CDM/TEDS 必须接官方 evaluator，不能以通用 edit-distance 伪造 CDM/TEDS。本仓库原本仅有评测配置，没有官方 `pdf_validation.py` 实现/GT。通过环境变量 `OMNIDOC_QUALITY_COMMAND` / `MIDV500_QUALITY_COMMAND` / `MIDV_LAIT_QUALITY_COMMAND` 指定**JSON argv 数组**，不执行 shell 字符串：

```bash
export OMNIDOC_QUALITY_COMMAND='["python","/data/tools/official_eval_adapter.py","--manifest","{manifest}","--mapping","{mapping}","--predictions","{predictions}","--output","{output}"]'
```

adapter 调用已安装的正式 evaluator，并写出如下契约。OmniDocBench 必须包含 text_edit/formula_cdm/table_teds；MIDV 必须包含 field_accuracy/precision/recall/f1/cer/wer。数值量纲与官方输出一致，记录 evaluator 版本和 GT 版本；必须报告 `evaluated_samples` 等于本轮 manifest 样本数。设计文档指定 **OmniDocBench 1.5**，验收要求 `benchmark_version="1.5"`；仓库原有 evaluation_pipeline 文档提到的 1.6 不能悄悄代替它。缺 required metric、样本覆盖不足或版本不符均为 incomplete。

```json
{
  "status": "complete",
  "source": "official_evaluator_and_version",
  "ground_truth_version": "actual_version",
  "benchmark_version": "1.5",
  "evaluated_samples": 100,
  "metrics": {"text_edit": 0.1, "formula_cdm": 0.9, "table_teds": 0.9}
}
```

这里只展示字段格式，不是实测结果。`complete` 表示注解/指标覆盖完整，不代表已达到业务质量阈值。adapter 可在同一 JSON 中提供 `baseline_thresholds`，每个必需指标对应 `{"operator":"<=","value":实际基线阈值}` 或 `{"operator":">=","value":实际基线阈值}`。例如 Text Edit/CER/WER 通常向下，CDM/TEDS/F1 通常向上，但实现不猜具体阈值或量纲。所有必需指标覆盖完整且达到显式阈值，`baseline_evaluation.status` 才为 `pass`，第13项才通过；缺阈值为 `not_evaluated`。没有 GT 和 evaluator 时，代码能采集性能，但不能宣称完成设计文档 §19 第13项。

## 每数据集输出

```text
config.json / environment.txt / manifest.jsonl
processor_config.json / processor_audit.json
per_sample_metrics.jsonl / warmup.json / retry_metrics.json
prediction_mapping.json / predictions/
metrics_before.prom / metrics_after.prom / vllm_metrics.csv
monitor/system_samples.jsonl / monitor/metrics_*.prom
worker_telemetry.json / profile_ranges.json / profile_metrics.json
profiles/<sample_id>/*.pt.trace.json[.gz] / cuda_events.json
trace_representative.json                         # 仅在有真实 worker trace 时存在
summary.json / stage_latency.csv / token_statistics.csv
padding_statistics.csv / throughput_statistics.csv / memory_statistics.csv
cpu_gpu_transfer.csv / operator_top50.csv / kernel_top50.csv / kernel_statistics.csv
bucket_by_image_area.csv / bucket_by_visual_tokens.csv / bucket_by_text_tokens.csv
bucket_by_output_tokens.csv / bucket_by_has_table.csv / bucket_by_has_formula.csv
bucket_by_document_type.csv / bucket_by_difficulty.csv
bucket_by_*_profile.csv                          # 独立诊断样本，不填稳态桶吞吐
quality.json / metric_coverage.csv / acceptance_report.json / bottleneck_report.md
```

TensorBoard 可将 `profiles/` 作为 logdir。没有真实 trace 时不制造空 trace 充数。桶吞吐明确为该桶对整个混合工作负载 wall time 的贡献，不冒称单独跑桶的吞吐；每桶 profiled 阶段与内存另看代表样本，不把整个 run 峰值复制给每个样本。

重新汇总已有数据不会启动推理：

```bash
python -m benchmark.collect_results --run-dir /data/results/xiaomi-profile-new/omnidocbench --small-kernel-us 10
```

单数据集入口可 `python -m benchmark.run_benchmark --help` 查看；`--workers` 支持后续并发扫描，默认入口保持8。改 dtype/CUDA Graph/chunked-prefill 等时启动新的结果目录，每次只改一个因素，不自动混入基线。

## 自检与验收

```bash
python -m unittest discover -s tests -p test_benchmark.py -v
bash -n benchmark/run_vllm_concurrency8.sh
```

自审对照见 `SELF_REVIEW.md`。只跑 benchmark 相关测试。无 PyTorch/vLLM/GPU 的机器可验证标准库采集、SSE、统计、质量适配、mock worker 和结果汇总；不能据此宣布真实 GPU profiler 已通过。实际环境版本/模型模块不同导致的缺口，以 acceptance_report 和原始 trace 为准，再针对安装版本适配，不根据版本号猜内部实现。

实现时参考的官方接口：
- https://github.com/vllm-project/vllm/blob/main/vllm/config/profiler.py
- https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/serve/dev/rpc/api_router.py
- https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/serve/dev/cache/api_router.py
- https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/serve/utils/api_utils.py
- https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/worker/gpu_model_runner.py
- https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/core/sched/output.py
- https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/envs.py

以上接口核对不等于对用户实际安装版本的运行验证。
