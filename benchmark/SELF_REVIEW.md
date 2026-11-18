# Benchmark 设计对照与自审

自审日期：2026-11-17。对照用户提供的《文档解析与证件识别_推理瓶颈分析方案_基线.md》。唯一约定的服务并发差异是 workers=8；每请求仍为一图/一页。

## 实现与验收映射

| 设计范围 | 实现与证据 | 自审结果 |
|---|---|---|
| §2–4 配置与框架 | launcher 保存实际 server argv/help/version；worker 保存 dtype、编译、graph、attention、cache、scheduler；server 只加载一次 | 不加 enforce-eager，不把 client workers 当实际 batch。实际后端依赖安装版本暴露的信息 |
| §3 生成边界 | 请求默认省略 max_tokens；服务使用 generation-config=vllm；记录 usage/finish_reason，调用安装版本的实际限长 helper | 不用伪无限上限，不拿 stop 代替 EOS；不支持的 helper 保留 null |
| §5 流水线 | client file/decode/resize/serialization/postprocess/write；独立 processor audit；代表请求 worker ranges、CUDA Events 与 trace | client/离线 audit/server GPU 证据分开。服务端 processor、tokenizer 阶段没有钩子时仍不可用 |
| §6 Token | usage + worker 实际 prompt IDs/mm spans + 同模型 tokenizer | 不按空白或字符算 token；输出 IDs 不同步异步 GPU；patch 和 visual tokens 分开 |
| §7 Padding | processor mask/grid/实际 shape；长度 max/mean、P95/median | 只有实际 mask 才报告对应 padding；调度 padding 和视觉内部对齐无证据则 null |
| §8、17 延迟 | cold readiness、warmup、steady；host TTFT/chunk/可验证 ITL；代表请求 GPU Events | 所有分布含 count/mean/std/min/P50/P90/P95/P99/max；不把 host decode span 当 GPU Decode |
| §9 吞吐 | 各数据集独立 wall window；requests/documents/pages/images/input/output/total/useful tokens | warmup/retry/profile 分离；phase-service rate 与设备吞吐分开；完整文档依赖 document_id |
| §10 资源 | nvidia-smi、Linux进程树/proc、可选DCGM、WSL Windows CIM、worker allocator/storage | 并发峰值只归属 run；权重/KV 实际去重；activation/input/精确碎片不做减法猜测 |
| §10 传输与同步 | 真实 CUDA memcpy/runtime/aten trace 与区间并集 | 缺方向/字节证据则 null；trace 覆盖范围明确 |
| §11 热点 | CUDA external/correlation ID 到 CPU算子/阶段，operator/kernel top50、小kernel阈值 | 算子无关联不假称 CUDA热点；嵌套阶段不重复记 kernel；CPU allocation 不当 GPU peak |
| §12 分桶 | area/output/visual/text/table/formula/type/difficulty，独立 profile bucket | 未提供标签不猜；共享 wall 吞吐明确为桶贡献；不把 run 峰值复制到每桶 |
| §13 质量 | 官方 evaluator argv适配，MIDV显式GT/提取规则；版本与样本覆盖；显式阈值比较 | ODB必须1.5；不伪造CDM/TEDS；coverage complete不等于质量达标 |
| §14–15 输出 | summary、逐样本JSONL、stage/token/padding/throughput/memory/transfer/kernel/buckets、原始trace/monitor | 每个样本包含失败；使用唯一ID，历史输出不得重用；metric_coverage记录缺失原因 |
| §16 安全边界 | 代表trace默认64 engine iterations；未知接口拒绝无界profiling；请求输出不截短 | 全请求CUDA Event与局部trace独立；只清理launcher记录的PID，无环境安装 |
| §18–20 验收与解释 | 独立结果目录支持后续并发/配置对照；13条验收与瓶颈报告 | 缺实际证据不通过；局部trace不得支持完整请求主导性结论；不把相关性当因果 |

## 修复的统计与采集问题

- 三个数据集分别结束计时，不再出现后续任务时间污染前一个数据集吞吐。
- 不再通过目录里的 Markdown 数量或空白分词推导成功率/token吞吐。
- 同名输入不覆盖；日志请求人口必须与 manifest 唯一ID完全一致。
- SDK隐式重试移除，重复输出重试单独计量并保留首轮/重试/采用版。
- 模型阶段有CPU host范围、GPU Event跨度和kernel work三种不同口径，不能互换。
- profiler先停止，随后才做一次CUDA Event最终同步；稳态请求不插入逐token同步。
- 有界profiler清理可恢复原对象，避免Kineto重复导出；局部窗口字段传入trace汇总和报告。
- 阶段验收要求同一个完整代表请求的阶段证据，不从不同样本拼出完整覆盖。
- 官方质量结果必须覆盖本轮manifest，版本一致；质量基线阈值缺失不冒报通过。

## 本地验证

- `python -m unittest discover -s tests -p test_benchmark.py -q`：41项通过。
- 对全部benchmark Python文件及该测试文件做AST解析：12个文件通过。
- 三个CLI的`--help`可运行：run_benchmark、prepare_datasets、collect_results。
- `bash -n benchmark/run_vllm_concurrency8.sh`通过。
- `git diff --check`通过。
- 未新建conda/venv，未安装torch/vLLM或其他依赖，未执行全项目测试。

## 真实环境验证边界

本机没有torch/vLLM/GPU运行环境，因此worker RPC、CUDA Graph下子模块可见性、实际profiler端点、真实token IDs、模型module映射、DCGM/WSL遥测和正式质量评测均尚未完成运行验证。mock测试不能替代这些检查。

服务器端processor/tokenizer耗时、视觉内部padding、activation/input显存归因、精确碎片、纯CUDA初始化时间和SM/Tensor Core计数器具有明确采集条件，不能保证任意vLLM/模型都能暴露。相关字段与缺失原因已经纳入覆盖表，而不是填0。

因此，代码已提供逐项采集、汇总和证据验收路径，但当前没有真实运行结果能证明所有最低标准已经满足。实际运行后必须检查各数据集`acceptance_report.json`、`metric_coverage.csv`与原始trace；存在missing_evidence时不可宣称设计全部验收通过。正式质量还需要GT、官方evaluator和同口径的基线阈值。

## 服务端 Processor 增补自审（2026-11-18）

- 增加实际multimodal apply与内部HF Processor两层计时，不将两者相加，不假设入口之外的chat template或HTTP解析已覆盖。
- 采用服务端专用import hook，覆盖标准路径Base类与模型子类，保留环境sitecustomize。已用新解释器和模拟vLLM包验证自动注入，不等于真实vLLM兼容性验证。
- 记录调用wall/线程CPU、成功/异常、实际类、签名和父子调用ID。返回值和异常保持原样；不记录任何请求内容。
- HTTP请求ID传播尚无验证，所以仅做调用级/阶段级统计。HF未调用不能断言缓存命中；未知接口或无观测为unavailable，不当0。
- 补scripts/check_benchmark_interfaces.py用于目标环境的只读版本、接口、模型配置元数据检查。无下载、安装或服务启动。
- 测试增补包括嵌套去重、异常恢复、子进程启动、已有sitecustomize保留、阶段分离和无证据非零。

本轮最终验证：46项benchmark定向测试通过（含8线程隔离）；16个Python文件AST通过；shell语法、CLI help、诊断脚本缺vLLM时的报告路径及git diff检查通过。实际GPU/vLLM运行仍未验证。

## 深度接口与GPU运行验证入口（2026-11-18）

- 环境脚本增加--deep：实际Processor注册、源码调用表达式、model runner方法及scheduler/output字段，不加载权重；移除两个不存在的写死类探测。
- 增加GPU冒烟脚本与证据汇总。只在显式运行时加载模型，8请求、32输出token、最多16个trace iteration，仅为兼容性验证。正式默认流程仍跑三个数据集且不加人为输出cap。
- worker增加loaded runner/model/sampler/视觉模块接口RPC，telemetry增加真实输入schema与输出类型分布，不同步异步结果。
- InputProcessor调用上下文仅在真实嵌套时关联request_id，测试验证离开上下文即恢复null，不将静态签名当作传播已成立。
- 本地51项定向测试通过，新增上下文、实际对象检查、缺证据不通过、钩子安装与执行证据分离、源码探测测试。
- 本机缺torch/vLLM/GPU，尚未运行真实模型冒烟检查。因此实际接口、调用链、模块与trace问题仍须以目标环境输出为准，不能将此处测试当作已通过GPU验证。
