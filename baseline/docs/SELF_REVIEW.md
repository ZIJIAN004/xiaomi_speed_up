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
- `bash -n baseline/harness/run_vllm_concurrency8.sh`通过。
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

## 就绪探测诊断修正（2026-11-18）

- 确认原launcher将所有HTTP探测异常静默忽略，导致服务已打印startup完成时仍可能等待完整deadline。
- 抽出benchmark.readiness，分别记录health/models结果，实时显示进度，失败原因和每轮证据落盘。401/403及health成功后的models 404快速失败；连接拒绝保留原始异常类别与原因，按真实单调时钟deadline重试。
- 未将任何特定环境根因当作已确认。公开vLLM鉴权实现区分/health和/v1路径，支持将二者分开检查；这不证明当前机器发生了鉴权失败。
- 本地55项定向测试通过，新增health/API双条件、HTTP401/403/404、网络失败deadline、已记录PID退出分支；没有实际远程HTTP/GPU验证。

## 完整服务启动现场取证（2026-11-18）

- 增加startup-only入口：保留原服务配置完成模型/API启动，跳过manifest、离线processor audit和所有推理请求。
- 诊断专用import hook跟踪Uvicorn Server及vLLM NoSignalServer/serve_http实际异步函数的进入、返回和异常。记录真实socket监听状态，不将startup日志文字当作监听成功。
- Linux/proc证据限制在launcher记录PID及其子进程；最终网络快照在cleanup之前。缺/proc或ss会明确记录不可用。
- 加入20秒线程/异步栈与应用startup完成后30秒诊断grace。基线默认行为不启用HTTP取证或额外grace。
- 本地60项定向测试通过，新增真实本地socket监听状态转换、原异常透传、proc归属解析、取消与异常区分、grace与关闭前取证测试。没有运行远程模型，不能声称连接拒绝已修复。

## 单卡离线迁移自审（2026-11-18）

- 默认入口改为run_offline_concurrency8.sh；一个AsyncLLM实例跨三个数据集复用。TP/PP/DP显式为1，并核对实际parallel_config。8个在途请求由原线程池持续补充，不按8请求整批等待，不改max_num_seqs/compile/cache等原调度默认值。
- 生成改为原生异步DELTA流，通过唯一request_id关联worker。真实prompt/output IDs用于token计数和输出分类；不依赖文本字符、logprobs或未同步的worker输出推断。合并delta不伪造逐token ITL。
- 保留原生模板及enable_thinking=False；tokenize=False后交给引擎并显式add_special_tokens=False，避免重复模板特殊token。独立processor audit与真实worker输入长度仍逐样本对比，记录padding_side；max_model_len保持16384。
- SamplingParams显式max_tokens=None，不落入默认16-token上限。已核对vLLM v0.32.0 InputProcessor源码：None会按实际prompt_len计算剩余上下文。正式不加输出cap；冒烟仍为32。
- worker collective_rpc、cache reset、start/stop profile均走Python API。保留有界trace、CUDA Event、scheduler telemetry、真实storage显存、系统采样、正式质量和原13条验收逻辑。启动检查要求恰好一个可观测GPU worker。
- Prometheus使用与上游相同的进程内registry/多进程registry逻辑，不启动HTTP exporter。CPU/RSS注明客户端和引擎frontend共用PID，不将重叠口径相加。
- HTTP/SSE/Base64/JSON专属指标不发生，不填0；TTFT、冷启动与输出间隔更换明确口径，并输出metric_migration.json。旧HTTP入口显式backend=http，保留但不再作为默认冒烟路径。
- 自审测试发现并修复argparse带连字符位置参数的属性名错误；补三个数据集只初始化一次引擎的回归测试。
- 60项原benchmark测试和13项离线测试通过。新增测试覆盖真实ID/DELTA、合并输出、终止事件、原生模板/特殊token、失败abort、单loop并发、非HTTP RPC/profiler/metrics、预测写盘、无worker输出时分类、三数据集共享引擎和完整离线报告生成。
- 本机未安装torch/vLLM，无GPU实测环境；没有安装依赖/创建环境。真实Qwen3.5模型的离线接口、图像处理、GPU阶段覆盖和profiler导出仍须由目标机器冒烟结果确认，不能由mock测试宣布通过。


## Trace 单副本存储自审（2026-11-18）

- profile采集只引用profiler_raw原文件，不再复制到每个代表样本目录；使用相对数据集目录的路径以支持完整结果目录跨主机搬迁。
- 汇总不再将代表traceEvents重序列化为大JSON，只写trace_representative.ref.json；原文件元数据与gzip格式不变。旧绝对路径仍支持读取。
- legacy exporter只执行一次Kineto导出；再次请求文件名时迁移同一payload，gzip格式转换成功后删除本次源文件。模拟磁盘转换失败验证原文件保留、半成品清理，不覆盖其他已有artifact。没有删除用户现有诊断数据。
- 测试验证profile目录没有trace复制件、gzip完整运行目录搬迁后仍能分析并保留字节内容、代表引用不生成第二份payload、旧绝对路径兼容及legacy单副本导出恢复。
- 62项benchmark和13项offline定向测试通过；本地没有torch/vLLM/GPU，真实worker导出流程未实测。本轮不修改监控采样、不缩短64/16步窗口、不增加磁盘容量上限，因此不能据此保证永不写满磁盘。

## 监控低写盘与标量保留自审（2026-11-18）

- 默认1秒绝对deadline采样，采样耗时不再额外叠加到周期。慢轮次跳过错过的时隙，不补采突发；保留实际开始/结束和耗时。阶段epoch检测跨阶段及离开后返回，正式统计仍按完整steady窗口过滤。
- 资源、系统内存、原生Prometheus和可选DCGM数值用float64列存RAM，缺测NaN，新增series回填缺测；不保留每轮详细进程树字典。Welford在线统计按阶段分开，缺测计数由该阶段总数减有效数，全部缺测列仍可见。
- 首轮及之后每60秒原子覆盖一个在线汇总；每种实际观测阶段保存首轮完整快照和最后一次完整快照，不再逐轮创建JSONL/Prometheus/DCGM文件。短阶段不保证被观测，不伪造边界快照。
- 正常结束一次性写不压缩标量ZIP，各列含标签、阶段、字节序元数据，保留精确采样分位数及可重算证据；按列读取统计，避免把全部列展开为Python字典。不在推理热路径压缩或排序。硬崩溃只能保留最近在线汇总，不能恢复丢失数组的精确分位数。
- 原before/after计数器reset判定、直方图差值、逐请求JSONL、预测文件、Processor调用、profiler 64/16窗口、模型/模板/tokenization/padding/max_length均不改。旧JSONL和逐轮Prometheus结果仍支持汇总。没有引入任何GPU同步。
- 本地429列×3600次追加微测平均0.557ms/次，单次汇总4.47ms，最终11.83MiB归档41.97ms。这只测存储，不含进程扫描/nvidia-smi/真实推理，不是远程吞吐回归证明。
- 回归测试发现旧stream.write残留，已修正。测试覆盖精确P99、动态列缺测、Welford统计、阶段窗口/epoch、计数器reset/直方图均值、旧格式兼容、少量文件、正常退出、慢采样deadline、可选DCGM与空采样归档。75项原benchmark/offline测试和11项监控测试通过。
- 本地没有torch/vLLM/GPU，真实目标机吞吐影响未验证；没有安装依赖或环境，没有修改/清理benchmark/dia及ANALYSIS_FLOW.md。本改动不保证profiler、预测和Processor日志永不写满磁盘。

## 预测去重与评价后文本筛选自审（2026-11-18）

- 原始输出改为一个加锁追加的raw_outputs.jsonl，不再每次创建.raw.txt；记录sample_id/phase/attempt键及共享存储格式。已收到文本先存储，再做后处理，异常文本也有证据。后处理Markdown只写一次，最终采用版本移动到predictions，不再copy2；采用重试后重写retry_metrics的路径。
- 保持完整数据集批量评价，正式吞吐窗口不包含评价/筛选。collect_run先验证评价状态/覆盖/必需指标/版本并完成token和trace汇总，再按留存策略清理新运行目录内显式记录的文件。未评价/失败/不完整/版本错误不清理；质量低于阈值不豁免清理规则。
- 默认固定种子随机10条、最多低分10条，加所有异常和选定代表样本并集，保留这些样本各phase/attempt的文本。低分使用明确方向的实际逐样本指标，缺少时标记不可用。内置字段评价新增真实逐样本Precision/Recall/F1/CER/WER/字段准确率，不改变总体微平均公式。
- 筛选只删除未选全文，不丢分数或指标行。活动路径清空，历史路径和清理状态显式记录；output_retention.json及summary说明删除数量、原因和重新评价限制。重新collect_run沿用quality.json并保留留存状态，不重新读已删除预测。
- 自审发现collect_run会给profile_metrics添加trace_analysis，不能用清理前内存对象覆盖；已改为读取分析后的profile_metrics再更新留存元数据。评分/监控/profiler标量证据均保留。
- 新增11项输出存储测试：200条8线程Unicode/多行JSONL、采用重试移动不复制、7种保留全量条件、低分方向、固定种子、失败/截断/重复/代表保留、指标和quality字节不变、越界路径拒绝、坏JSON不删除Markdown、已清理映射明确报错、profile分析保留，以及20请求端到端评分→清理→重新汇总。
- 定向测试合计97项通过（原benchmark/offline 75项、监控11项、输出存储11项），两个launcher通过bash语法检查，git diff --check无空白错误。20请求端到端测试采用mock离线引擎和真实字段评价，清理后保留3个代表样本的正式预测，20条逐样本分数及请求指标仍完整。
- 没有修改模型、模板、tokenization、padding_side、max_length、采样超参、8并发或profiler 64/16窗口。没有torch/vLLM/GPU实测，真实目标机吞吐未验证。未清理历史dia目录或改ANALYSIS_FLOW.md；评价前仍需全量暂存，临时筛选文件短暂占额外空间，无磁盘配额保证。

## OmniDocBench v1.5 两阶段启动入口自审（2026-11-18）

- 读取ANALYSIS_FLOW.md并核对Xiaomi公开evaluation_pipeline/README；确认官方当前使用v1.6，不能把原整页offline入口冒充两阶段。本轮新增omnibench/scripts/run_omnidocbench_v15.sh和专用编排，数据地址留空，模型默认/home/mr_jiang/models/xiaomi-ocr-0/model，GPU 1，layout batch8/threshold.25，VLM offline workers8。
- 版本依据为opendatalab/OmniDocBench的实际v1_5分支pdf_validation.py、task/end2end_run_eval.py、metrics/cal_metric.py和README。脚本只调用用户已有的干净v1_5评价器，核对HEAD/ref并记录commit；不安装、下载或切换评价器，不使用v1.6配置。v1_5默认CDM worker数与新版README不同，不冒称强制16。
- GT驱动检查1355条页面、GT重复名、图片缺失/多余、stem唯一性；Python定量报告计数/覆盖率。PaddleX在独立子进程结束后再加载VLM，单卡可见性映射为物理1→逻辑0。版本字段写根目录config/environment/quality。
- 区域crop复用官方helper，在原分辨率页上裁剪后限长边2048，不写裁剪文件；text/table/formula prompt逐区域传入原生模板、Processor audit与实际离线请求，避免沿用整页prompt。enable_thinking=False、tokenize=False、引擎add_special_tokens=False不变；padding_side仍实际记录；MAX_MODEL_LEN16384不改，MAX_TOKENS为空仍用剩余上下文，不自动设16k输出cap。
- 区域输出仅strip，页面阶段复用官方OTSL转换和merge_page_to_markdown，按原图片stem命名，不把哈希region ID交给页面评价器。保留实际request IDs、usage/token IDs、TTFT、有限profiler、实际Processor、scheduler、显存、资源采样和新标量/留存逻辑。
- request_unit=region，num_pages=null，区域请求吞吐不再标成pages/s，桶内serial pages/s也不伪报；单独增加区域数量分桶、crop时间、主区域全部成功页面的共享稳态完成率和页面合并/span统计。整条pipeline elapsed明确包含启动/warmup/profile，不冒充稳态时间。
- 实际调用CDM而非CDM_plain，解析官方all层及0–1指标，Overall按v1.5定义换算0–100。自审测试发现并修正all嵌套遗漏。评价覆盖来自官方per-page Edit输出并集，不能由预测文件数推断；缺指标/覆盖不足/失败保留全文并非零退出。
- 全部分数和组件级metric证据保留。页面选中集合传播到region留存，使用明确parent-page覆盖口径，不伪称region有独立质量分数。仅sampled且完整评价后去掉评价器GT/pred文本副本和本次CDM渲染件；评分不完整和all模式不清理。
- 新增11项定向测试覆盖数据计数、覆盖失败、默认1355校验、官方裁剪/prompt、layout精确路径、页面合并/失败区域、真实分数schema与单位、评分覆盖证据、v1_5固定/脏仓库拒绝、本地评价命令和真实CDM配置、评价器渲染/文本清理，以及6页完整编排。外部layout/evaluator和GPU全部mock，不执行下载代码或真实模型。
- 75项原benchmark/offline、22项监控/留存及11项两阶段测试共108项通过；新入口和两个既有launcher通过bash语法检查。没有安装环境、没有访问远程主机、没有修改ANALYSIS_FLOW.md或历史dia。实际PaddleX/vLLM/TeX/CDM兼容性、真实模型质量和吞吐仍未验证。


## 2026-11-18 端到端路线调整

本节取代此前两阶段专用入口描述。按用户确认，run_omnidocbench_v15与对应shell入口改为完整页面推理，不执行PaddleX或区域生成，移除LAYOUT_MODEL_DIR选项。此前区域辅助函数保留但新入口不导入、不调用。

自审核对：一页一个主请求、8在途、一次离线引擎、原图片stem命名且移动不复制、采用重试路径同步到全部指标人口、评分前保留全量文本、v1_5评分commit记录、真实官方结果覆盖、失败或不完整评分非零退出并保留全部输出。质量报告接入inference/collect_run，瓶颈报告不再错误显示未评分；根目录保留原始官方归一化质量报告。冷启动/稳态/重试/profile/评分计时分离。

新增整页流程mock回归：完整40×20图像不裁剪、整页prompt、6页官方命名与页面吞吐、只构建一次引擎、评分后留存，以及评分器异常时6页全保留并返回错误。既有测试仍检查缺CDM、真实覆盖、错版本和外部评价命令。未执行真实vLLM、GPU、PaddleX或TeX，不声称运行环境已验证。用户的ANALYSIS_FLOW.md和dia/未修改。


## Docker评分适配收尾

默认使用官方v1.5环境镜像并挂载干净v1_5源码，记录镜像ID和commit，数据只读、结果独立落盘，不自动拉取镜像。实际TeX/CDM自检置于模型加载前，CDM显式worker异常不误报完整评分。同步修正旧监控路径，按GT映射文档类型/表格/公式分桶，补全低分筛选的Edit指标方向；这些标签不作为模型提示词。受控路径禁止被共享参数覆盖。按用户要求仅做runtime定向测试、一次模拟整页主流程和Shell语法检查，不运行全套或真实GPU评测。本地没有验证真实Docker/TeX；由独立smoke脚本在目标机验证。
