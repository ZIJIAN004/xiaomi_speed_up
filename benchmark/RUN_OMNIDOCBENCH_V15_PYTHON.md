# OmniDocBench v1.5 端到端评测启动说明

本流程在远程主机上使用本地 Python 评分环境，不依赖 Docker 和 PaddleX。

## 运行路径

```text
项目：/home/mr_jiang/speed_up/Xiaomi-OCR-0
数据：/home/mr_jiang/speed_up/Xiaomi-OCR-0/benchmark/data
评分器：/home/mr_jiang/speed_up/Xiaomi-OCR-0/evaluation_pipeline/OmniDocBench
模型：/home/mr_jiang/models/xiaomi-ocr-0/model
评分环境：/home/mr_jiang/miniforge3/envs/omnidocbench_v15
TeX Live：/home/mr_jiang/tools/texlive
```

数据目录应包含：

```text
benchmark/data/
├── images/
└── OmniDocBench.json
```

评分器必须是官方 `v1_5` 分支，且工作区保持 clean。

## 环境自检

先同步项目代码并激活评分环境：

```bash
cd /home/mr_jiang/speed_up/Xiaomi-OCR-0
git pull

conda activate omnidocbench_v15
export PATH=/home/mr_jiang/tools/texlive/bin/x86_64-linux:$PATH
```

运行最小评分环境和数据自检：

```bash
OMNIDOC_EVAL_PYTHON=/home/mr_jiang/miniforge3/envs/omnidocbench_v15/bin/python \
bash scripts/test_omnidocbench_python.sh
```

自检必须输出：

```text
PASS: Python evaluator, TeX/CDM runtime, and dataset preflight
```

自检内容包括：

- 官方 OmniDocBench v1.5 evaluator commit 和 clean worktree；
- evaluator Python import；
- `xelatex`、`magick`、`gs`、`node`；
- CDM 所需 TeX 宏包；
- `Source Han Sans SC` 字体；
- GT 与图片覆盖；
- 真实 CDM 同公式渲染、bbox 和 matching，要求 F1 ≥ 0.99。

自检失败时不要启动模型推理，先修复对应环境问题。

## 正式启动

正式流程使用两个环境，但只激活推理环境：

```text
当前 shell 的 speed_up 环境：模型、Transformers、vLLM、GPU 推理
OMNIDOC_EVAL_PYTHON 指定的 omnidocbench_v15 环境：官方评分器、CDM、TEDS
TEXLIVE_ROOT 指定的用户 TeX Live：xelatex、kpsewhich、TeX 宏包
```

不要激活 `omnidocbench_v15` 后直接启动整个脚本，否则模型推理会缺少 `transformers`。推荐使用 tmux，避免 SSH 断开导致任务停止：

```bash
tmux new -s omnidocbench_v15
```

在 tmux 中执行下面的完整命令：

```bash
cd /home/mr_jiang/speed_up/Xiaomi-OCR-0

git pull

conda activate speed_up

# 可选但推荐：显式固定模型推理解释器；评分解释器由 OMNIDOC_EVAL_PYTHON 单独指定
export SPEED_UP_PYTHON=/home/mr_jiang/miniforge3/envs/speed_up/bin/python

export OMNIDOC_ROOT=/home/mr_jiang/speed_up/Xiaomi-OCR-0/benchmark/data
export OMNIDOC_EVALUATOR_ROOT=/home/mr_jiang/speed_up/Xiaomi-OCR-0/evaluation_pipeline/OmniDocBench
export OMNIDOC_EVAL_BACKEND=python
export OMNIDOC_EVAL_PYTHON=/home/mr_jiang/miniforge3/envs/omnidocbench_v15/bin/python
export TEXLIVE_ROOT=/home/mr_jiang/tools/texlive
export MODEL_PATH=/home/mr_jiang/models/xiaomi-ocr-0/model
export GPU_ID=1

# 只做启动前环境/数据/CDM 自检，不加载模型
bash scripts/test_omnidocbench_python.sh

# 正式端到端评测：整页推理，不使用 PaddleX
bash scripts/run_omnidocbench_v15.sh
```

自检必须先输出：

```text
PASS: Python evaluator, TeX/CDM runtime, and dataset preflight
```

自检通过后，正式脚本会再次在模型加载前执行 evaluator v1_5 和 CDM preflight。两个 preflight 都通过后才会加载 Xiaomi-OCR-0。

如果自检失败，不要运行正式评测；先保留错误输出和结果目录，不要删除证据。

正式脚本运行期间不要在同一个 GPU 上启动第二个模型任务。

退出 tmux 但保留任务：

```text
Ctrl-b d
```

重新连接：

```bash
tmux attach -t omnidocbench_v15
```

## 运行后查看结果

正式运行会生成新的时间戳目录：

```bash
find /home/mr_jiang/speed_up/Xiaomi-OCR-0/benchmark/results \
  -maxdepth 1 -type d -name 'omnidocbench_v15_*' -print
```

假设本次目录为：

```bash
RESULT=/home/mr_jiang/speed_up/Xiaomi-OCR-0/benchmark/results/omnidocbench_v15_TIMESTAMP
```

检查最终质量状态：

```bash
cat "$RESULT/quality.json"
```

必须重点确认：

```text
status=complete
benchmark_version=1.5
evaluator_version=1.5
sample_coverage_complete=true
prediction_coverage_complete=true
```

检查 evaluator 实际输出目录：

```bash
find "$RESULT/evaluator_work/result" -maxdepth 2 -type f -print
```

日志中出现 Pillow 的：

```text
Palette images with Transparency expressed as bytes should be converted to RGBA images
```

只是 warning。只要 `quality.json` 为 `complete`、必需指标存在且覆盖完整，就不影响本次评分结论。

## 运行模式

当前命令使用：

```text
端到端整页推理
本地 Python evaluator
OmniDocBench v1.5
单卡离线 AsyncLLM
8 个页面请求在途
不使用 PaddleX
不使用 PP-DocLayoutV3
不使用 Docker
```

`OMNIDOC_EVAL_BACKEND=python` 必须显式设置。脚本默认 backend 是 Docker；远程主机没有 Docker 时不能省略该变量。

## 关键默认配置

```text
GPU_ID=1
GPU_MEM_UTIL=0.90
MAX_MODEL_LEN=16384
DTYPE=auto
WARMUP=5
REPRESENTATIVES=5
PROFILE_MAX_ITERATIONS=64
MONITOR_INTERVAL=1
REQUEST_TIMEOUT=600
RETRY_REPETITIVE=1
```

如需调整，使用环境变量覆盖，例如：

```bash
GPU_ID=1 \
WARMUP=5 \
REPRESENTATIVES=5 \
PROFILE_MAX_ITERATIONS=64 \
OMNIDOC_EVAL_BACKEND=python \
OMNIDOC_EVAL_PYTHON=/home/mr_jiang/miniforge3/envs/omnidocbench_v15/bin/python \
bash scripts/run_omnidocbench_v15.sh
```

不要把 runtime smoke 的 `max_tokens=32` 配置带入正式评测。正式入口未显式设置 `MAX_TOKENS` 时，使用当前正式脚本的剩余上下文策略。

## 结果目录

结果写入：

```text
/home/mr_jiang/speed_up/Xiaomi-OCR-0/benchmark/results/omnidocbench_v15_TIMESTAMP/
```

重点文件：

```text
quality.json
summary.json
bottleneck_report.md
config.json
environment.txt
quality_evaluator.log
evaluator_config.json
evaluator_invocation.json
```

正式评分必须检查：

```bash
cat benchmark/results/omnidocbench_v15_TIMESTAMP/quality.json
```

合格状态应包括：

```json
{
  "status": "complete",
  "benchmark_version": "1.5",
  "evaluator_version": "1.5",
  "sample_coverage_complete": true,
  "prediction_coverage_complete": true
}
```

实际结果目录名以本次运行生成的时间戳为准。

## 评分失败处理

若正式评分失败、缺少必需指标或样本覆盖不完整：

- 入口应返回非零状态；
- 预测文本应保留；
- 不要把 F1=0 自动解释为模型质量为零；
- 先查看 `quality_evaluator.log` 和 `quality.json`；
- 不要删除结果目录后重跑，先保留失败证据。

## 运行边界

本流程只做 Xiaomi-OCR-0 的整页端到端评测。它不评估：

- PaddleX 版面检测；
- PP-DocLayoutV3；
- two-stage region OCR；
- Docker evaluator 环境；
- 其他数据集。

`benchmark/dia/` 是本地诊断生成物，不属于正式提交内容。
