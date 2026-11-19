# 文档解析任务评测流程

本目录提供 Xiaomi-OCR-0 的端到端文档解析路径：整页图片直接推理得到页面级 Markdown。正式 benchmark 使用 `benchmark/` 中记录的 OmniDocBench v1.5 官方评分器。英文说明见 [README.md](README.md)。

## 环境版本

评测环境参考 [OmniDocBench 的 Environment Setup and Running 说明](https://github.com/opendatalab/OmniDocBench/blob/main/README.md#environment-setup-and-running)，主要组件如下：

| Component | Version |
| --- | --- |
| Python | 3.10.x |
| TeX Live | 2025 |
| pdflatex | 3.141592653-2.6-1.40.28 |
| ImageMagick | 7.1.1-47 |
| Ghostscript | 9.55.0 |

评测使用的 Docker 镜像来自：

```text
ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204
```

GPU 推理需要单独准备模型与 vLLM 环境。具体安装见 [INSTALL.md](../INSTALL.md)。


## 输入数据格式

以下命令均在仓库根目录运行。输入为 JSONL 文件，每行包含一张页面图片的路径：

```json
{"images": ["/absolute/path/to/page.png"]}
```

图片路径需能被推理进程访问。输出按 `<图片文件名>.md` 命名，因此输入图片的文件名应唯一，并与评测标注中的页面名称对应。评测标注需使用 OmniDocBench 支持的 JSON 格式。

按实际位置设置输入文件和模型路径：

```bash
MODEL=SeerRay-Lab/Xiaomi-OCR-0
INPUT_JSONL=/absolute/path/to/images.jsonl
OUTPUT_DIR="$PWD/outputs"
GPUS=0,1,2,3,4,5,6,7
mkdir -p "$OUTPUT_DIR"
```

## vLLM 推理超参数

端到端主推理使用以下生成参数：

| 参数 | 数值 |
| --- | --- |
| `temperature` | `0.0` |
| `top_p` | `0.8` |
| `presence_penalty` | `0.0` |
| `repetition_penalty` | `1.0` |
| `max_tokens` | `16384` |
| `chat_template_kwargs.enable_thinking` | `false` |
| 请求超时 | `600 s` |



## 端到端推理

使用 [`e2e_img2md.py`](e2e_img2md.py) 生成逐页 Markdown：

```bash
python -m evaluation_pipeline.e2e_img2md \
  --input-jsonl "$INPUT_JSONL" \
  --output_dir "$OUTPUT_DIR/e2e" \
  --model_path "$MODEL" --served_model_name Xiaomi-OCR-0 \
  --task document --gpu_devices "$GPUS" --port 26000 \
  --gpu_mem_util 0.8 --workers 8 \
  --presence_penalty 0.0 --repetition_penalty 1.0 \
  --max_tokens 16384 --request_timeout 600 \
  --retry-repetitive --retry-presence-penalty 1.5 \
  --start_server --stop_server
```

后处理会将 OTSL 转成最终 Markdown 中的 HTML 表格，评测使用最终的逐页 `.md` 文件。

## 评测

正式评测使用 OmniDocBench v1.5 官方 `pdf_validation.py`。该配置包含以下指标：

| 类别 | 指标 |
| --- | --- |
| 文本 | Edit Distance |
| 公式 | Edit Distance、CDM |
| 表格 | TEDS、Edit Distance |
| 阅读顺序 | Edit Distance |

匹配方式为 `quick_match`；matching、CDM、TEDS 的 worker 数均为 `16`；`quick_match_truncated_timeout_sec` 和 `match_timeout_sec` 均为 `1200`；`timeout_fallback_max_chunk_span=200`，`timeout_fallback_order_penalty=0.05`。复现时保留这些设置。配置中的默认数据路径是示例路径，下面的命令在临时副本中替换为实际挂载路径。

将 `GT` 设置为标注 JSON，`PRED_DIR` 设置为待评测的 Markdown 目录。

```bash
GT=/absolute/path/to/ground_truth.json
PRED_DIR="$OUTPUT_DIR/e2e"  # 两阶段输出改为 "$OUTPUT_DIR/stage2"
RESULT_DIR="$PWD/evaluation_results"
mkdir -p "$RESULT_DIR"
docker pull ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204

docker run --rm --entrypoint bash \
  -v "$PWD/evaluation_pipeline/configs/end2end.yaml:/repro/end2end.yaml:ro" \
  -v "$GT:/repro/gt.json:ro" \
  -v "$PRED_DIR:/repro/predictions:ro" \
  -v "$RESULT_DIR:/workspace/result" \
  -w /workspace \
  ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204 -c '
set -euo pipefail
python - <<"PY_CONFIG"
import yaml
with open("/repro/end2end.yaml") as f:
    config = yaml.safe_load(f)
dataset = config["end2end_eval"]["dataset"]
dataset["ground_truth"]["data_path"] = "/repro/gt.json"
dataset["prediction"]["data_path"] = "/repro/predictions"
with open("/tmp/xiaomi-ocr-eval.yaml", "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY_CONFIG
python pdf_validation.py --config /tmp/xiaomi-ocr-eval.yaml
'
```

## Real5-OmniDocBench：复现结果参考

| 方法 | 参数量 | Overall ↑ | 扫描 ↑ | 弯曲 ↑ | 屏幕拍摄 ↑ | 光照 ↑ | 倾斜 ↑ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| e2e | 0.8B | 93.45 | 94.524 | 93.401 | 93.279 | 93.858 | 92.170 |
| 两阶段 | 0.8B | 95.24 | 96.41 | 95.46 | 94.47 | 95.39 | 94.46 |

Overall 为五个子集成绩的算术平均。两阶段推理使用 PaddleX 版 PP-DocLayoutV3，阈值为 0.25。

## OmniDocBench v1.6：复现结果参考

| Method | overall↑ | text_block_Edit_dist↓ | display_formula_CDM↑ | display_formula_Edit_dist↓ | table_TEDS↑ | table_TEDS_structure_only↑ | table_Edit_dist↓ | reading_order_Edit_dist↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| e2e | 95.1926 | 0.0341 | 96.4553 | 0.0937 | 92.5325 | 94.805 | 0.0707 | 0.1301 |
| 两阶段 | 96.8277 | 0.0313 | 98.5044 | 0.0765 | 95.1087 | 97.1929 | 0.0483 | 0.1221 |

## Wild-OmniDocBench：复现结果参考

| Method | overall↑ | text_block_Edit_dist↓ | display_formula_CDM↑ | display_formula_Edit_dist↓ | table_TEDS↑ | table_TEDS_structure_only↑ | table_Edit_dist↓ | reading_order_Edit_dist↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| e2e | 86.4806 | 0.1125 | 90.0734 | 0.1475 | 80.6184 | 84.3165 | 0.1658 | 0.1927 |
| 两阶段 | 87.9438 | 0.1233 | 89.5485 | 0.1568 | 86.6129 | 90.9422 | 0.1104 | 0.1931 |
