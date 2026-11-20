# IDSpace decoding data collection

This directory collects Xiaomi-OCR-0 inference data for later EAGLE3 decoding-SFT. It deliberately does **not** run the repository bottleneck instrumentation, profiling, warmup, or benchmark aggregation.

The input is the prepared IDSpace `train.jsonl` format already used by `idspace/scripts/evaluate_vllm.py`, with at least:

```json
{"image": "relative/path.png", "target": {...}}
```

The collector writes one JSON object per line containing the source row, prompt, raw model response, actual token IDs when the Transformers backend exposes them, and timing.

## Direct Transformers backend

Use an existing Python environment that already has the Xiaomi-OCR-0 Transformers dependencies. Do not run from the downloaded model directory.

```powershell
python "C:\Users\zijia\Desktop\Diligram\xiaomi_speed_up\Eagle3\decoding_SFT\collect_idspace.py" `
  --model "C:\models\Xiaomi-OCR-0" `
  --input-jsonl "C:\data\IDSpace\prepared\train.jsonl" `
  --image-root "C:\data\IDSpace\prepared\images" `
  --output-jsonl "C:\data\IDSpace\runs\xiaomi_train_inference.jsonl" `
  --trust-remote-code
```

For a first local smoke run, add `--max-samples 2` and use a new output file.

## OpenAI-compatible vLLM/SGLang backend

This path sends direct inference requests to an already-running local server and still skips all benchmark instrumentation:

```powershell
python "C:\Users\zijia\Desktop\Diligram\xiaomi_speed_up\Eagle3\decoding_SFT\collect_idspace.py" `
  --backend openai `
  --base-url "http://127.0.0.1:8000/v1" `
  --model "SeerRay-Lab/Xiaomi-OCR-0" `
  --input-jsonl "C:\data\IDSpace\prepared\train.jsonl" `
  --image-root "C:\data\IDSpace\prepared\images" `
  --output-jsonl "C:\data\IDSpace\runs\xiaomi_train_inference.jsonl"
```

The script uses deterministic decoding (`temperature=0`, `top_p=0.8`) to match the repository's IDSpace inference settings. It does not modify the input JSONL or create a benchmark manifest.
