# IDSpace decoding data collection

This collector reuses the existing IDSpace inference contract from `idspace/scripts/evaluate_vllm.py` and the local vLLM/processor path from `benchmark/offline_backend.py`:

- same IDSpace `train.jsonl` row format;
- same `image` + `image-root` resolution;
- same identity-card prompt;
- native `AutoProcessor.apply_chat_template` with `enable_thinking=False`;
- local vLLM `AsyncLLM`, no HTTP and no remote host.

It intentionally removes the benchmark-only pieces: bottleneck monitor, profiler, warmup, representative selection, quality scoring, retry/retention, and benchmark metrics.

## PowerShell command

The model must be a **local directory** containing `config.json`; the script refuses a remote model name.

```powershell
python "C:\Users\zijia\Desktop\Diligram\xiaomi_speed_up\Eagle3\decoding_SFT\collect_idspace.py" `
  --model "C:\models\Xiaomi-OCR-0" `
  --input-jsonl "C:\data\IDSpace\prepared\train.jsonl" `
  --image-root "C:\data\IDSpace\prepared\images" `
  --output-jsonl "C:\data\IDSpace\runs\xiaomi_train_inference.jsonl" `
  --gpu 0 `
  --max-samples 2
```

Remove `--max-samples 2` after the smoke run. Add `--trust-remote-code` only if the local checkpoint requires it. The script uses one local vLLM engine, not the repository benchmark runner, and writes one JSON object per sample. It does not use the OpenAI client, HTTP, Transformers `generate`, or the benchmark bottleneck pipeline.

Default generation settings match the existing IDSpace path:

```text
temperature=0
top_p=0.8
repetition_penalty=1.0
max_tokens=None
```
