# vLLM concurrency=8 benchmark

This benchmark runs the existing **end-to-end** Xiaomi-OCR-0 pipeline against three local image sets:

- OmniDocBench
- MIDV-500
- MIDV-LAIT

It uses one vLLM server on physical GPU 1 and `--workers 8` on the client. `batch_size` is not passed to the e2e pipeline; its default is therefore unchanged. (`batch_size` belongs to the two-stage layout detector, not the vLLM request path.)

## Remote usage

From the repository root on the remote host:

```bash
bash benchmark/run_vllm_concurrency8.sh \
  /data/OmniDocBench/images \
  /data/MIDV-500/images \
  /data/MIDV-LAIT/images \
  SeerRay-Lab/Xiaomi-OCR-0 \
  /data/benchmarks/xiaomi-vllm-c8
```

Optional environment variables:

```bash
GPU_ID=1 PORT=26000 GPU_MEM_UTIL=0.80 \
  bash benchmark/run_vllm_concurrency8.sh ...
```

The script:

1. Recursively collects supported image files and writes absolute-path JSONL manifests.
2. Starts vLLM once, without `--enforce-eager`, on GPU 1.
3. Runs each dataset with `--workers 8`, `temperature=0`, `top_p=0.8`, `max_tokens=16384`, and the repository's existing repetition retry behavior.
4. Writes predictions and a lightweight `summary.json`.

The script does not download datasets or invent dataset labels. Supply the exact local roots used by the benchmark. The official quality evaluators still need to be run with the corresponding ground-truth files:

- OmniDocBench: use `evaluation_pipeline/configs/end2end.yaml` and the official `pdf_validation.py` container.
- MIDV-500 / MIDV-LAIT: run the project/benchmark evaluator using its exact field-level annotations, then retain field accuracy, precision, recall, F1, CER, WER, failure rate, empty-output rate, and truncation rate alongside the throughput summary.

The collector deliberately does not call an evaluator without explicit ground-truth paths.
