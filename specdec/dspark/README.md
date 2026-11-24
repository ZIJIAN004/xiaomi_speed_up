# MiniCPM-OCR 1.0 DSpark draft training

This directory retrains the DSpark draft for a fine-tuned MiniCPM-OCR 1.0. It starts from the official `openbmb/MiniCPM-OCR-1.0-DSpark` weights and trains on IDSpace ground truth. The output is in the official checkpoint layout, so vLLM 0.34 serves it unchanged:

```bash
--speculative-config '{"method":"dspark","model":"<run>/best","num_speculative_tokens":15}'
```

The environment is `envs/setup_minicpm_ocr_env.sh`: vLLM 0.34.0, transformers 5.22, and speculators installed with `--no-deps` at commit 8f993e3.

## Pipeline

```text
IDSpace train.jsonl + image (serving resize: longest side ≤ 2048), user turn = [image, prompt]
    ↓  MiniCPM processor, native chat template, enable_thinking=False
Frozen fine-tuned MiniCPM-OCR (HF, bf16, batch size 1)
    ├── forward hooks: output of layers[k-1] for k in [3,7,11,15,19]   → hidden_states [1,T,5·1024]
    ├── pre-hook on the final norm: pre-norm last hidden state         → verifier_last_hidden_states
    └── pre-hook on the text model: position_ids must be 0..T-1 (as vLLM feeds MiniCPM-OCR)
    ↓
speculators DSparkDraftModel (5 layers, block 15, draft vocab 114688), warm-started from the official draft
    DFlash anchored blocks + Markov head + confidence head
    targets = verifier_lm_head(verifier_norm(last hidden)) = the target's own logits on the draft vocab
    loss    = 0.1·CE + 0.9·TV (per-slot decay exp(−k/4)) + 1.0·BCE(confidence, 1 − TV)
    ↓
best/ and last/ in the official layout (same keys, shapes, dtypes and config.json)
```

### Why each piece looks like this

- **Aux features = vLLM's.** vLLM records aux layer k as the residual stream after k decoder layers (`_maybe_add_hidden_state(aux, layer_idx+1, hidden, residual)`). In HF that is the output of `layers[k-1]`. The hooks read exactly that tensor, so `output_hidden_states` conventions never matter.
- **Targets come from the target itself.** `verifier_norm` is a frozen copy of the target's own final norm, which uses the Qwen3.5 `(1 + w)` convention. `verifier_lm_head` holds the rows of the target head at the draft's 114688 target ids. These follow the fine-tuned target automatically.
- **Draft RoPE is full width.** The official draft uses `rope_dim 256` and `theta 1e7`, while the target uses partial RoPE (0.25). `load_official_draft` fails if the built rotary embedding does not match.
- **Vocabulary.** `d2t` is delta-encoded (`target = i + d2t[i]`) in the checkpoint, in speculators and in vLLM. `t2d` is derived from it, and the official draft vocabulary is kept.
- **Loss mask.** With `sample_from_anchor=True`, slot k of the block anchored at p predicts token p+k+1 from the target distribution at p+k. speculators masks that slot with `loss_mask[p+k]`. So `loss_mask[i]` marks positions whose *next* token is an answer token or the closing `<|im_end|>`. That is the assistant mask shifted left by one; `test_dspark_pipeline.py` checks this.
- **Frozen tensors.** These are the embedding (vLLM shares the target's), the verifier head and norm, and, as in speculators, the draft `lm_head`. `check_alignment.py` reports whether the official `lm_head` equals the target head rows. If it does, train with `--lm-head-init verifier`, so the draft head follows the fine-tuned target. `--train-lm-head` unfreezes it.

### Hyperparameters

Defaults come from speculators. From `TrainerConfig`: AdamW, weight decay 0.01, linear schedule with 1% warmup, gradient clipping 1.0, bf16 autocast over fp32 master weights. From `examples/train/dspark_qwen3_0_6b_sharegpt_online.sh`: lr 3e-4, 5 epochs, loss `{"ce":0.1,"tv":0.9}`, confidence α 1.0, decay γ 4.0.

Two values are adapted to batch-size-one multimodal samples:

- **`--max-anchors 256`.** It is fixed so the compiled draft keeps one shape and covers a full IDSpace answer. `anchors_truncated_frac` in the logs shows how often an answer had more anchor positions than this.
- **`--grad-accum 16`.** Sixteen answers give about the ~3k anchors of one speculators packed batch (`max_anchors 3072`).

Validation reports DSpark's own metrics:

- **`eal`**: greedy accepted length per block, including the bonus token. This is vLLM's mean acceptance length at temperature 0.
- **`accept_len`**: expectation under rejection sampling.
- **`position_k_acc`**, `accept_rate`, and confidence calibration.

`best/` is written only by an epoch whose validation `eal` (`--select-metric`) beats the initial draft on the same target. A missing `best/` means: keep the official draft.

## Files

- `minicpm_dspark.py` covers official config to speculators config, official weights in and out, verifier attachment, the target feature hooks, and metric reduction.
- `idspace_minicpm_dataset.py` renders IDSpace samples as served (prompt, image→text order, resize) and builds the shifted loss mask. It reuses the EAGLE-3 IDSpace loader and seeded split.
- `train_dspark.py` trains, evaluates (`--eval-only`) and checkpoints. Resume with `--resume <run>/last/trainer_state.pt`.
- `check_alignment.py` is the gate run before training (below).
- `run_dspark.sh` is the launcher for `check`, `eval`, `train` and `smoke`. It uses GPU 1 by default, refuses GPU 0, keeps WSL alive and logs to `runs/dspark/<run>/`.
- `test_dspark_pipeline.py` holds CPU checks on tiny configs: vocab maps, bit-exact load/save round trip, hooks against manual residuals, the position guard, a forward/backward step that touches only trainable tensors, and the loss mask.

## Order of runs

```bash
cd ~/speed_up   # repo root on the server
PY=/home/mr_jiang/miniforge3/envs/minicpm_ocr/bin/python

# 0. CPU unit checks
$PY -m unittest specdec/dspark/test_dspark_pipeline.py

# 1. Alignment gate: ORIGINAL model + official draft on IDSpace validation.jsonl
bash specdec/dspark/run_dspark.sh check
#    passes when: the round trip is bit-identical, base and full model features agree,
#    rebuilt target argmax agrees ≥ 99.9%, and positions are sequential.
#    Then compare its val/eal with vLLM on the same validation rows:
#    summary.json -> speculative_decoding.mean_acceptance_length (also counts the bonus token)
PYTHONPATH=$PWD $PY baseline/idspace/scripts/evaluate_vllm.py --split validation --gpu 1 \
  --model /home/mr_jiang/models/minicpm-ocr-1.0/model --output runs/dspark/vllm_official_draft \
  --speculative-config '{"method":"dspark","model":"/home/mr_jiang/models/minicpm-ocr-1.0/dspark","num_speculative_tokens":15}'

# 2. Smoke test: 2 optimizer steps (compile, memory, checkpoint layout)
bash specdec/dspark/run_dspark.sh smoke

# 3. Official draft on the fine-tuned target (how much did fine-tuning cost?)
FT_MODEL=/home/mr_jiang/models/minicpm-ocr-1.0-ft/model bash specdec/dspark/run_dspark.sh eval

# 4. Train (wrap in tmux; add --lm-head-init verifier if step 1 says so)
FT_MODEL=... bash specdec/dspark/run_dspark.sh train --wandb-mode offline

# 5. Serve <run>/best with vLLM and measure acceptance and throughput with the benchmark harness.
```

## Known limits

- **Off-policy gate.** The gate uses IDSpace ground truth as the answer. The original model's own outputs differ in format, so offline `eal` and vLLM's acceptance length are expected to be close, not identical. After fine-tuning, the target's outputs approach the ground truth and the gap should shrink. Training on the target's own greedy outputs is the speculators default (response regeneration). It would need an answers file, which is not implemented here.
- **Prompt must match serving.** The default is the IDSpace prompt from `specdec/eagle3/idspace_gt_dataset.py`. If the fine-tuned model is served with another prompt or image order, pass `--prompt-file` / `--text-first`. The default order is image then text, as in the official MiniCPM-OCR vLLM examples and ms-swift fine-tuning data.
- **CUDA only.** Training needs CUDA: flex attention and the fused Triton losses are compiled. `--no-compile` is for debugging only. The unit tests run on CPU through the eager attention and eager loss paths.
- **Not yet run on the server.** As of this commit nothing has been run on the server: not the unit tests, not the gate, not a training run.
