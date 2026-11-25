# MiniCPM-OCR 1.0 DSpark draft training

This directory retrains the DSpark draft for a fine-tuned MiniCPM-OCR 1.0. It starts from the official `openbmb/MiniCPM-OCR-1.0-DSpark` weights and trains on IDSpace ground truth for the identity-card KIE stage, rendered exactly as the MiniCPM-OCR LoRA was trained. The output is in the official checkpoint layout, so vLLM 0.34 serves it unchanged:

```bash
--speculative-config '{"method":"dspark","model":"<run>/best","num_speculative_tokens":15}'
```

The environment is `envs/setup_minicpm_ocr_env.sh`: vLLM 0.34.0, transformers 5.22, and speculators installed with `--no-deps` at commit 8f993e3.

## Pipeline

```text
LoRA data run: train.jsonl (all 54029 rows, reshuffled every epoch) / validation.jsonl (6011, held out) + layout.jsonl
    ↓  crop = union of layout blocks + 3% margin, ≤ 262144 px; user turn = [image, 16-field prompt]
    ↓  MiniCPM processor, max_slice_nums=1, native chat template, enable_thinking=False
Frozen fine-tuned MiniCPM-OCR = base + LoRA merged (HF, bf16, batch size 1)
    ├── forward hooks: output of layers[k-1] for k in [3,7,11,15,19]   → hidden_states [1,T,5·1024]
    ├── pre-hook on the final norm: pre-norm last hidden state         → verifier_last_hidden_states
    └── pre-hook on the text model: position_ids must be 0..T-1 (as vLLM feeds MiniCPM-OCR)
    ↓
speculators DSparkDraftModel (5 layers, block 15, draft vocab 114688), warm-started from the official draft,
draft vocabulary patched for the KIE answers (same size; build_kie_draft_vocab.py)
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
- **Task = the LoRA's KIE stage.** `kie_task.py` mirrors the LoRA run's `common.py`: the same 16-field prompt, union-of-blocks crop plus a 3% margin, 262144-pixel budget, and `max_slice_nums=1`. The message structure mirrors its `train.py` collate. A draft trained on other inputs would see hidden states the served target never produces. `--task idspace_legacy` keeps the earlier full-image, 12-field setup.
- **Vocabulary.** `d2t` is delta-encoded (`target = i + d2t[i]`) in the checkpoint, in speculators and in vLLM. `t2d` is derived from it. The official vocabulary lacks tokens that almost every compact-JSON answer uses: `":"`, `","` and pieces of field names. A draft cannot propose those, so each one ends the accepted run. `build_kie_draft_vocab.py` re-points the official vocabulary's rarest tail ids (skipping any token the answers use) at the missing answer tokens, most frequent first. The size stays 114688, so config.json, every shape and vLLM's loader are unchanged. Only those rows change: d2t/t2d, the `lm_head` row (set to the target head row of the new token) and the `markov_w2` row (zeroed). `markov_w1` and the embedding are indexed by target id and stay as they are. Its report gives the answer-token coverage before and after, on train and validation.
- **Merged target.** `merge_lora.py` adds `(α/r)·B·A` to each base tensor in the safetensors files, so key names, dtypes and config.json stay the base's (HF and vLLM load it like the base). `--verify-samples` compares teacher-forced predictions of merged and PEFT on KIE validation samples.
- **Loss mask.** With `sample_from_anchor=True`, slot k of the block anchored at p predicts token p+k+1 from the target distribution at p+k. speculators masks that slot with `loss_mask[p+k]`. So `loss_mask[i]` marks positions whose *next* token is an answer token or the closing `<|im_end|>`. That is the assistant mask shifted left by one; `test_dspark_pipeline.py` checks this.
- **Frozen tensors.** These are the embedding (vLLM shares the target's), the verifier head and norm, and, as in speculators, the draft `lm_head`. `check_alignment.py` reports whether the official `lm_head` equals the target head rows. If it does, train with `--lm-head-init verifier`, so the draft head follows the fine-tuned target. `--train-lm-head` unfreezes it.

### Hyperparameters

Defaults come from speculators. From `TrainerConfig`: AdamW, weight decay 0.01, linear schedule with 1% warmup, gradient clipping 1.0, bf16 autocast over fp32 master weights. From `examples/train/dspark_qwen3_0_6b_sharegpt_online.sh`: lr 3e-4, loss `{"ce":0.1,"tv":0.9}`, confidence α 1.0, decay γ 4.0. The example trains 5 epochs on ShareGPT; here the default is 3 (`--epochs`), because the draft is warm-started and IDSpace answers share one fixed JSON schema. If validation `eal` still rises in epoch 3, rerun with `--epochs 5` (the linear schedule depends on the total).

Two values are adapted to batch-size-one multimodal samples:

- **`--max-anchors 256`.** It is fixed so the compiled draft keeps one shape and covers a full IDSpace answer. `anchors_truncated_frac` in the logs shows how often an answer had more anchor positions than this.
- **`--grad-accum 16`.** Sixteen answers give about the ~3k anchors of one speculators packed batch (`max_anchors 3072`).

Validation reports DSpark's own metrics:

- **`eal`**: greedy accepted length per block, including the bonus token. This is vLLM's mean acceptance length at temperature 0.
- **`accept_len`**: expectation under rejection sampling.
- **`position_k_acc`**, `accept_rate`, and confidence calibration.

An epoch counts as better only if validation `eal` (`--select-metric`) rises **and** validation loss falls, both against the best checkpoint so far; the first comparison is with the initial draft on the same target (`--no-require-loss-decrease` drops the loss condition). Each better epoch is saved to `epoch_XX/` and to `best/` (always the latest better one); `last/` is the end of the most recent epoch, with `trainer_state.pt` for `--resume`. A missing `best/` means: keep the official draft.

## Files

- `minicpm_dspark.py` covers official config to speculators config, official weights in and out, the vocabulary patch, verifier attachment, the target feature hooks, and metric reduction.
- `kie_task.py` holds the KIE prompt, crop and pixel budget of the LoRA run.
- `idspace_minicpm_dataset.py` renders samples (KIE crop or legacy full image, image→text order) and builds the shifted loss mask. It reuses the EAGLE-3 IDSpace loader. `--train-samples N` trains on a seeded random subset, recorded in `data_split.json`.
- `merge_lora.py` merges the LoRA into the base checkpoint and verifies the merge against PEFT.
- `build_kie_draft_vocab.py` writes the vocabulary patch (`--draft-vocab`) and its coverage report.
- `train_dspark.py` trains, evaluates (`--eval-only`) and checkpoints. Resume with `--resume <run>/last/trainer_state.pt`.
- `check_alignment.py` is the gate run before training (below).
- `run_dspark.sh` is the launcher for `merge`, `vocab`, `check`, `eval`, `train`, `smoke` and `lengths`. It uses GPU 1 by default, refuses GPU 0, keeps WSL alive and logs to `runs/dspark/<run>/`.
- `measure_lengths.py` reports token lengths from the processor only: sequence, image and answer tokens, anchor positions against `--max-anchors`, and the size an offline feature cache would take (the shared prefix stops at the first image token).
- `test_dspark_pipeline.py` holds CPU checks on tiny configs: vocab maps, bit-exact load/save round trip, hooks against manual residuals, the position guard, a forward/backward step that touches only trainable tensors, the loss mask, the KIE crop, the training subset, the vocabulary patch (rows, save, reload) and the LoRA merge.

## Order of runs

```bash
cd ~/speed_up   # repo root on the server
PY=/home/mr_jiang/miniforge3/envs/minicpm_ocr/bin/python

# 0. CPU unit checks
$PY -m unittest specdec/dspark/test_dspark_pipeline.py

# 0a. Merge the LoRA (base + adapter -> $LORA_DIR/merged), verified against PEFT on 32 KIE samples
bash specdec/dspark/run_dspark.sh merge
# 0b. Vocabulary patch -> runs/dspark/kie_draft_vocab/draft_vocab.{pt,json}; check its coverage numbers
bash specdec/dspark/run_dspark.sh vocab

# 1. Alignment gate: ORIGINAL model + official draft on the KIE validation.jsonl
bash specdec/dspark/run_dspark.sh check
#    passes when: the round trip is bit-identical, base and full model features agree,
#    rebuilt target argmax agrees ≥ 99.9%, and positions are sequential.
#    Then compare its val/eal with vLLM on the same validation rows:
#    summary.json -> speculative_decoding.mean_acceptance_length (also counts the bonus token)
PYTHONPATH=$PWD $PY baseline/idspace/scripts/evaluate_vllm.py --split validation --gpu 1 \
  --model /home/mr_jiang/models/minicpm-ocr-1.0/model --output runs/dspark/vllm_official_draft \
  --speculative-config '{"method":"dspark","model":"/home/mr_jiang/models/minicpm-ocr-1.0/dspark","num_speculative_tokens":15}'

# 2. Data lengths (CPU), then a smoke test: 2 optimizer steps (compile, memory, checkpoint layout).
#    smoke passes --profile-stages, so its log prints ms/sample for target forward, draft forward
#    and draft backward. The flag is off by default because it adds CUDA syncs.
bash specdec/dspark/run_dspark.sh lengths
bash specdec/dspark/run_dspark.sh smoke

# 3. Official draft (unpatched) on the merged fine-tuned target (how much did fine-tuning cost?)
bash specdec/dspark/run_dspark.sh eval

# 4. Train: all train rows, patched vocabulary (wrap in tmux; add --lm-head-init verifier
#    if step 1 says so; TRAIN_SAMPLES=N for a random subset, DRAFT_VOCAB= for the official vocabulary)
bash specdec/dspark/run_dspark.sh train --epochs 10 --wandb-mode offline

# 5. Serve <run>/best with vLLM and measure acceptance and throughput with the benchmark harness.
```

## Known limits

- **Serve with `max_slice_nums=1`.** The LoRA was trained and evaluated with one image slice, while the processor default is 9. vLLM must get the same setting (`mm_processor_kwargs`) and the same crop, or both the target and the draft see other inputs.
- **One draft for both stages.** If one vLLM server also serves the stage-1 block recognition (short `Text Recognition:` outputs), it uses this draft there too. This training covers only the KIE stage.

- **Off-policy gate.** The gate uses IDSpace ground truth as the answer. The original model's own outputs differ in format, so offline `eal` and vLLM's acceptance length are expected to be close, not identical. After fine-tuning, the target's outputs approach the ground truth and the gap should shrink. Training on the target's own greedy outputs is the speculators default (response regeneration). It would need an answers file, which is not implemented here.
- **Prompt must match serving.** The default is the 16-field KIE prompt of the LoRA run (`kie_task.py`). If the fine-tuned model is served with another prompt or image order, pass `--prompt-file` / `--text-first`. The default order is image then text, as in the LoRA training data and the official MiniCPM-OCR vLLM examples.
- **CUDA only.** Training needs CUDA: flex attention and the fused Triton losses are compiled. `--no-compile` is for debugging only. The unit tests run on CPU through the eager attention and eager loss paths.
- **Not yet run on the server.** As of this commit nothing has been run on the server: not the unit tests, not the gate, not a training run.
