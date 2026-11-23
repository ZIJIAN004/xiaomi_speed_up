# Xiaomi-EAGLE decoding-SFT

This directory is for the official ground-truth EAGLE-3 training pipeline on IDSpace. It does not pre-collect model-generated answers.

## Fixed training design

```text
IDSpace train.jsonl + image
    ↓
Xiaomi-OCR-0 processor with official target answer
    ↓
Frozen Xiaomi-OCR-0 target forward
    ├── selected target hidden states → feature fusion
    └── target logits → teacher distribution
    ↓
New EAGLE draft block, multi-step teacher-forced rollout
    ↓
New draft LM head, shared by draft training and draft inference
    ↓
Token-level soft cross-entropy, weighted by 0.8^step
```

Top-k gating is disabled by default; `--top-k` enables k=3, and `--top-k 5` selects k=5. A failed check masks subsequent rollout losses at that position, not the current loss. Tokens remain ground-truth teacher-forced, with no sampled-token feedback.

## Planned files

- `inspect_idspace_target.py` — target/processor alignment preflight.
- `idspace_gt_dataset.py` — IDSpace ground-truth multimodal dataset and assistant loss span.
- `qwen_eagle_draft.py` — new EAGLE draft feature network and full-vocabulary draft head.
- `train_eagle_idspace.py` — frozen-target training loop and multi-step distillation loss.

## Draft head

For Xiaomi-OCR-0 hidden size `H=1024` and vocabulary size `V=248320`:

```text
hidden_states[1], hidden_states[12], hidden_states[24]
    → new Linear(3H, H) feature fusion

feature H + frozen target token embedding H
    → 2H draft-attention input
    → new Qwen3.5-style-compatible draft attention
    → new H-dimensional draft state
    → new Qwen-style MLP/RMSNorm block
    → new draft LM head H → V
```

The draft LM head is a new parameter set. It is used by both training and draft inference. The original Xiaomi target LM head remains frozen and is used only for the target teacher distribution and later speculative verification.

## Training

Training uses the official IDSpace `target` as the assistant ground-truth answer. It does not first generate a pseudo-answer with vLLM. The target model runs teacher-forced forward at training time to provide hidden states and logits. Token embeddings are looked up from the frozen target embedding table at training time.

The first loss is:

```text
L = Σᵢ 0.8ⁱ · CE(target_distribution, draft_logitsᵢ)
```

Only the ground-truth assistant answer span and its following `<|im_end|>` contribute to the loss, so the draft learns to propose end-of-turn. The prompt, image context, and other chat-template tokens remain context.

Optimization follows official EAGLE-3 `ds_config.json`: peak lr 5e-5, DeepSpeed WarmupDecayLR (log warmup over `--warmup-ratio` 1.5% of updates, then linear decay to 0), AdamW betas (0.9, 0.95), weight decay 0, gradient clipping 0.5, gradient accumulation 2. Draft parameters are FP32 master weights; the draft forward runs under bfloat16 autocast. Very short smoke runs spend most updates in warmup (the first update has lr 0). Masked CE is averaged over batch × sequence, matching official EAGLE-3 reduction.

## Validation, metrics and checkpoints

- **Split.** `--val-fraction` (default 0.05) rows of `--train-jsonl` are held out by a fixed `--seed` (default 42) shuffle. The indices are written to `<output-dir>/data_split.json` and verified on every restart. `build_draft_vocab.py` uses the same split with the same defaults, so validation answers never shape the draft vocabulary.
- **Metrics.** Each sample logs official-style `train/ploss_i` and `train/acc_i` for rollout depth i. `acc_i` is the draft's greedy token compared with the target's greedy token in target-id space; targets outside the draft vocabulary count as misses. Each epoch logs `train_epoch/*` and `val/*` (`val/acc_i`, `val/ploss_i`, `val/loss`). It also logs `val/est_accept_len` = Σₖ Πᵢ≤ₖ accᵢ, a teacher-forced estimate of accepted draft tokens per round, excluding the target's bonus token.
- **Checkpoints.** Each epoch atomically writes `last.pt`. When `val/est_accept_len` improves, it also writes `best.pt`. Add `--keep-epoch-checkpoints` to keep `epoch_N.pt` as well. A checkpoint holds the draft weights (including `draft_ids`/`t2d`), the `draft_config` needed to rebuild the draft, the optimizer, the scheduler, every RNG state and the args. `--resume last.pt` continues at the next epoch with the identical schedule and shuffle order.

## Reduced draft vocabulary (official EAGLE-3 d2t/t2d)

A full 248320-way draft head costs about as much as the target head, which is a large share of a 0.8B target step. `build_draft_vocab.py` counts the tokenized ground-truth answers (the exact loss tokens) and picks the smallest frequency-ranked vocabulary reaching `--coverage` (default 0.999), rounded to 128 and capped at 32000. It also reports coverage on the validation split.

```bash
python Eagle3/decoding_SFT/build_draft_vocab.py \
  --train-jsonl <train.jsonl> --tokenizer <model dir> --output-dir <dir>
python Eagle3/decoding_SFT/train_eagle_idspace.py ... --draft-vocab <dir>/draft_vocab.pt
```

With `--draft-vocab`, the draft head outputs only `draft_ids`. The teacher is `softmax(target_logits[..., draft_ids])`, and positions whose full-vocab teacher argmax is outside the draft vocabulary are masked, as in the official `t2d` position mask. Full logits are reduced once per sample and freed before the rollout. `draft_ids` and `t2d` are saved in the checkpoint for inference-time id mapping. W&B logs `train/teacher_argmax_in_draft_vocab`.

## Native multimodal context and training cache

The target language-model pre-hook captures its actual post-vision `inputs_embeds` and native `position_ids` in one target forward. Missing native context is an error, not a silent fallback. Draft RoPE is copied from the loaded target rotary module, including its checkpoint configuration and partial rotary dimension. Its floating buffers retain their original precision when draft weights are cast. Four-plane positions drop only the leading text plane for rotary computation.

TTT round r combines feature[t] with the complete multimodal embedding at t+r+1 and supervises teacher logits[t+r+1], which predict token[t+r+2]. Embeddings, teacher logits, and assistant masks shift consistently; visual embeddings are real captured values, not text-table placeholder lookups. The training cache anchors dense causal attention on round 0, then adds same-column keys/values from rounds 1 through the current round. Rotary positions are native base positions + rollout depth. This differentiable cache is not an inference token cache; variable sequence length within a rollout is rejected.

Batch size is one. Processor tensors retain their native ranks. Tokenization uses the rendered native chat template without adding special tokens again, with no padding or truncation. No padded tokens are accepted, so tokenizer padding side does not change this path. No arbitrary max_length is imposed; model context capacity is checked before the target forward.

## Focused CPU checks

```bash
python Eagle3/decoding_SFT/train_eagle_idspace.py --toy
python Eagle3/decoding_SFT/train_eagle_idspace.py --toy --top-k
python Eagle3/decoding_SFT/test_eagle_training.py
```

Toy mode needs PyTorch, not a target checkpoint or Transformers. The tests cover explicit attention-edge equivalence, multi-axis positions, partial RoPE, four-plane conversion, sequence-length guards, rollout alignment, cumulative gating, native context capture, fusion gradients, and frequency precision across dtype casts. Set `QWEN35_SOURCE` to a native `modeling_qwen3_5.py` source file to additionally run its extracted rotary class as a source oracle; this does not validate an installed checkpoint end to end.
