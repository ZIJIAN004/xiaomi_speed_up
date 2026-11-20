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

The first version deliberately excludes top-k reliability gating, loss truncation, sampled-token feedback, and benchmark/bottleneck instrumentation.

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

Only the ground-truth assistant answer span contributes to the loss. The prompt, image context, and chat-template boundary tokens remain context.
