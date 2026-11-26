# Per-slot gradient directions in DSpark training

DSpark trains every slot of a draft block in one forward through shared weights. It weights slot k's loss by the fixed decay `exp(-k/γ)`. A weight rescales a slot's gradient but cannot change its direction.

This folder measures, along a from-scratch training run, two things:

- whether later slots' gradients point against earlier slots';
- how much of any such conflict a different γ could remove.

Setup:

- **Target:** the original MiniCPM-OCR 1.0, not the LoRA merge.
- **Draft:** the official DSpark architecture and KIE vocabulary patch, with speculators' own random init (`--random-init`). Its lm_head is the target head rows, frozen.
- **Data:** the KIE rows of the LoRA run.

## Files

- `grad_probe.py` is the probe. It is called from `specdec/dspark/train_dspark.py` at the steps given by `--probe-steps` and `--probe-every`.
- `analyze_grad_probe.py` turns the probe outputs into CSV tables, a report and figures.
- `run_probe.sh` is the launcher. `check` runs 2 optimizer steps with probes at 0, 1 and 2. `train` is the experiment run. `analyze` runs the analysis. Outputs go to `runs/insight/<run>/`.
- `test_grad_probe.py` holds the CPU checks. It reuses the tiny fixtures of `specdec/dspark/test_dspark_pipeline.py`.

## Method

1. **Split the loss by slot.** The training loss is split exactly by slot: `L = Σ_k w_k S_k + α C`. Here `S_k` is slot k's undecayed loss and `C` is the confidence-head term.
2. **Take each term's gradient.** At each probe step, every term's gradient is taken with `torch.autograd.grad` on a fixed set of training rows (`--probe-samples`, default 64). The rows are dealt into `--probe-groups` disjoint groups (default 4).
3. **Keep Gram matrices.** For each module, the probe keeps Gram matrices of the per-group mean gradients. It keeps them in the raw geometry and in the AdamW geometry, where coordinate i is weighted by `1/(√v̂_i+ε)`.

What the Gram matrices give:

- **Signal view (cross-group products).** Unbiased for `⟨E g_a, E g_b⟩`, because the groups' noise is independent. These are the systematic directions.
- **Batch view (same-group products).** What one batch of 16 rows sees: signal plus noise/16. This matches the optimizer batch of the run.
- **Progress ratio.** `ρ_k = <c_k g_k, G>/|c_k g_k|² = 1 + earlier_k + later_k + conf_k`, where `G = Σ_b c_b g_b`. A negative `later_k` means the slots after k slow slot k's loss down, to first order.
- **Other γ values.** Re-weighting the stored undecayed gradients gives the decomposition under any other γ (`whatif.csv`) without retraining.

Every probe runs two self-checks, reported in `checks.csv` and in the log:

- the term gradients must sum to the real training gradient (cos ≈ 1);
- a small real step along the joint gradient must move each slot's loss as predicted (correlation and ratio near 1).

The probe never touches `.grad` and restores all RNG state, so the training run itself is unchanged by it.

## Cost

- **CPU RAM.** `groups × (block+1) × trainable × 4` bytes. With 4 groups this is 4 × 16 × 179.4M × 4 ≈ 46 GB.
- **GPU.** About 2 GB on top of training (gradient buffer and step-check copies). For the same reason, compiled backwards run without donated buffers while probing.

## Run

```bash
cd <repo root on the server>
bash specdec/insight/run_probe.sh check              # cost and self-checks first
bash specdec/insight/run_probe.sh train --tracker swanlab   # in tmux
bash specdec/insight/run_probe.sh analyze runs/insight/<run>/probe
python -m unittest specdec/insight/test_grad_probe.py      # CPU
```
