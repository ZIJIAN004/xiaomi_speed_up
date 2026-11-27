# Block-routed draft vocabulary for DSpark

A DSpark round drafts a block of 15 tokens. Its two vocabulary-sized pieces are
the draft `lm_head` (1024 → 114688) and the Markov `W2` (256 → 114688). Together
they are about 60 % of the draft's per-round multiply-adds and of its weight reads.
Idea: a learned router looks at the block once and picks a few token groups.
The 15 slots, and the 15 sequential Markov steps, then score only
**shared set ∪ picked groups**.

Prior work routes per token (DynaSpec arXiv 2510.13847, SpecVocab arXiv 2602.13836, both on EAGLE).
Static frequency shortlists are FR-Spec and VocabTrim. The official DSpark vocabulary is already one of them.
Routing once per block of a block drafter was not found (two searches, 2026-11-27).

The per-slot gradient experiment that used to live here was retired. No slot conflict was measured, and eal was already near its ceiling. It is in git history before this rewrite.

## Groupings (the "experts")

A grouping puts every draft id either in the always-on **shared set** or in exactly one of K groups.
The shared set holds the most frequent answer tokens, like the shared expert of DeepSeekMoE.
The router is learned in every variant. The variants differ only in how the groups are made.

| | groups from | status |
|---|---|---|
| A | capped spherical k-means of the draft `lm_head` rows (`group_lm_head.py`, `balanced_kmeans.py`) | implemented |
| B | partition of the in-block co-occurrence graph of answer tokens | planned |
| D | groups learned with the router: soft token→group assignment plus budget and balance penalties; first as D-lite, alternating router training and token reassignment, starting from B | planned |

## Measurement 1: oracle shortlist (`block_coverage.py`)

Here the router is perfect: it picks exactly the groups of the block's target tokens. That gives the smallest shortlist each grouping allows, before any router is trained.

Every answer position starts one block of 15. For each prefix length m (4/8/12/15) the script reports:

- the shortlist size: p50/p90/p99, and as a fraction of the draft vocabulary;
- groups per block;
- the share of blocks the shared set alone covers (the static-shortlist baseline);
- the share of blocks with a token outside the draft vocabulary;
- the per-position oracle (one pick per slot) as the comparison point.

The answers are the ground-truth compact JSON plus `<|im_end|>`, tokenised as in `build_kie_draft_vocab.py`. This stands in for the target's own greedy output. The shared set is ranked on train, and coverage is measured on validation.

## Files

- `vocab_common.py`: draft vocabulary (+ KIE patch), draft head rows as served (frozen head, patched rows from the fine-tuned target head), answer → draft ids, the grouping file format.
- `balanced_kmeans.py`: k-means++ seeding, capped greedy assignment (group size ≤ ⌈N/K·(1+slack)⌉), no empty groups.
- `group_lm_head.py`: grouping A → `<stem>.pt` + `<stem>.json`. The json holds sizes, k-means history and the decoded tokens of the heaviest groups.
- `block_coverage.py`: measurement 1 for one or more grouping files.
- `run_vocab_route.sh`: launcher with `group-a`, `sweep-a` (K ∈ {256, 1024, 4096} × shared ∈ {0, 2048}, then coverage) and `coverage`. CPU only; output goes to `runs/vocab_route/$TASK/`.
- `test_vocab_route.py`: CPU unit tests.

## Run

```bash
cd ~/speed_up/Xiaomi-OCR-0
PY=/home/mr_jiang/miniforge3/envs/minicpm_ocr/bin/python
$PY -m unittest specdec/insight/test_vocab_route.py
TASK=whole_card bash specdec/insight/run_vocab_route.sh sweep-a   # or TASK=kie
```

## Next

1. Grouping B and the rule baseline (digits / CJK / Latin / punctuation), measured with the same coverage script.
2. Measurement 2: vLLM profiling of one round (target verify, draft backbone, `lm_head`, Markov loop). It turns the oracle shortlist into an upper bound on end-to-end speedup.
3. The block router and its training (design under discussion, not implemented).
