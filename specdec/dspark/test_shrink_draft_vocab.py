"""CPU checks for shrink_draft_vocab.py (tiny official-layout draft, no real checkpoints).

Run in the minicpm_ocr env:  python -m unittest specdec/dspark/test_shrink_draft_vocab.py
Covers: vocabulary plan (answer tokens, specials, frequency-order padding), row
provenance (source rows bit-exact, patched / missing tokens from the target head with
zero Markov rows), the written directory loading through minicpm_dspark with the
new size, and the block coverage count.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
import torch._dynamo

torch._dynamo.config.disable = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import minicpm_dspark as md  # noqa: E402
from shrink_draft_vocab import coverage, plan_vocab, shrink_state, source_vocab, write_draft_dir  # noqa: E402
from test_dspark_pipeline import V, make_official_dir  # noqa: E402


class PlanTest(unittest.TestCase):
    def test_plan(self):
        source = torch.tensor([10, 3, 7, 1, 20, 5])          # official order = frequency order
        counts = Counter({7: 4, 1: 1, 42: 3})                 # 42 is not in the source vocab
        vocab, sources = plan_vocab(counts, source, special={5, 99}, min_count=1, align=3)
        # keep {7, 1, 42} + special 5 = 4 -> padded to 6 with 10, 3 (first unused in source order)
        self.assertEqual(vocab, [1, 3, 5, 7, 10, 42])
        self.assertEqual(sources["padding"], [10, 3])
        self.assertEqual(sources["not_in_source"], [42])
        self.assertEqual(sources["special"], [5])
        vocab, _ = plan_vocab(counts, source, special=set(), min_count=2, align=1)
        self.assertEqual(vocab, [7, 42])


class ShrinkTest(unittest.TestCase):
    def test_rows_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            official = make_official_dir(Path(tmp))
            state = md._load_safetensors_dir(official)
            source_ids, _ = source_vocab(state, V, None)
            # patch: draft id 0 re-pointed at a target id outside the source vocab
            outside = sorted(set(range(V - 1)) - set(source_ids.tolist()))
            new_ids = source_ids.clone()
            new_ids[0] = outside[0]
            patch = Path(tmp) / "patch.pt"
            torch.save({"format": md.VOCAB_PATCH_FORMAT, "base_d2t": state["d2t"].clone(),
                        "d2t": new_ids - torch.arange(new_ids.numel()), "replaced_draft_ids": torch.tensor([0])},
                       patch)
            ids, stale = source_vocab(state, V, patch)
            self.assertEqual(stale, {0})
            kept_source = ids[1:6].tolist()
            missing = outside[1]                                # in no source row at all
            vocab = sorted(kept_source + [int(ids[0]), missing])
            target_head = torch.randn(V, state["lm_head.weight"].shape[1])
            new_state, fresh = shrink_state(state, ids, stale, vocab, target_head)
            self.assertEqual(sorted(fresh), sorted([int(ids[0]), missing]))
            head, w2 = new_state["lm_head.weight"], new_state["markov_head.markov_w2.weight"]
            for i, t in enumerate(vocab):
                if t in fresh:
                    self.assertTrue(torch.equal(head[i], target_head[t].to(head.dtype)))
                    self.assertEqual(float(w2[i].abs().sum()), 0.0)
                else:
                    row = ids.tolist().index(t)
                    self.assertTrue(torch.equal(head[i], state["lm_head.weight"][row]))
                    self.assertTrue(torch.equal(w2[i], state["markov_head.markov_w2.weight"][row]))
            self.assertTrue(torch.equal(new_state["markov_head.markov_w1.weight"],
                                        state["markov_head.markov_w1.weight"]))

            config = json.loads((official / "config.json").read_text(encoding="utf-8"))
            out = Path(tmp) / "small"
            write_draft_dir(official, out, config, new_state, {"test": True})
            self.assertEqual(json.loads((out / "config.json").read_text())["draft_vocab_size"], len(vocab))
            draft = md.load_official_draft(out, attn_impl="eager")
            self.assertEqual(draft.target_ids.tolist(), vocab)
            self.assertEqual(draft.model.lm_head.weight.shape[0], len(vocab))
            self.assertTrue(torch.equal(draft.model.lm_head.weight.detach().to(head.dtype), head))
            with self.assertRaises(FileExistsError):
                write_draft_dir(official, out, config, new_state, {})


class CoverageTest(unittest.TestCase):
    def test_block_coverage(self):
        class Tok:
            def convert_tokens_to_ids(self, token):
                return 0

            def __call__(self, text, add_special_tokens=False):
                return SimpleNamespace(input_ids=[int(c) for c in text if c.isdigit()])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jsonl"
            path.write_text(json.dumps({"target": {"k": "1213"}}) + "\n"
                            + json.dumps({"target": {"k": "12"}}) + "\n", encoding="utf-8")
            # row 1 ids = [1, 2, 1, 3, eot 0]; vocab lacks 3. row 2 ids = [1, 2, 0]: no miss at all
            report = coverage(path, Tok(), {0, 1, 2}, block=2)
            self.assertEqual(report["tokens"], 8)
            self.assertAlmostEqual(report["token_coverage"], 7 / 8)
            # row 1 blocks [1,2] [2,1] [1,3]x [3,0]x [0] -> 3 of 5; row 2 -> 3 of 3
            self.assertAlmostEqual(report["block_coverage"], 6 / 8)


if __name__ == "__main__":
    unittest.main()
