"""CPU checks for the block-routed vocabulary study (no model checkpoints needed).

Run in the minicpm_ocr env:  python -m unittest specdec/insight/test_vocab_route.py
Covers: oracle shortlist on a hand example (prefixes, OOV, shared-only, per-position),
capped k-means (recovers separated clusters, respects the cap, fills empty groups),
draft head rows with and without the vocab patch, answer -> draft id mapping, the
shared-set choice and the grouping file checks.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vocab_common as vc  # noqa: E402
from balanced_kmeans import assign_with_capacity, balanced_spherical_kmeans, capacity, fill_empty  # noqa: E402
from block_coverage import PAD, block_shortlists, block_windows, coverage_report, window_groups  # noqa: E402
from group_lm_head import choose_shared  # noqa: E402

S = vc.SHARED
# draft ids 0,1 shared; group 0 = {2,3,4}, group 1 = {5,6}, group 2 = {7,8,9}
TOKEN_GROUP = np.array([S, S, 0, 0, 0, 1, 1, 2, 2, 2])


class CoverageTest(unittest.TestCase):
    def setUp(self):
        self.seq = [np.array([0, 2, 5, 3, vc.OOV])]
        self.groups = window_groups(block_windows(self.seq, 3), TOKEN_GROUP)
        self.sizes = np.array([3, 2, 3])

    def test_windows(self):
        windows = block_windows(self.seq, 3)
        self.assertEqual(windows.tolist(), [[0, 2, 5], [2, 5, 3], [5, 3, -1], [3, -1, PAD], [-1, PAD, PAD]])

    def test_full_block(self):
        stats = block_shortlists(self.groups, self.sizes, 2, 3)
        self.assertEqual(stats["oov"].tolist(), [False, False, True, True, True])
        self.assertEqual(stats["shortlist"][:2].tolist(), [7, 7])   # shared 2 + g0 3 + g1 2
        self.assertEqual(stats["groups"][:2].tolist(), [2, 2])      # g0 twice in block 1 counts once

    def test_prefix_one(self):
        stats = block_shortlists(self.groups, self.sizes, 2, 1)
        self.assertEqual(stats["shortlist"][:4].tolist(), [2, 5, 4, 5])
        self.assertEqual(stats["groups"][:4].tolist(), [0, 1, 1, 1])

    def test_report(self):
        report = coverage_report(self.seq, TOKEN_GROUP, 3, 3, [1, 3])
        self.assertAlmostEqual(report["token_oov_frac"], 0.2)
        self.assertAlmostEqual(report["per_position_oracle"]["mean"], (2 + 5 + 4 + 5) / 4)
        self.assertAlmostEqual(report["prefix"]["1"]["shared_only_frac"], 0.25)
        self.assertAlmostEqual(report["prefix"]["3"]["shortlist"]["mean"], 7.0)
        self.assertAlmostEqual(report["prefix"]["3"]["oov_block_frac"], 0.6)
        self.assertAlmostEqual(report["prefix"]["3"]["shortlist"]["mean_frac"], 0.7)


class KMeansTest(unittest.TestCase):
    def test_recovers_separated_clusters(self):
        gen = torch.Generator().manual_seed(0)
        centers = torch.eye(8)[:3] * 5
        rows = torch.cat([c + 0.3 * torch.randn(30, 8, generator=gen) for c in centers])
        labels, history = balanced_spherical_kmeans(rows, 3, iters=20, slack=0.1, seed=1, topk=3)
        truth = np.repeat(np.arange(3), 30)
        for g in range(3):   # each found group is pure
            self.assertEqual(len(set(truth[labels == g])), 1)
        self.assertLessEqual(np.bincount(labels, minlength=3).max(), capacity(90, 3, 0.1))
        self.assertTrue(history)

    def test_capacity_and_leftovers(self):
        values = np.ones((6, 1))
        indices = np.zeros((6, 1), dtype=np.int64)          # every row lists only group 0
        sims = np.tile(np.array([[1.0, 0.5, 0.2]]), (6, 1))
        labels = assign_with_capacity(values, indices, cap=2, k=3, full_similarity=lambda rows: sims[rows])
        self.assertEqual(np.bincount(labels, minlength=3).tolist(), [2, 2, 2])

    def test_fill_empty(self):
        x = torch.nn.functional.normalize(torch.tensor([[1., 0.], [0.9, 0.1], [0., 1.]]), dim=1)
        centroids = torch.nn.functional.normalize(torch.tensor([[1., 0.], [0., 1.], [0.7, 0.7]]), dim=1)
        labels = fill_empty(np.array([0, 0, 1]), x, centroids)
        self.assertEqual(sorted(np.bincount(labels, minlength=3).tolist()), [1, 1, 1])

    def test_capacity_errors(self):
        with self.assertRaises(ValueError):
            capacity(3, 4, 0.1)


class VocabTest(unittest.TestCase):
    def make_dirs(self, root: Path):
        from safetensors.torch import save_file

        torch.manual_seed(0)
        target_ids = torch.tensor([1, 3, 4, 7, 9, 12])
        official = root / "official"
        official.mkdir()
        save_file({"d2t": target_ids - torch.arange(6), "lm_head.weight": torch.randn(6, 4)},
                  str(official / "model.safetensors"))
        (official / "config.json").write_text(json.dumps({"vocab_size": 20}), encoding="utf-8")
        target = root / "target"
        target.mkdir()
        self.target_head = torch.randn(20, 4)
        save_file({"model.language_model.embed_tokens.weight": self.target_head},
                  str(target / "model.safetensors"))
        return official, target, target_ids

    def test_head_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            official, target, target_ids = self.make_dirs(Path(tmp))
            base = vc.read_tensor(official, "lm_head.weight")
            vocab = vc.load_draft_vocab(official, None)
            self.assertTrue(torch.equal(vocab["target_ids"], target_ids))
            self.assertTrue(torch.equal(vc.load_draft_head(official, vocab, target), base))
            verifier = vc.load_draft_head(official, vocab, target, "verifier")
            self.assertTrue(torch.equal(verifier, self.target_head[target_ids]))

            new_ids = target_ids.clone()
            new_ids[5] = 19                                   # draft id 5 re-pointed at target 19
            patch = Path(tmp) / "patch.pt"
            torch.save({"format": vc.VOCAB_PATCH_FORMAT, "base_d2t": target_ids - torch.arange(6),
                        "d2t": new_ids - torch.arange(6), "replaced_draft_ids": torch.tensor([5])}, patch)
            vocab = vc.load_draft_vocab(official, patch)
            head = vc.load_draft_head(official, vocab, target)
            self.assertTrue(torch.equal(head[5], self.target_head[19]))
            self.assertTrue(torch.equal(head[:5], base[:5]))

            inverse = vc.inverse_lookup(vocab["target_ids"], 20)
            self.assertEqual(vc.to_draft([np.array([19, 12, 3])], inverse)[0].tolist(), [5, vc.OOV, 1])

    def test_answer_sequences(self):
        class FakeTokenizer:
            def convert_tokens_to_ids(self, token):
                return 99

            def __call__(self, text, add_special_tokens=False):
                return SimpleNamespace(input_ids=[ord(c) % 50 for c in text])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jsonl"
            path.write_text(json.dumps({"target": {"k": "v"}}) + "\n\n", encoding="utf-8")
            seqs = vc.answer_sequences(path, FakeTokenizer())
            self.assertEqual(len(seqs), 1)
            self.assertEqual(seqs[0][-1], 99)
            self.assertEqual(seqs[0].size, len('{"k":"v"}') + 1)   # compact JSON + <|im_end|>


class GroupsFileTest(unittest.TestCase):
    def test_choose_shared(self):
        counts = np.array([5, 0, 9, 9, 1])
        self.assertEqual(choose_shared(counts, 2).tolist(), [2, 3])
        self.assertEqual(choose_shared(counts, 10).tolist(), [0, 2, 3, 4])   # never ids that never occur
        self.assertEqual(choose_shared(counts, 0).size, 0)

    def test_round_trip_and_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.pt"
            vc.save_groups(path, TOKEN_GROUP, 3, torch.arange(10), "test", {"a": 1})
            data = vc.load_groups(path)
            self.assertTrue(np.array_equal(data["token_group"], TOKEN_GROUP))
            with self.assertRaises(ValueError):
                vc.save_groups(path, TOKEN_GROUP, 4, torch.arange(10), "test", {})   # group 3 empty


if __name__ == "__main__":
    unittest.main()
