"""Focused CPU regressions; no checkpoint, data downloads, or CUDA required."""
import unittest
import ast
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from qwen_eagle_draft import DraftAttention, XiaomiEagleDraft
from toy_rotary import ToyRotary
from idspace_gt_dataset import (IDSpaceGroundTruthDataset, collate_single, serialize_target,
                                split_train_val)
from train_eagle_idspace import (TeacherTargets, estimated_accept_length, prepare_teacher,
                                 rollout_loss, save_checkpoint, shift_left,
                                 target_forward_with_context, update_topk_active,
                                 warmup_decay_factor)
from build_draft_vocab import choose_size


class TrainingTests(unittest.TestCase):
    def test_cache_matches_explicit_allowed_edges(self):
        torch.manual_seed(3)
        attn = DraftAttention(16, 4, 2, 8, 1e-6, ToyRotary())
        b, t = 2, 5
        positions = torch.arange(t).view(1, 1, t).expand(3, b, t).clone()
        positions[1] *= 2
        positions[2] *= 3
        cache = None
        for depth in range(3):
            x = torch.randn(b, t, 32, requires_grad=True)
            result, cache = attn(x, cache, positions)
            raw, gate = attn.q_proj(x).view(b, t, 4, 16).chunk(2, -1)
            q = attn.q_norm(raw).transpose(1, 2)
            cos, sin = attn.rotary_emb(x, positions + depth)
            d = cos.shape[-1]
            rot = q[..., :d]
            turned = torch.cat((-rot[..., d // 2:], rot[..., :d // 2]), -1)
            q = torch.cat((rot * cos[:, None] + turned * sin[:, None], q[..., d:]), -1)
            expected = torch.zeros_like(q)
            for row in range(t):
                # Independent graph: base prefix, then one same-column key per rollout.
                keys = [cache[0]['k'].repeat_interleave(2, 1)[:, :, :row + 1]]
                vals = [cache[0]['v'].repeat_interleave(2, 1)[:, :, :row + 1]]
                for entry in cache[1:]:
                    keys.append(entry['k'].repeat_interleave(2, 1)[:, :, row:row + 1])
                    vals.append(entry['v'].repeat_interleave(2, 1)[:, :, row:row + 1])
                keys, vals = torch.cat(keys, 2), torch.cat(vals, 2)
                weights = (q[:, :, row:row + 1] * keys).sum(-1).mul(8 ** -0.5).softmax(-1)
                expected[:, :, row] = (weights[..., None] * vals).sum(2)
            expected = expected.transpose(1, 2).reshape(b, t, 32)
            expected = attn.o_proj(expected * gate.reshape(b, t, 32).sigmoid())
            torch.testing.assert_close(result, expected)
        result.sum().backward()
        self.assertTrue(all(entry['k'].grad_fn is not None for entry in cache))

    @unittest.skipUnless(os.environ.get('QWEN35_SOURCE'), 'set QWEN35_SOURCE to native Qwen source for source oracle')
    def test_against_native_rotary_source(self):
        # Extract only the rotary class, never import/execute the downloaded module.
        source = ast.parse(Path(os.environ['QWEN35_SOURCE']).read_text(encoding='utf-8'))
        node = next(n for n in source.body if isinstance(n, ast.ClassDef)
                    and n.name == 'Qwen3_5TextRotaryEmbedding')
        namespace = {'torch': torch, 'nn': nn, 'Qwen3_5TextConfig': object,
                     'dynamic_rope_update': lambda fn: fn}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<native-rotary>', 'exec'), namespace)
        config = SimpleNamespace(max_position_embeddings=1024, head_dim=8,
                                 hidden_size=16, num_attention_heads=4,
                                 rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                                                  'partial_rotary_factor': .5, 'mrope_section': [1, 1, 0]})
        # PyTorch 2.2 lacks nn.Buffer; this CPU oracle needs only its tensor value.
        if hasattr(nn, 'Buffer'):
            native = namespace['Qwen3_5TextRotaryEmbedding'](config)
        else:
            with patch.object(nn, 'Buffer', lambda value, persistent=False: value, create=True):
                native = namespace['Qwen3_5TextRotaryEmbedding'](config)
        fixture = ToyRotary()
        positions = torch.randint(0, 40, (3, 2, 5))
        x = torch.randn(2, 5, 32)
        for a, b in zip(native(x, positions), fixture(x, positions)):
            torch.testing.assert_close(a, b)
        native_attn = DraftAttention(16, 4, 2, 8, 1e-6, native)
        self.assertEqual(native_attn(x, position_ids=positions)[0].shape, (2, 5, 16))

    def test_four_planes_and_cache_shape_guard(self):
        attn = DraftAttention(16, 4, 2, 8, 1e-6, ToyRotary())
        x = torch.randn(2, 5, 32)
        p = torch.arange(5).view(1, 1, 5).expand(4, 2, 5)
        a, cache = attn(x, position_ids=p)
        b, _ = attn(x, position_ids=p[1:])
        torch.testing.assert_close(a, b)
        with self.assertRaises(ValueError):
            attn(x[:, :1], cache, p[..., :1])

    def test_rollout_alignment(self):
        class RecordingDraft:
            draft_ids = torch.arange(3)
            def __init__(self):
                self.embeddings = []
            def step(self, feature, embedding, cache, position_ids):
                self.embeddings.append(embedding.clone())
                return feature, torch.zeros(1, 6, 3, requires_grad=True), (cache or []) + [{}]
        draft = RecordingDraft()
        embeddings = torch.arange(6.).view(1, 6, 1)
        teacher = embeddings.expand(1, 6, 3)
        mask = torch.tensor([[False, False, True, True, True, False]])
        seen = []
        def loss(t, d, m, active):
            seen.append((t.clone(), m.clone()))
            return d.sum() * 0
        targets = TeacherTargets(teacher, torch.zeros(1, 6, dtype=torch.long),
                                 torch.ones(1, 6, dtype=torch.bool), None)
        with patch('train_eagle_idspace.distill_loss', loss):
            rollout_loss(draft, embeddings, embeddings, targets, mask, None, 3, .8,
                         torch.full((1, 1, 1), -1.))
        for depth in range(3):
            self.assertEqual(draft.embeddings[depth][0, 0, 0].item(), depth + 1)
            self.assertEqual(seen[depth][0][0, 0, 0].item(), depth + 1)
            expected = mask
            for _ in range(depth + 2):
                expected = shift_left(expected)
            torch.testing.assert_close(seen[depth][1], expected)

    def test_dtype_cast_preserves_rotary_buffers(self):
        attention = DraftAttention(16, 4, 2, 8, 1e-6, ToyRotary())
        original = attention.rotary_emb.inv_freq.clone()
        attention.to(dtype=torch.bfloat16)
        self.assertEqual(attention.q_proj.weight.dtype, torch.bfloat16)
        self.assertEqual(attention.rotary_emb.inv_freq.dtype, torch.float32)
        torch.testing.assert_close(attention.rotary_emb.inv_freq, original, rtol=0, atol=0)

    def test_gate_is_cumulative(self):
        teacher = torch.tensor([[[4., 0.], [4., 0.]]])
        topk = teacher.topk(1, dim=-1).indices
        ids = torch.arange(2)
        wrong = teacher.flip(-1)
        active = torch.ones(1, 2, dtype=torch.bool)
        active = update_topk_active(topk, wrong, active, active, ids)
        self.assertFalse(active.any())
        active = update_topk_active(topk, teacher, torch.ones_like(active), active, ids)
        self.assertFalse(active.any())

    def test_gate_compares_in_target_id_space(self):
        # Draft logit 0 is target id 5; target top-1 is id 5 -> reliable.
        topk = torch.tensor([[[5]]])
        draft_logits = torch.tensor([[[3., 0.]]])
        active = torch.ones(1, 1, dtype=torch.bool)
        kept = update_topk_active(topk, draft_logits, active, active, torch.tensor([5, 9]))
        self.assertTrue(kept.all())

    def test_reduced_vocab_teacher_and_mask(self):
        logits = torch.tensor([[[0., 1., 5., 2.], [9., 1., 0., 2.]]])
        ids = torch.tensor([1, 2])
        t2d = torch.tensor([False, True, True, False])
        teacher = prepare_teacher(logits, ids, t2d, 0)
        torch.testing.assert_close(teacher.logits, logits[..., [1, 2]])
        # Position 1's full-vocab argmax (id 0) is outside the draft vocab.
        self.assertEqual(teacher.in_vocab.tolist(), [[True, False]])
        self.assertIsNone(teacher.topk_ids)
        draft = XiaomiEagleDraft(16, 4, 32, 4, 2, 8, 1e-6, low_layer=0, mid_layer=1,
                                 high_layer=2, rotary_emb=ToyRotary(), draft_ids=ids)
        self.assertEqual(draft.draft_lm_head.out_features, 2)
        self.assertEqual(draft.t2d.tolist(), t2d.tolist())
        self.assertEqual(draft.to(dtype=torch.bfloat16).draft_ids.dtype, torch.long)
        with self.assertRaises(ValueError):
            XiaomiEagleDraft(16, 4, 32, 4, 2, 8, 1e-6, rotary_emb=ToyRotary(),
                             draft_ids=torch.tensor([2, 1]))

    def test_warmup_decay_schedule(self):
        # DeepSpeed WarmupDecayLR, log warmup over 10 of 110 updates.
        self.assertEqual(warmup_decay_factor(0, 10, 110), 0.0)
        self.assertAlmostEqual(warmup_decay_factor(9, 10, 110), 1.0)
        self.assertAlmostEqual(warmup_decay_factor(10, 10, 110), 1.0)
        self.assertAlmostEqual(warmup_decay_factor(60, 10, 110), 0.5)
        self.assertEqual(warmup_decay_factor(110, 10, 110), 0.0)
        self.assertEqual(warmup_decay_factor(500, 10, 110), 0.0)

    def test_dataset_mask_includes_end_of_turn(self):
        from PIL import Image
        answer = serialize_target({"name": "A"})
        vocab = {ch: i + 10 for i, ch in enumerate(sorted(set(answer)))}
        eot, newline = 3, 4

        class Tok:
            unk_token_id = 0
            def convert_tokens_to_ids(self, token):
                return eot if token == "<|im_end|>" else 0
            def __call__(self, text, add_special_tokens=False):
                return SimpleNamespace(input_ids=[vocab[ch] for ch in text])

        class Proc:
            tokenizer = Tok()
            def apply_chat_template(self, messages, **kwargs):
                return "rendered"
            def __call__(self, **kwargs):
                ids = [1, 2] + self.tokenizer(answer).input_ids + [eot, newline]
                return {"input_ids": torch.tensor([ids]),
                        "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}

        with tempfile.TemporaryDirectory() as tmp:
            Image.new("RGB", (4, 4)).save(Path(tmp) / "a.png")
            rows = Path(tmp) / "train.jsonl"
            rows.write_text(json.dumps({"image": "a.png", "target": {"name": "A"}}) + "\n", encoding="utf-8")
            sample = IDSpaceGroundTruthDataset(rows, tmp, Proc())[0]
        mask = sample["assistant_mask"][0].tolist()
        n = len(answer)
        self.assertEqual(mask, [False, False] + [True] * n + [True, False])

    def test_round_accuracy_in_target_id_space(self):
        class ConstantDraft:
            draft_ids = torch.tensor([2, 5])
            def step(self, feature, embedding, cache, position_ids):
                # Draft index 1 everywhere -> proposes target id 5.
                return feature, torch.tensor([[[0., 1.]] * 4]), (cache or []) + [{}]
        argmax_ids = torch.tensor([[9, 5, 7, 5]])
        teacher = TeacherTargets(torch.zeros(1, 4, 2), argmax_ids,
                                 torch.ones(1, 4, dtype=torch.bool), None)
        e = torch.zeros(1, 4, 1)
        *_, stats = rollout_loss(ConstantDraft(), e, e, teacher, torch.ones(1, 4, dtype=torch.bool),
                                 None, 1, .8, torch.zeros(1, 1, 1))
        # Round 0: positions 0,1 supervised (mask shifted twice); their teacher
        # ids are argmax[1]=5 (hit) and argmax[2]=7 (miss).
        self.assertEqual((stats.hits, stats.totals), ([1], [2]))

    def test_estimated_accept_length(self):
        self.assertAlmostEqual(estimated_accept_length([0.5, 0.5]), 0.5 + 0.25)
        self.assertEqual(estimated_accept_length([1.0, 1.0, 0.0, 1.0]), 2.0)

    def test_train_val_split_is_fixed_and_disjoint(self):
        train, val = split_train_val(100, 0.05, 42)
        self.assertEqual((len(train), len(val)), (95, 5))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(sorted(train + val), list(range(100)))
        self.assertEqual(split_train_val(100, 0.05, 42), (train, val))
        self.assertNotEqual(split_train_val(100, 0.05, 43)[1], val)
        self.assertEqual(split_train_val(10, 0.0, 42), (list(range(10)), []))
        self.assertEqual(len(split_train_val(10, 0.01, 42)[1]), 1)  # non-zero fraction keeps >= 1
        with self.assertRaises(ValueError):
            split_train_val(1, 0.5, 42)

    def test_checkpoint_write_is_atomic_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            save_checkpoint(path, {"epoch": 0})
            save_checkpoint(path, {"epoch": 1})
            self.assertEqual(torch.load(path)["epoch"], 1)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["last.pt"])

    def test_vocab_size_choice(self):
        ranked = [50, 30, 10, 5, 3, 2]  # total 100
        self.assertEqual(choose_size(ranked, 0.9, 1, 100), 3)
        self.assertEqual(choose_size(ranked, 0.9, 4, 100), 4)
        self.assertEqual(choose_size(ranked, 0.9, 128, 100), 6)  # capped by distinct tokens
        self.assertEqual(choose_size(ranked, 1.0, 1, 2), 2)       # capped by max size

    def test_native_context_capture_and_hook_cleanup(self):
        class Text(nn.Module):
            def forward(self, **kwargs):
                return kwargs['inputs_embeds']
        class Target(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.language_model = Text()
            def forward(self, **kwargs):
                e = torch.randn(1, 5, 8)
                p = torch.arange(5).view(1, 1, 5).expand(3, 1, 5)
                return self.model.language_model(inputs_embeds=e, position_ids=p)
        target = Target()
        result, embeds, positions = target_forward_with_context(target, {})
        torch.testing.assert_close(result, embeds)
        self.assertEqual(positions.shape, (3, 1, 5))
        self.assertFalse(target.model.language_model._forward_pre_hooks)

    def test_collator_preserves_native_ranks(self):
        sample = {'input_ids': torch.zeros(1, 6, dtype=torch.long),
                  'position_ids': torch.zeros(4, 1, 6, dtype=torch.long),
                  'pixel_values': torch.zeros(1, 12),
                  'image_grid_thw': torch.ones(1, 3, dtype=torch.long),
                  'assistant_mask': torch.ones(1, 6, dtype=torch.bool)}
        result = collate_single([sample])
        for key in sample:
            self.assertIs(result[key], sample[key])
        with self.assertRaises(NotImplementedError):
            collate_single([sample, sample])

    def test_fusion_gradient(self):
        draft = XiaomiEagleDraft(16, 32, 32, 4, 2, 8, 1e-6,
                                low_layer=0, mid_layer=1, high_layer=2, rotary_emb=ToyRotary())
        states = tuple(torch.randn(1, 6, 16) for _ in range(3))
        e = torch.randn(1, 6, 16)
        teacher = prepare_teacher(torch.randn(1, 6, 32), draft.draft_ids, draft.t2d, 0)
        loss, *_ = rollout_loss(draft, draft.initial_feature(states), e,
                               teacher, torch.ones(1, 6, dtype=torch.bool),
                               None, 3, .8, torch.zeros_like(e[:, :1]))
        loss.backward()
        self.assertGreater(draft.fusion.proj.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
