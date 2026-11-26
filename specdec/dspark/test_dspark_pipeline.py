"""Focused CPU checks for the MiniCPM-OCR DSpark glue (tiny configs, no checkpoints).

Run in the minicpm_ocr env:  python -m unittest specdec/dspark/test_dspark_pipeline.py
Covers: d2t/t2d decoding, official-layout load/save round trip, verifier
attachment from a target, aux-feature hooks (= residual after k layers),
the position guard, one DSpark forward/backward with frozen verifier tensors,
the shifted loss mask of the IDSpace dataset, the KIE crop, the seeded training
subset, the draft-vocabulary patch and the LoRA merge.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch._dynamo
from torch import nn

torch._dynamo.config.disable = True  # CPU eager path; the real run compiles on CUDA
sys.path.insert(0, str(Path(__file__).resolve().parent))
import minicpm_dspark as md  # noqa: E402

V, H, D_VOCAB, BLOCK, TARGET_LAYERS = 50, 32, 30, 4, 3

TINY_OFFICIAL = {
    "architectures": ["Qwen3DSparkModel"], "model_type": "qwen3", "block_size": BLOCK,
    "hidden_size": H, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 8, "rms_norm_eps": 1e-6, "vocab_size": V,
    "draft_vocab_size": D_VOCAB, "max_position_embeddings": 256, "attention_bias": False,
    "attention_dropout": 0.0, "hidden_act": "silu", "tie_word_embeddings": False,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default", "rope_dim": 8},
    "layer_types": ["full_attention", "full_attention"], "num_target_layers": TARGET_LAYERS,
    "markov_rank": 8, "enable_confidence_head": True, "confidence_head_with_markov": True,
    "dflash_config": {"mask_token_id": V - 1, "target_layer_ids": [1, 2], "projector_type": "dspark",
                      "markov_head_type": "vanilla", "markov_rank": 8, "enable_confidence_head": True,
                      "confidence_head_alpha": 1.0, "attention_mode": "gqa"},
}


def make_official_dir(root: Path) -> Path:
    """A tiny checkpoint in the official layout: no verifier tensors, delta d2t, bf16 weights."""
    from safetensors.torch import save_file
    from speculators.models.dspark import DSparkDraftModel

    torch.manual_seed(0)
    model = DSparkDraftModel(md.build_draft_config(TINY_OFFICIAL, attn_impl="eager"))
    target_ids = torch.randperm(V - 1)[:D_VOCAB].sort().values
    state = {}
    for name, value in model.state_dict().items():
        if name in md.VERIFIER_OWNED or name in md.DERIVED or name == "d2t":
            continue
        state[name] = torch.randn_like(value.float()).to(torch.bfloat16) * 0.05
    state["d2t"] = target_ids - torch.arange(D_VOCAB)
    out = root / "official"
    out.mkdir()
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(TINY_OFFICIAL), encoding="utf-8")
    return out


class FakeNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(H) * 0.1)

    def forward(self, x):  # Qwen3.5 convention: x_norm * (1 + w)
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + self.weight)).type_as(x)


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(H, H)

    def forward(self, x):
        return x + torch.tanh(self.proj(x))


class FakeText(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(V, H)
        self.layers = nn.ModuleList([FakeLayer() for _ in range(TARGET_LAYERS)])
        self.norm = FakeNorm()

    def forward(self, inputs_embeds=None, position_ids=None):
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = FakeText()
        self.positions = None  # set to test the position guard

    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
        embeds = self.language_model.embed_tokens(input_ids)
        hidden = self.language_model(inputs_embeds=embeds, position_ids=self.positions)
        return SimpleNamespace(last_hidden_state=hidden)


class FakeTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=TARGET_LAYERS))
        self.model = FakeBase()
        self.lm_head = nn.Linear(H, V, bias=False)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight  # tied, like MiniCPM-OCR

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
        hidden = self.model(input_ids).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(hidden))


class VocabMapTest(unittest.TestCase):
    def test_delta_decoding(self):
        target_ids = torch.tensor([2, 5, 6, 9])
        ids, t2d = md.vocab_maps(target_ids - torch.arange(4), 12)
        self.assertTrue(torch.equal(ids, target_ids))
        self.assertEqual(t2d.nonzero().squeeze(-1).tolist(), [2, 5, 6, 9])

    def test_rejects_collisions_and_range(self):
        with self.assertRaises(ValueError):
            md.vocab_maps(torch.tensor([3, 2]), 12)       # both -> 3
        with self.assertRaises(ValueError):
            md.vocab_maps(torch.tensor([0, 20]), 12)


class OfficialLayoutTest(unittest.TestCase):
    def test_round_trip_is_bit_identical(self):
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as tmp:
            official = make_official_dir(Path(tmp))
            draft = md.load_official_draft(official, attn_impl="eager")
            out = md.save_official(draft, Path(tmp) / "saved", {"note": "test"})
            before, after = load_file(str(official / "model.safetensors")), load_file(str(out / "model.safetensors"))
            self.assertEqual(set(before), set(after))
            for name in before:
                self.assertEqual(before[name].dtype, after[name].dtype, name)
                self.assertTrue(torch.equal(before[name], after[name]), name)
            self.assertEqual((out / "config.json").read_bytes(), (official / "config.json").read_bytes())
            self.assertFalse((Path(tmp) / "saved.tmp").exists())

    def test_missing_trainable_tensor_is_an_error(self):
        from safetensors.torch import load_file, save_file
        with tempfile.TemporaryDirectory() as tmp:
            official = make_official_dir(Path(tmp))
            state = load_file(str(official / "model.safetensors"))
            state.pop("confidence_head.proj.weight")
            save_file(state, str(official / "model.safetensors"))
            with self.assertRaises(ValueError):
                md.load_official_draft(official, attn_impl="eager")


class TargetFeatureTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.target = FakeTarget().eval()
        self.input_ids = torch.randint(0, V - 1, (1, 24))

    def test_aux_is_residual_after_k_layers(self):
        extractor = md.TargetFeatureExtractor(self.target, [1, 2])
        feats = extractor({"input_ids": self.input_ids})
        text = self.target.model.language_model
        with torch.no_grad():
            h = text.embed_tokens(self.input_ids)
            after = []
            for layer in text.layers:
                h = layer(h)
                after.append(h)
        self.assertTrue(torch.allclose(feats["hidden_states"], torch.cat(after[:2], dim=-1)))
        self.assertTrue(torch.allclose(feats["verifier_last_hidden_states"], after[-1]))  # pre-norm
        self.assertNotIn("logits", feats)  # base model path skips the LM head

    def test_non_sequential_positions_are_rejected(self):
        self.target.model.positions = torch.arange(24).flip(0).view(1, 1, -1).expand(3, 1, -1)
        with self.assertRaises(RuntimeError):
            md.TargetFeatureExtractor(self.target, [1, 2])({"input_ids": self.input_ids})
        self.target.model.positions = torch.arange(24).view(1, 1, -1).expand(3, 1, -1)
        md.TargetFeatureExtractor(self.target, [1, 2])({"input_ids": self.input_ids})

    def test_layer_ids_are_validated(self):
        with self.assertRaises(ValueError):
            md.TargetFeatureExtractor(self.target, [0, 2])


class DraftStepTest(unittest.TestCase):
    def test_forward_backward_trains_only_draft_tensors(self):
        from speculators.models.dspark import DSparkDraftModel
        torch.manual_seed(2)
        target = FakeTarget().eval()
        with tempfile.TemporaryDirectory() as tmp:
            draft = md.load_official_draft(make_official_dir(Path(tmp)), attn_impl="eager")
        md.attach_verifier(draft, target, "checkpoint")
        trainable = md.configure_trainable(draft.model, train_lm_head=False)
        self.assertNotIn("lm_head.weight", trainable)
        self.assertIn("markov_head.markov_w1.weight", trainable)
        self.assertIn("confidence_head.proj.weight", trainable)
        # Verifier head rows are the tied target head at the draft's target ids.
        expected = target.lm_head.weight.detach()[draft.target_ids]
        self.assertTrue(torch.equal(draft.model.verifier_lm_head.weight, expected))

        model = draft.model
        input_ids = torch.randint(0, V - 1, (1, 24))
        loss_mask = torch.zeros(1, 24, dtype=torch.bool)
        loss_mask[:, 8:20] = True
        feats = md.TargetFeatureExtractor(target, model.config.aux_hidden_state_layer_ids)({"input_ids": input_ids})
        kwargs, _ = DSparkDraftModel.get_trainer_kwargs(
            loss_fn='{"ce": 0.1, "tv": 0.9}', loss_implementation="eager", max_anchors=8)
        frozen_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        _, loss, metrics = model(**md.draft_inputs(feats, input_ids, loss_mask), **kwargs)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)
            else:
                self.assertIsNone(param.grad, name)
                self.assertTrue(torch.equal(param, frozen_before[name]), name)
        reduced = md.reduce_metrics({k: float(v) for k, v in metrics.items()})
        for key in ("loss", "eal", "accept_len", "accept_rate"):
            self.assertIn(key, reduced)
        self.assertGreaterEqual(reduced["eal"], 1.0)  # counts the bonus token


class FakeTokenizer:
    """Whitespace tokenizer over a growing vocabulary; special tokens are words."""

    def __init__(self):
        self.vocab = {"<unk>": 0}
        self.unk_token_id = 0

    def _id(self, word):
        return self.vocab.setdefault(word, len(self.vocab))

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[self._id(w) for w in text.split()])

    def convert_tokens_to_ids(self, token):
        return self._id(token)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()

        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        user = " ".join("<image>" if part["type"] == "image" else part["text"] for part in messages[0]["content"])
        answer = " ".join(part["text"] for part in messages[1]["content"])
        return f"<|im_start|> user {user} <|im_end|> <|im_start|> assistant {answer} <|im_end|>"

    def __call__(self, text, images, return_tensors, **kwargs):
        self.calls.append({"size": images[0].size, **kwargs})
        ids = self.tokenizer(text[0]).input_ids
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}


class DatasetMaskTest(unittest.TestCase):
    def test_loss_mask_marks_positions_predicting_answer_and_eot(self):
        from PIL import Image
        from idspace_minicpm_dataset import IDSpaceMiniCPMDataset, serialize_target

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            Image.new("RGB", (3000, 1000)).save(root / "images" / "a.png")
            target = {"name": "A B"}
            (root / "train.jsonl").write_text(json.dumps({"image": "images/a.png", "target": target}) + "\n")
            processor = FakeProcessor()
            sample = IDSpaceMiniCPMDataset(root / "train.jsonl", root, processor, prompt="read card",
                                           max_slice_nums=None)[0]
        ids = sample["inputs"]["input_ids"][0].tolist()
        answer = processor.tokenizer(serialize_target(target)).input_ids
        start = ids.index(answer[0])
        eot = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.assertEqual(ids[start + len(answer)], eot)
        assistant = sample["assistant_mask"][0]
        loss_mask = sample["loss_mask"][0]
        self.assertEqual(assistant.nonzero().squeeze(-1).tolist(), list(range(start, start + len(answer) + 1)))
        # loss_mask[i] == assistant_mask[i + 1]: position i's target distribution predicts token i+1.
        self.assertTrue(torch.equal(loss_mask[:-1], assistant[1:]))
        self.assertFalse(bool(loss_mask[-1]))
        self.assertEqual(processor.calls[-1]["size"], (2048, 683))   # legacy serving resize
        self.assertNotIn("max_slice_nums", processor.calls[-1])

    def test_kie_crop_budget_and_slices(self):
        from PIL import Image
        from idspace_minicpm_dataset import IDSpaceMiniCPMDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            Image.new("RGB", (4000, 3000)).save(root / "images" / "a.png")
            (root / "train.jsonl").write_text(json.dumps({"index": 7, "image": "images/a.png",
                                                          "target": {"name": "A"}}) + "\n")
            processor = FakeProcessor()
            regions = {7: [1000, 1000, 3000, 2000]}          # 2000 x 1000 crop -> scaled to the budget
            IDSpaceMiniCPMDataset(root / "train.jsonl", root, processor, prompt="p", regions=regions)[0]
            call = processor.calls[-1]
            self.assertEqual(call["max_slice_nums"], 1)
            ratio = (262144 / (2000 * 1000)) ** 0.5
            self.assertEqual(call["size"], (int(2000 * ratio), int(1000 * ratio)))
            with self.assertRaises(KeyError):
                IDSpaceMiniCPMDataset(root / "train.jsonl", root, processor, prompt="p", regions={8: [0, 0, 1, 1]})


class KieTaskTest(unittest.TestCase):
    def test_region_union_margin_and_clip(self):
        from kie_task import region_from_layout
        layout = {"image_width": 1000, "image_height": 800,
                  "blocks": [{"bbox": [100, 200, 300, 400]}, {"bbox": [250, 150, 990, 300]}]}
        # union [100,150,990,400]; margins int(890*.03)=26, int(250*.03)=7; x2 clipped to 1000
        self.assertEqual(region_from_layout(layout), [74, 143, 1000, 407])
        small = {"image_width": 50, "image_height": 50, "blocks": [{"bbox": [10, 10, 20, 20]}]}
        self.assertEqual(region_from_layout(small), [8, 8, 22, 22])    # margin at least 2 px
        empty = {"image_width": 640, "image_height": 480, "blocks": []}
        self.assertEqual(region_from_layout(empty), [0, 0, 640, 480])

    def test_prompt_lists_the_sixteen_fields_in_order(self):
        from kie_task import FIELDS, KIE_PROMPT
        schema = json.loads(KIE_PROMPT.split("\n")[1])
        self.assertEqual(tuple(schema), FIELDS)
        self.assertEqual(len(FIELDS), 16)


class TrainSubsetTest(unittest.TestCase):
    def test_seeded_subset_is_recorded_and_stable(self):
        from train_dspark import resolve_split
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(val_jsonl=Path("val.jsonl"), val_fraction=0.05, seed=42, train_samples=10,
                                   max_samples=0, train_jsonl=Path("train.jsonl"), task="kie",
                                   layout_jsonl=Path("layout.jsonl"), output_dir=Path(tmp))
            train_idx, val_idx = resolve_split(100, args)
            self.assertEqual(len(train_idx), 10)
            self.assertEqual(val_idx, [])
            self.assertEqual(train_idx, sorted(set(train_idx)))
            self.assertNotEqual(train_idx, list(range(10)))          # random, not the first rows
            self.assertEqual(resolve_split(100, args)[0], train_idx)  # restart: same rows
            args.seed = 7
            with self.assertRaises(ValueError):                       # same dir, different split
                resolve_split(100, args)


class SelectionRuleTest(unittest.TestCase):
    def test_needs_higher_eal_and_lower_loss(self):
        from train_dspark import is_improvement
        best = {"val/eal": 2.0, "val/loss": 0.5, "epoch": -1}
        self.assertTrue(is_improvement({"val/eal": 2.1, "val/loss": 0.4}, best, "val/eal", True))
        self.assertFalse(is_improvement({"val/eal": 2.1, "val/loss": 0.6}, best, "val/eal", True))
        self.assertFalse(is_improvement({"val/eal": 1.9, "val/loss": 0.4}, best, "val/eal", True))
        self.assertFalse(is_improvement({"val/eal": 2.0, "val/loss": 0.4}, best, "val/eal", True))
        self.assertTrue(is_improvement({"val/eal": 2.1, "val/loss": 0.6}, best, "val/eal", False))
        self.assertFalse(is_improvement({}, best, "val/eal", True))


class DraftVocabPatchTest(unittest.TestCase):
    def test_patch_repoints_tail_ids_and_survives_save(self):
        from collections import Counter
        from safetensors.torch import load_file
        from build_kie_draft_vocab import plan_patch

        torch.manual_seed(3)
        target = FakeTarget().eval()
        with tempfile.TemporaryDirectory() as tmp:
            official = make_official_dir(Path(tmp))
            draft = md.load_official_draft(official, attn_impl="eager")
            ids = draft.target_ids.clone()
            outside = sorted(set(range(V - 1)) - set(ids.tolist()))
            # answers use two tokens missing from the draft vocab and the last official token
            counts = Counter({outside[0]: 5, outside[1]: 9, int(ids[-1]): 3})
            new_ids, freed, added = plan_patch(ids, counts, set(counts))
            self.assertEqual(added, [outside[1], outside[0]])            # most frequent first
            self.assertEqual(freed, [D_VOCAB - 2, D_VOCAB - 3])          # tail, skipping the protected last id
            d2t = new_ids - torch.arange(D_VOCAB)
            patch = {"format": md.VOCAB_PATCH_FORMAT, "base_d2t": draft.model.d2t.clone(), "d2t": d2t,
                     "replaced_draft_ids": torch.tensor(freed), "added_target_ids": torch.tensor(added),
                     "removed_target_ids": ids[freed]}
            torch.save(patch, Path(tmp) / "vocab.pt")
            markov_before = draft.model.markov_head.markov_w2.weight.detach().clone()
            md.apply_draft_vocab(draft, Path(tmp) / "vocab.pt")
            self.assertTrue(torch.equal(draft.target_ids, new_ids))
            self.assertTrue(bool(draft.model.t2d[outside[0]]) and bool(draft.model.t2d[outside[1]]))
            w2 = draft.model.markov_head.markov_w2.weight
            self.assertTrue(bool((w2[freed] == 0).all()))
            keep = [i for i in range(D_VOCAB) if i not in freed]
            self.assertTrue(torch.equal(w2[keep], markov_before[keep]))

            lm_before = draft.model.lm_head.weight.detach().clone()
            md.attach_verifier(draft, target, "checkpoint")
            head = target.lm_head.weight.detach()
            self.assertTrue(torch.equal(draft.model.verifier_lm_head.weight, head[new_ids]))
            self.assertTrue(torch.equal(draft.model.lm_head.weight[freed], head[new_ids[freed]]))
            self.assertTrue(torch.equal(draft.model.lm_head.weight[keep], lm_before[keep]))

            out = md.save_official(draft, Path(tmp) / "saved")
            saved = load_file(str(out / "model.safetensors"))
            self.assertTrue(torch.equal(saved["d2t"].long(), d2t))
            self.assertEqual((out / "config.json").read_bytes(), (official / "config.json").read_bytes())
            reloaded = md.load_official_draft(out, attn_impl="eager")
            self.assertTrue(torch.equal(reloaded.target_ids, new_ids))
            with self.assertRaises(ValueError):                          # built from another vocabulary
                md.apply_draft_vocab(reloaded, Path(tmp) / "vocab.pt")


class MergeLoraTest(unittest.TestCase):
    def test_merge_adds_scaled_low_rank_delta(self):
        from safetensors.torch import load_file, save_file
        import merge_lora

        torch.manual_seed(4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, adapter = root / "base", root / "adapter"
            base.mkdir()
            adapter.mkdir()
            weights = {"model.layers.0.q.weight": torch.randn(6, 4).to(torch.bfloat16),
                       "model.norm.weight": torch.randn(4).to(torch.bfloat16)}
            save_file(weights, str(base / "model.safetensors"), metadata={"format": "pt"})
            (base / "config.json").write_text("{}")
            a, b = torch.randn(2, 4), torch.randn(6, 2)
            save_file({"base_model.model.model.layers.0.q.lora_A.weight": a,
                       "base_model.model.model.layers.0.q.lora_B.weight": b},
                      str(adapter / "adapter_model.safetensors"))
            (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 4}))
            args = SimpleNamespace(base=base, adapter=adapter, output=root / "merged", overwrite=False)
            info = merge_lora.merge(args)
            merged = load_file(str(root / "merged" / "model.safetensors"))
            expected = (weights["model.layers.0.q.weight"].double() + 2.0 * (b.double() @ a.double())
                        ).to(torch.bfloat16)
            self.assertTrue(torch.equal(merged["model.layers.0.q.weight"], expected))
            self.assertTrue(torch.equal(merged["model.norm.weight"], weights["model.norm.weight"]))
            self.assertEqual(info["merged_modules"], 1)
            self.assertTrue((root / "merged" / "config.json").is_file())
            with self.assertRaises(FileExistsError):
                merge_lora.merge(args)
            (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 4,
                                                                     "use_dora": True}))
            args.overwrite = True
            with self.assertRaises(ValueError):
                merge_lora.merge(args)


class TrackerTest(unittest.TestCase):
    def test_swanlab_run_id_kept_and_epoch_axis(self):
        import os
        from unittest import mock
        from tracker import Tracker

        calls = []
        fake = SimpleNamespace(init=lambda **kw: calls.append(("init", kw)),
                               define_metric=lambda key, **kw: calls.append(("define", key, kw)),
                               log=lambda m, step: calls.append(("log", m, step)),
                               finish=lambda: calls.append(("finish",)))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"swanlab": fake}):
            args = SimpleNamespace(tracker="swanlab", tracker_project="p", tracker_run_name=None,
                                   tracker_mode="online", output_dir=Path(tmp) / "run1")
            args.output_dir.mkdir()
            with mock.patch.dict(os.environ, {"SWANLAB_API_KEY": ""}), self.assertRaises(SystemExit):
                Tracker(args, {})                                    # no key: fail, do not prompt
            with mock.patch.dict(os.environ, {"SWANLAB_API_KEY": "k"}):
                t = Tracker(args, {"lr": 1.0})
                t.log({"val/eal": 2.0}, step=5, epoch=0)
                t.finish()
                first = calls[0][1]
                self.assertEqual((first["name"], first["resume"], first["config"]), ("run1", "allow", {"lr": 1.0}))
                self.assertIn(("define", "val/*", {"x_axis": "epoch"}), calls)
                self.assertIn(("log", {"epoch": 0, "val/eal": 2.0}, 5), calls)
                calls.clear()
                Tracker(args, {})                                    # --resume: same run id
                self.assertEqual(calls[0][1]["id"], first["id"])
        self.assertIsNone(Tracker(SimpleNamespace(tracker="none"), {}).lib)


class RandomInitTest(unittest.TestCase):
    def test_keeps_layout_and_vocab_not_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            official = make_official_dir(Path(tmp))
            loaded = md.load_official_draft(official, attn_impl="eager")
            torch.manual_seed(0)
            before = torch.rand(3)
            torch.manual_seed(0)
            a = md.load_official_draft(official, attn_impl="eager", random_init_seed=3)
            self.assertTrue(torch.equal(torch.rand(3), before))          # global RNG untouched
            b = md.load_official_draft(official, attn_impl="eager", random_init_seed=3)
            c = md.load_official_draft(official, attn_impl="eager", random_init_seed=4)
            self.assertTrue(torch.equal(a.model.d2t, loaded.model.d2t))
            self.assertTrue(torch.equal(a.target_ids, loaded.target_ids))
            self.assertEqual(a.key_dtypes, loaded.key_dtypes)
            self.assertEqual(a.random_init_seed, 3)
            sa, sb, sc, so = (d.model.state_dict() for d in (a, b, c, loaded))
            # lm_head is not drawn by speculators' init; attach_verifier sets it (checked below)
            weights = [k for k in loaded.key_dtypes if k not in ("d2t", "lm_head.weight")]
            for k in weights:
                self.assertTrue(torch.equal(sa[k], sb[k]), k)               # seeded
                self.assertFalse(torch.equal(sa[k].float(), so[k].float()), k)  # not the checkpoint
            self.assertTrue(any(not torch.equal(sa[k], sc[k]) for k in weights))
            target = FakeTarget().eval()
            md.attach_verifier(a, target, "verifier")
            self.assertTrue(torch.equal(a.model.lm_head.weight, target.lm_head.weight.detach()[a.target_ids]))
            out = md.save_official(a, Path(tmp) / "saved")
            from safetensors.torch import load_file
            self.assertEqual(set(load_file(str(out / "model.safetensors"))),
                             set(load_file(str(official / "model.safetensors"))))

    def test_args(self):
        from train_dspark import parse_args
        base = ["--model", "m", "--init-draft", "d", "--train-jsonl", "t", "--image-root", "i",
                "--output-dir", "o", "--layout-jsonl", "l"]
        self.assertEqual(parse_args(base).lm_head_init, "checkpoint")
        args = parse_args(base + ["--random-init", "--seed", "7"])
        self.assertEqual((args.lm_head_init, args.init_seed), ("verifier", 7))
        with self.assertRaises(SystemExit):
            parse_args(base + ["--random-init", "--lm-head-init", "checkpoint"])
        self.assertEqual(parse_args(base + ["--probe-steps", "5,0,5"]).probe_steps, [0, 5])
        with self.assertRaises(SystemExit):
            parse_args(base + ["--probe-steps", "0", "--probe-samples", "6", "--probe-groups", "4"])


class GradProbeMathTest(unittest.TestCase):
    def test_decomposition_matches_hand_example(self):
        from grad_probe import decompose
        g = torch.tensor([[1.0, 0.0], [-0.3, 0.3], [-0.1, -0.1]], dtype=torch.float64)  # already weighted
        dec = decompose(g @ g.T, torch.ones(3, dtype=torch.float64), 3)
        self.assertAlmostEqual(dec["rho"][0], 0.6)
        self.assertAlmostEqual(dec["later"][0], -0.4)
        self.assertEqual(dec["earlier"][0], 0.0)
        # slot 1: own 0.18; earlier <g1,g0>/0.18 = -0.3/0.18; later <g1,g2>/0.18 = 0
        self.assertAlmostEqual(dec["earlier"][1], -0.3 / 0.18)
        self.assertAlmostEqual(dec["later"][1], 0.0)

    def test_what_if_rescales_the_undecayed_gradients(self):
        from grad_probe import what_if
        raw = torch.tensor([[1.0, 0.0], [-1.0, 1.5], [-1.2, -1.2]], dtype=torch.float64)
        result = what_if(raw @ raw.T, [0, 5, 10], [4.0, float("inf")], alpha=None)
        expected = -(math.exp(-5 / 4) * 1.0 + math.exp(-10 / 4) * 1.2)
        self.assertAlmostEqual(result["4.0"]["later"][0], expected)
        self.assertAlmostEqual(result["inf"]["later"][0], -2.2)
        self.assertEqual(result["inf"]["weight"], [1.0, 1.0, 1.0])

    def test_cross_group_products_drop_the_noise_bias(self):
        from grad_probe import batch_view, signal_view, split_signals
        torch.manual_seed(0)
        mu = torch.tensor([[1.0, 0.0, 0.0], [-0.5, 0.5, 0.0]], dtype=torch.float64)  # 2 terms
        groups, size, trials = 4, 16, 400
        sig, bat = torch.zeros(2, 2, dtype=torch.float64), torch.zeros(2, 2, dtype=torch.float64)
        for _ in range(trials):
            means = mu + torch.randn(groups, size, 2, 3, dtype=torch.float64).mean(1)  # noise sd 1 per sample
            pair = torch.einsum("iax,jbx->ijab", means, means)
            sig += signal_view(pair) / trials
            bat += batch_view(pair) / trials
            self.assertEqual(len(split_signals(pair)), 3)
        true = mu @ mu.T
        self.assertTrue(torch.allclose(sig, true, atol=0.03))
        self.assertAlmostEqual(float(bat[0, 0] - true[0, 0]), 3 / size, delta=0.03)  # noise / batch

    def test_schedule(self):
        from grad_probe import parse_steps, probe_due
        steps = parse_steps("0,10, 5")
        self.assertEqual(steps, [0, 5, 10])
        due = [s for s in range(40) if probe_due(s, steps, 10)]
        self.assertEqual(due, [0, 5, 10, 20, 30])
        self.assertEqual([s for s in range(5) if probe_due(s, [], 0)], [])


class GradProbeModelTest(unittest.TestCase):
    def setUp(self):
        from speculators.models.dspark import DSparkDraftModel
        torch.manual_seed(3)
        self.target = FakeTarget().eval()
        with tempfile.TemporaryDirectory() as tmp:
            draft = md.load_official_draft(make_official_dir(Path(tmp)), attn_impl="eager")
        md.attach_verifier(draft, self.target, "checkpoint")
        md.configure_trainable(draft.model, train_lm_head=False)
        self.model = draft.model.float()
        self.kwargs, _ = DSparkDraftModel.get_trainer_kwargs(
            loss_fn='{"ce": 0.1, "tv": 0.9}', loss_implementation="eager", max_anchors=8)
        self.batches = []
        for _ in range(8):
            input_ids = torch.randint(0, V - 1, (1, 24))
            loss_mask = torch.zeros(1, 24, dtype=torch.bool)
            loss_mask[:, 6:20] = True
            feats = md.TargetFeatureExtractor(self.target, self.model.config.aux_hidden_state_layer_ids)(
                {"input_ids": input_ids})
            self.batches.append(md.draft_inputs(feats, input_ids, loss_mask))

    def test_split_reproduces_the_training_loss_and_gradient(self):
        from grad_probe import param_layout, slot_losses
        batch = self.batches[0]
        torch.manual_seed(11)
        _, loss, _ = self.model(**batch, **self.kwargs)
        torch.manual_seed(11)
        split = slot_losses(self.model, batch, self.kwargs)
        self.assertEqual(split.labels, [f"slot{k}" for k in range(BLOCK)] + ["conf"])
        self.assertAlmostEqual(float(split.total), float(loss), places=5)
        layout = param_layout(self.model)
        true = torch.autograd.grad(loss, layout.params, allow_unused=True)
        parts = [torch.autograd.grad(t, layout.params, retain_graph=True, allow_unused=True) for t in split.terms]
        for i, param in enumerate(layout.params):
            recon = sum(float(c) * (p[i] if p[i] is not None else torch.zeros_like(param))
                        for c, p in zip(split.coef, parts, strict=True))
            ref = true[i] if true[i] is not None else torch.zeros_like(param)
            self.assertTrue(torch.allclose(recon, ref, atol=1e-6, rtol=1e-4), layout.names[i])

    def test_probe_end_to_end_leaves_training_state_alone(self):
        from grad_probe import probe_scalars, run_probe
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-3)
        torch.manual_seed(1)
        _, loss, _ = self.model(**self.batches[0], **self.kwargs)
        loss.backward()
        optimizer.step()                                   # populate exp_avg_sq; .grad stays set
        grads_before = [p.grad.clone() for p in params]
        weights_before = [p.detach().clone() for p in params]
        torch.manual_seed(123)
        rng_before = torch.get_rng_state()
        summary, tensors = run_probe(self.model, self.kwargs, lambda i: self.batches[i], 8, groups=4, seed=5,
                                     optimizer=optimizer, step_check=True, step_frac=1e-3, log=lambda _: None)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        for p, g, w in zip(params, grads_before, weights_before, strict=True):
            self.assertTrue(torch.equal(p.grad, g))
            self.assertTrue(torch.equal(p.detach(), w))
        recon = summary["checks"]["reconstruction"]
        self.assertGreater(recon["cos"], 0.99999)
        self.assertAlmostEqual(recon["loss_true"], recon["loss_split"], places=5)
        for kind in ("raw", "adam"):
            check = summary["checks"]["step"][kind]
            self.assertGreater(check["corr"], 0.9, kind)
            self.assertAlmostEqual(check["ratio"], 1.0, delta=0.2, msg=kind)
            view = summary["views"][kind]["backbone"]["signal"]
            self.assertEqual(len(view["rho"]), BLOCK)
            cos = torch.tensor(view["cos"])
            self.assertTrue(torch.allclose(cos, cos.T, equal_nan=True))
        fine = tensors["pair_grams"]["raw"]
        self.assertIn("markov_head", fine)
        self.assertEqual(tuple(next(iter(fine.values())).shape), (4, 4, BLOCK + 1, BLOCK + 1))
        scalars = probe_scalars(summary)
        self.assertIn("probe/recon_cos", scalars)
        self.assertTrue(all(math.isfinite(v) for v in scalars.values()))

    def test_analysis_script_runs(self):
        import analyze_grad_probe
        from grad_probe import run_probe
        with tempfile.TemporaryDirectory() as tmp:
            probe_dir = Path(tmp) / "probe"
            probe_dir.mkdir()
            for step in (0, 3):
                summary, tensors = run_probe(self.model, self.kwargs, lambda i: self.batches[i], 8, groups=4,
                                             seed=5, step_check=False, log=lambda _: None)
                summary["opt_step"] = step
                torch.save(tensors, probe_dir / f"step_{step:06d}.pt")
                (probe_dir / f"step_{step:06d}.json").write_text(json.dumps(summary), encoding="utf-8")
            analyze_grad_probe.main([str(probe_dir)])
            out = probe_dir / "analysis"
            for name in ("decomposition.csv", "whatif.csv", "noise.csv", "checks.csv", "report.txt"):
                self.assertTrue((out / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
