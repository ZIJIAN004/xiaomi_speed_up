"""Focused CPU checks for the MiniCPM-OCR DSpark glue (tiny configs, no checkpoints).

Run in the minicpm_ocr env:  python -m unittest specdec/dspark/test_dspark_pipeline.py
Covers: d2t/t2d decoding, official-layout load/save round trip, verifier
attachment from a target, aux-feature hooks (= residual after k layers),
the position guard, one DSpark forward/backward with frozen verifier tensors,
and the shifted loss mask of the IDSpace dataset.
"""
from __future__ import annotations

import json
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

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        user = " ".join("<image>" if part["type"] == "image" else part["text"] for part in messages[0]["content"])
        return f"<|im_start|> user {user} <|im_end|> <|im_start|> assistant {messages[1]['content']} <|im_end|>"

    def __call__(self, text, images, return_tensors, padding, truncation, add_special_tokens):
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
            sample = IDSpaceMiniCPMDataset(root / "train.jsonl", root, processor, prompt="read card")[0]
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


if __name__ == "__main__":
    unittest.main()
