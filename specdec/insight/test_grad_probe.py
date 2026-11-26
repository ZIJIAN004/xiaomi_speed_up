"""CPU checks for the per-slot gradient probe (tiny DSpark config, no checkpoints).

Run in the minicpm_ocr env:  python -m unittest specdec/insight/test_grad_probe.py
Covers: the rho/earlier/later decomposition on a hand example, what-if gammas,
cross-group products dropping the noise bias, the probe schedule, the exact
loss/gradient split against the training forward, and one end-to-end probe
(reconstruction, real-step validation, untouched .grad / weights / RNG, analysis).
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch._dynamo

torch._dynamo.config.disable = True  # CPU eager path; the real run compiles on CUDA
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dspark"))
import minicpm_dspark as md  # noqa: E402
from test_dspark_pipeline import BLOCK, V, FakeTarget, make_official_dir  # noqa: E402


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
        # tiny block: the last slot can have no gradient (cos NaN), then cos_early_late is dropped
        self.assertIn("probe/later_mean_early", scalars)
        from probe_charts import swanlab_charts
        summary["opt_step"] = 1
        charts = swanlab_charts(summary, tensors)
        self.assertEqual(set(charts), {"probe_chart/cos_heatmap", "probe_chart/slot_decomposition",
                                       "probe_chart/whatif_later"})
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
