"""CPU tests; run directly to avoid unrelated robot-hardware pytest plugins.

The actual QwenPI append method is extracted from its source so this test does
not download/instantiate the multi-billion-parameter VLM. No conditioning
implementation is duplicated in the harness.
"""

import ast
import __future__
from dataclasses import dataclass
import json
from types import SimpleNamespace
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "src/starVLA/model/modules/dd_contact.py"
spec = importlib.util.spec_from_file_location("dd_contact_under_test", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DDConditioner = module.DDConditioner

qwen_source = ast.parse((ROOT / "src/starVLA/model/framework/VLM4A/QwenPI.py").read_text())
encoder_node = next(n for n in qwen_source.body if isinstance(n, ast.ClassDef) and n.name == "TactileEncoder")
qwen_class = next(n for n in qwen_source.body if isinstance(n, ast.ClassDef) and n.name == "Qwen_PI")
append_node = next(n for n in qwen_class.body if isinstance(n, ast.FunctionDef) and n.name == "_append_tactile_condition")
namespace = {"torch": torch, "np": np, "nn": nn, "List": list}
exec(compile(ast.Module(body=[encoder_node, append_node], type_ignores=[]), "QwenPI.py", "exec"), namespace)


class AppendHarness(nn.Module):
    _append_tactile_condition = namespace["_append_tactile_condition"]

    def __init__(self, enable_dd=True):
        super().__init__()
        self.use_tactile = True
        self.use_dd_contact = enable_dd
        self.tactile_encoder = namespace["TactileEncoder"](history_len=30, channels=2, latent_dim=8)
        self.tactile_proj = nn.Linear(8, 16)
        if enable_dd:
            self.dd_contact = DDConditioner(hidden_dim=16, latent_dim=8)


# Test the native force processor's saved fields and mask behavior without
# importing the unrelated deployment/HF stack into this CPU-only unit test.
processor_source = ast.parse((ROOT / "src/lerobot/policies/starvla/processor_starvla.py").read_text())
processor_node = next(n for n in processor_source.body if isinstance(n, ast.ClassDef)
                      and n.name == "StarVLAPrepareTactileHistoryProcessorStep")
processor_node.decorator_list = [ast.Name(id="dataclass", ctx=ast.Load())]
ast.fix_missing_locations(processor_node)
processor_namespace = {
    "torch": torch, "dataclass": dataclass, "ProcessorStep": object,
    "OBS_STATE": "observation.state", "TransitionKey": SimpleNamespace(OBSERVATION="observation"),
}
exec(compile(ast.Module(body=[processor_node], type_ignores=[]), "processor_starvla.py", "exec",
             flags=__future__.annotations.compiler_flag), processor_namespace)
NativeForceProcessor = processor_namespace["StarVLAPrepareTactileHistoryProcessorStep"]


class TestDDConditioning(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1021)
        torch.set_num_threads(2)

    def test_zero_residual_preserves_existing_tactile_token_exactly(self):
        harness = AppendHarness()
        hidden = [torch.randn(2, 3, 16), torch.randn(2, 3, 16)]
        examples = [dict(tactile_history=np.random.default_rng(i).normal(size=(30, 2)),
                         tactile_dd=torch.randn(2, 3, 96, 128)) for i in range(2)]
        with_dd, mask_dd = harness._append_tactile_condition(hidden, None, examples)
        harness.use_dd_contact = False
        no_dd, mask_no_dd = harness._append_tactile_condition(hidden, None, examples)
        for actual, expected in zip(with_dd, no_dd, strict=True):
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(tuple(actual.shape), (2, 4, 16))
        self.assertTrue(torch.equal(mask_dd, mask_no_dd))
        self.assertEqual(tuple(mask_dd.shape), (2, 4))

    def test_enabled_missing_dd_is_error_disabled_is_compatible(self):
        harness = AppendHarness()
        examples = [dict(tactile_history=np.zeros((30, 2), dtype=np.float32))]
        hidden = [torch.zeros(1, 2, 16)]
        with self.assertRaisesRegex(ValueError, "requires decoded tactile_dd"):
            harness._append_tactile_condition(hidden, None, examples)
        harness.use_dd_contact = False
        output, _ = harness._append_tactile_condition(hidden, None, examples)
        self.assertEqual(tuple(output[0].shape), (1, 3, 16))

    def test_shape_and_raw_uint16_are_rejected(self):
        model = DDConditioner(16, 8)
        for shape in [(2, 6, 96, 128), (2, 2, 3, 128, 96), (2, 1, 3, 96, 128)]:
            with self.assertRaises(ValueError):
                model(torch.zeros(shape))
        with self.assertRaisesRegex(ValueError, "floating point"):
            model(torch.zeros(2, 2, 3, 96, 128, dtype=torch.uint16))
        for scale in ([1, 0, 1], [1, float("nan"), 1], [1, 1]):
            with self.assertRaises(ValueError):
                DDConditioner(16, 8, channel_scale=scale)

    def test_spatial_location_finger_order_and_amplitude_remain_observable(self):
        model = DDConditioner(16, 8)
        left_top = torch.zeros(1, 2, 3, 96, 128)
        left_top[:, 0, 0, 8:24, 8:24] = 1.0
        left_bottom = torch.zeros_like(left_top)
        left_bottom[:, 0, 0, 64:80, 80:96] = 1.0
        feature = model.encode(left_top)
        self.assertGreater((feature - model.encode(left_bottom)).abs().max().item(), 1e-6)
        self.assertGreater((feature - model.encode(left_top.flip(1))).abs().max().item(), 1e-6)
        self.assertGreater((feature - model.encode(left_top * 2)).abs().max().item(), 1e-6)

    def test_action_gradient_reaches_projection_then_encoder(self):
        model = DDConditioner(16, 8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dd = torch.randn(2, 2, 3, 96, 128)
        # Frozen downstream action mapping still propagates gradients to DD.
        action_head = nn.Linear(16, 3)
        action_head.requires_grad_(False)
        old_force_token = torch.randn(2, 16)
        target_action = torch.randn(2, 3)
        def action_loss():
            return (action_head(old_force_token + model(dd)) - target_action).square().mean()
        action_loss().backward()
        self.assertGreater(model.projector.weight.grad.abs().sum().item(), 0)
        self.assertEqual(model.finger_encoder[0].weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        action_loss().backward()
        self.assertGreater(model.finger_encoder[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.fusion[0].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(action_head.weight.grad)

    def test_native_force_processor_config_roundtrip_masks_cut_history_and_preserves_dd(self):
        original = NativeForceProcessor(enabled=True, history_len=30,
            force_indices=(10, 11, 12, 16, 17, 18), output_key="observation.tactile_history")
        serialized = json.loads(json.dumps(original.get_config()))
        restored = NativeForceProcessor(**serialized)
        self.assertTrue(restored.enabled)
        self.assertEqual(restored.history_len, 30)
        self.assertEqual(tuple(restored.force_indices), (10, 11, 12, 16, 17, 18))
        # The safe dataset replaces pre-cut values with a valid retained state,
        # and this exact original processor must then zero their force history.
        state = torch.ones(1, 30, 22)
        state[..., 10:13] = torch.tensor([1., 2., 3.])
        state[..., 16:19] = torch.tensor([4., 5., 6.])
        pad = torch.ones(1, 30, dtype=torch.bool)
        pad[:, -1] = False
        dd = torch.randn(1, 2, 3, 96, 128)
        transition = {"observation": {"observation.state": state,
            "observation.state_is_pad": pad, "observation.tactile_dd": dd}}
        result = restored(transition)["observation"]
        force = result["observation.tactile_history"]
        self.assertTrue(torch.equal(force[:, :-1], torch.zeros(1, 29, 2)))
        torch.testing.assert_close(force[:, -1], torch.tensor([[9., 5. ** .5 + 41. ** .5]]))
        self.assertTrue(torch.equal(result["observation.tactile_dd"], dd))
        self.assertTrue(torch.equal(original(transition)["observation"]["observation.tactile_history"], force))

    def test_saved_scales_and_conditioner_roundtrip(self):
        original = DDConditioner(16, 8, channel_scale=(2.0, 3.0, 4.0))
        nn.init.normal_(original.projector.weight)
        restored = DDConditioner(16, 8)
        restored.load_state_dict(original.state_dict(), strict=True)
        dd = torch.randn(2, 2, 3, 96, 128)
        self.assertTrue(torch.equal(original(dd), restored(dd)))
        self.assertTrue(torch.equal(original.channel_scale, restored.channel_scale))
        self.assertFalse(any("classifier" in name for name, _ in original.named_parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
