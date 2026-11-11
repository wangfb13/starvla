"""Focused trainer checks that run without loading the VLM or robot datasets."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'train_dd_recovery.py'
spec = importlib.util.spec_from_file_location('train_dd_recovery', PATH)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class TrainerTest(unittest.TestCase):
    def sampler(self, rank=0, world=1, completed=0, steps=100):
        return trainer.WeightedRankSampler([.7 / 3] * 3 + [.3 / 2] * 2,
            steps=steps, completed_steps=completed, batch_size=2,
            accumulation_steps=2, rank=rank, world_size=world, seed=123)

    def test_rank_streams_reconstruct_global_stream_and_resume(self):
        rank0, rank1 = list(self.sampler(0, 2)), list(self.sampler(1, 2))
        combined = [x for pair in zip(rank0, rank1) for x in pair]
        expected = list(self.sampler(0, 1, steps=200))
        self.assertEqual(combined, expected)
        self.assertEqual(list(self.sampler(1, 2, completed=37)), rank1[37 * 4:])

    def test_weighted_stream_has_intended_replay_fraction(self):
        indices = np.asarray(list(self.sampler(steps=10000)))
        self.assertAlmostEqual(float((indices < 3).mean()), .7, delta=.012)
        with self.assertRaises(ValueError):
            trainer.WeightedRankSampler([0, 0], steps=1, completed_steps=0, batch_size=1,
                accumulation_steps=1, rank=0, world_size=1, seed=0)

    def test_processor_fingerprint_detects_stat_mutation(self):
        class Step:
            def __init__(self):
                self.value = torch.tensor([1., 2.])
            def state_dict(self):
                return {'action.mean': self.value, 'action.std': torch.tensor(3.)}
        class Pipeline:
            steps = [Step()]
        pipeline = Pipeline()
        before = trainer.processor_fingerprint(pipeline)
        self.assertEqual(before, trainer.processor_fingerprint(pipeline))
        pipeline.steps[0].value[0] += .01
        self.assertNotEqual(before, trainer.processor_fingerprint(pipeline))

    def test_batch_keeps_physical_dd_and_rejects_padding(self):
        raw = {trainer.DD_KEY: torch.randn(2, 2, 3, 96, 128),
               'action_is_pad': torch.zeros(2, 32, dtype=torch.bool)}
        batch = trainer.prepare_batch(raw, lambda x: dict(x), torch.device('cpu'), first=True)
        self.assertTrue(torch.equal(batch[trainer.DD_KEY], raw[trainer.DD_KEY]))
        raw['action_is_pad'][1, -1] = True
        with self.assertRaises(RuntimeError):
            trainer.prepare_batch(raw, lambda x: dict(x), torch.device('cpu'))
        raw['action_is_pad'].zero_()
        raw['contact_label'] = torch.tensor([0, 1])
        with self.assertRaises(RuntimeError):
            trainer.prepare_batch(raw, lambda x: dict(x), torch.device('cpu'))

    @staticmethod
    def tied_model():
        model = torch.nn.Module()
        model.embedding = torch.nn.Embedding(4, 3)
        model.lm_head = torch.nn.Linear(3, 4, bias=False)
        model.lm_head.weight = model.embedding.weight
        return model

    def test_verified_tied_embedding_alias_is_loaded_not_initialized(self):
        model = self.tied_model()
        result = trainer.verify_loaded_keys(model, {'lm_head.weight'}, ['embedding.weight'], [], allow_new_dd=False)
        self.assertEqual(result['initialized_dd_keys'], [])
        alias = result['verified_tied_aliases']['embedding.weight']
        self.assertEqual(alias['serialized_peer'], 'lm_head.weight')
        self.assertTrue(alias['same_parameter_object'])
        self.assertTrue(alias['same_storage_view'])

    def test_equal_but_independent_weights_are_not_tied(self):
        model = self.tied_model()
        model.lm_head.weight = torch.nn.Parameter(model.embedding.weight.detach().clone())
        self.assertTrue(torch.equal(model.embedding.weight, model.lm_head.weight))
        with self.assertRaisesRegex(RuntimeError, 'without a verified loaded tied peer'):
            trainer.verify_loaded_keys(model, {'lm_head.weight'}, ['embedding.weight'], [], allow_new_dd=False)

    def test_tied_peer_must_be_serialized_and_successfully_loaded(self):
        model = self.tied_model()
        with self.assertRaises(RuntimeError):
            trainer.verify_loaded_keys(model, set(), ['embedding.weight'], [], allow_new_dd=False)
        with self.assertRaises(RuntimeError):
            trainer.verify_loaded_keys(model, {'lm_head.weight'}, ['embedding.weight', 'lm_head.weight'], [], allow_new_dd=False)
        with self.assertRaises(RuntimeError):
            trainer.verify_loaded_keys(model, {'lm_head.weight'}, ['embedding.weight'], ['unknown.weight'], allow_new_dd=False)
        with self.assertRaises(RuntimeError):
            trainer.verify_loaded_keys(model, {'lm_head.weight'}, ['other.weight'], [], allow_new_dd=False)
        with self.assertRaises(RuntimeError):
            trainer.verify_loaded_keys(model, {'lm_head.weight'}, [trainer.DD_PREFIX + 'projector.weight'], [], allow_new_dd=False)

    def test_optimizer_sharded_roundtrip(self):
        try:
            import safetensors  # noqa: F401
        except ImportError:
            self.skipTest('safetensors unavailable in local CPU test environment')
        torch.manual_seed(1)
        first = torch.nn.Linear(3, 2)
        optim1 = torch.optim.AdamW(first.parameters(), lr=.01)
        for _ in range(2):
            optim1.zero_grad()
            first(torch.ones(4, 3)).square().mean().backward()
            optim1.step()
        second = torch.nn.Linear(3, 2)
        second.load_state_dict(first.state_dict())
        optim2 = torch.optim.AdamW(second.parameters(), lr=9.)
        with tempfile.TemporaryDirectory() as tmp:
            trainer.optimizer_save(optim1, tmp, max_bytes=32)
            trainer.optimizer_load(optim2, tmp)
        for model, optim in [(first, optim1), (second, optim2)]:
            optim.zero_grad()
            model(torch.ones(4, 3)).square().mean().backward()
            optim.step()
        for a, b in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()
