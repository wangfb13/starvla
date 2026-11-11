"""Boundary and numeric invariants for the action-only DD recovery extension."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[2] / "src/lerobot/datasets/dd_recovery.py"
spec = importlib.util.spec_from_file_location("dd_recovery_isolated", MODULE_PATH)
dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)


class DDNumericTests(unittest.TestCase):
    def test_signed_uint16_decode_preserves_physical_values(self):
        packed = np.array([[[926, 29000, 31000], [0, 30000, 30000]]], dtype=np.uint16)
        decoded = dd.decode_dd_rgb48(packed)
        self.assertEqual(decoded.shape, (3, 1, 2))
        self.assertEqual(decoded.dtype, np.float32)
        np.testing.assert_allclose(decoded[:, 0, 0], [0.926, -1.0, 1.0])
        np.testing.assert_array_equal(decoded[:, 0, 1], [0, 0, 0])

    def test_rejects_rgb8_and_wrong_channel_order(self):
        with self.assertRaises(ValueError):
            dd.decode_dd_rgb48(np.zeros((96, 128, 3), np.uint8))
        with self.assertRaises(ValueError):
            dd.decode_dd_rgb48(np.zeros((3, 96, 128), np.uint16))

    def test_complete_future_chunk_stays_inside_segment(self):
        offsets = list(range(6, 38))
        frames = dd.safe_anchor_frames(145, 658, offsets)
        self.assertEqual((frames[0], frames[-1], len(frames)), (145, 620, 476))
        targets = frames[:, None] + np.asarray(offsets)
        self.assertGreaterEqual(targets.min(), 145)
        self.assertLess(targets.max(), 658)
        self.assertEqual(targets[-1, -1], 657)
        self.assertEqual(len(dd.safe_anchor_frames(145, 180, offsets)), 0)

    def test_history_excludes_every_pre_cut_sample(self):
        offsets = list(range(-29, 1))
        self.assertEqual(dd.history_cut_mask(145, 145, offsets).sum(), 29)
        self.assertEqual(dd.history_cut_mask(150, 145, offsets).sum(), 24)
        self.assertEqual(dd.history_cut_mask(174, 145, offsets).sum(), 0)

    def test_group_split_reproducible_and_disjoint(self):
        split = dd.grouped_episode_split(list(range(34)), 20261021)
        self.assertEqual(split, dd.grouped_episode_split(list(reversed(range(34))), 20261021))
        self.assertEqual({k: sum(v == k for v in split.values()) for k in ("train", "val", "test")},
                         {"train": 24, "val": 5, "test": 5})

    def test_replay_weights_are_source_level_70_30(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest = {"schema_version": 1, "cache_ready": True, "fps": 30,
                        "action_offsets": list(range(6, 38)), "episodes": [
                            {"source": "recovery", "episode_index": 0, "start": 30, "end": 100,
                             "dataset_from_index": 0, "split": "train", "cache": "unused.npy"},
                            {"source": "normal1", "episode_index": 0, "start": 0, "end": 100,
                             "dataset_from_index": 0, "split": "train", "cache": "unused.npy"}]}
            path.write_text(json.dumps(manifest))
            base = SimpleNamespace(meta=None)
            data = dd.DDRecoveryDataset({"recovery": base, "normal1": base}, path)
            weights = data.sampling_weights(0.7)
            is_recovery = np.array([a["source"] == "recovery" for a in data.anchors])
            self.assertAlmostEqual(float(weights[is_recovery].sum()), 0.3)
            self.assertAlmostEqual(float(weights[~is_recovery].sum()), 0.7)
            self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_actual_reviewed_recovery_boundaries(self):
        import csv
        annotation = Path(__file__).resolve().parents[2] / "scripts/dd_recovery_annotations/recovery_phase_annotations.csv"
        with annotation.open() as stream:
            rows = list(csv.DictReader(stream))
        retained = sum(int(r['recovery_bc_end_exclusive']) - int(r['recovery_bc_start']) for r in rows)
        anchors = sum(len(dd.safe_anchor_frames(int(r['recovery_bc_start']), int(r['recovery_bc_end_exclusive']), list(range(6, 38)))) for r in rows)
        self.assertEqual(retained, 15743)
        self.assertEqual(anchors, 14485)
        self.assertEqual(sum(int(r['recovery_bc_start']) for r in rows), 4856)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for runtime wrapper tests")
class DDWrapperTests(unittest.TestCase):
    def test_runtime_history_isolation_and_float32_input(self):
        import torch
        class Base:
            meta = None
            delta_timestamps = {'observation.state': [i / 30 for i in range(-29, 1)]}
            def __getitem__(self, index):
                history = torch.arange(index - 29, index + 1).float()[:, None].repeat(1, 22)
                return {'episode_index': torch.tensor(0), 'frame_index': torch.tensor(index),
                        'observation.state': history, 'action': torch.zeros(32, 10),
                        'action_is_pad': torch.zeros(32, dtype=torch.bool)}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            np.save(directory / 'dd.npy', np.zeros((100, 2, 3, 96, 128), dtype=np.float16))
            manifest = {'schema_version': 1, 'cache_ready': True, 'fps': 30, 'action_offsets': list(range(6, 38)),
                        'episodes': [{'source': 'recovery', 'episode_index': 0, 'start': 30, 'end': 100,
                                      'dataset_from_index': 0, 'split': 'train', 'cache': 'dd.npy'}]}
            path = directory / 'manifest.json'
            path.write_text(json.dumps(manifest))
            data = dd.DDRecoveryDataset({'recovery': Base()}, path)
            item = data[0]
            self.assertEqual(len(data), 33)
            self.assertEqual(item[dd.DD_KEY].dtype, torch.float32)
            self.assertEqual(tuple(item[dd.DD_KEY].shape), (2, 3, 96, 128))
            self.assertEqual(int(item['observation.state_is_pad'].sum()), 29)
            self.assertTrue(bool((item['observation.state'] == 30).all()))
            self.assertNotIn('contact_label', item)
            self.assertFalse(hasattr(data, 'contact_labels'))


if __name__ == '__main__':
    unittest.main()
