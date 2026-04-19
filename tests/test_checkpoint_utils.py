import os
import tempfile
import unittest

import torch

from core.checkpoint_utils import extract_model_state_dict, find_latest_checkpoint


class CheckpointUtilsTests(unittest.TestCase):
    def _touch(self, path: str, mtime: int) -> None:
        with open(path, "wb") as file:
            file.write(b"")
        os.utime(path, (mtime, mtime))

    def test_find_latest_checkpoint_prefers_epoch_and_kind(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            p1 = os.path.join(temp_dir, "epoch_1_last.pth")
            p2 = os.path.join(temp_dir, "epoch_2_best.pth")
            p3 = os.path.join(temp_dir, "epoch_2_last.pth")
            self._touch(p1, mtime=100)
            self._touch(p2, mtime=200)
            self._touch(p3, mtime=150)

            latest = find_latest_checkpoint(temp_dir)
            self.assertEqual(latest, p3)

    def test_find_latest_checkpoint_uses_mtime_for_non_pattern_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            old_file = os.path.join(temp_dir, "old.pth")
            new_file = os.path.join(temp_dir, "new.pth")
            self._touch(old_file, mtime=100)
            self._touch(new_file, mtime=200)

            latest = find_latest_checkpoint(temp_dir)
            self.assertEqual(latest, new_file)

    def test_extract_model_state_dict_from_model_state_dict_key(self) -> None:
        checkpoint = {
            "model_state_dict": {
                "module.layer.weight": torch.tensor([1.0]),
                "module.layer.bias": torch.tensor([0.0]),
            }
        }
        state = extract_model_state_dict(checkpoint)
        self.assertIn("layer.weight", state)
        self.assertIn("layer.bias", state)
        self.assertNotIn("module.layer.weight", state)

    def test_extract_model_state_dict_from_state_dict_key(self) -> None:
        checkpoint = {"state_dict": {"weight": torch.tensor([1.0])}}
        state = extract_model_state_dict(checkpoint)
        self.assertIn("weight", state)

    def test_extract_model_state_dict_from_raw_state_dict(self) -> None:
        checkpoint = {"weight": torch.tensor([1.0]), "bias": torch.tensor([0.0])}
        state = extract_model_state_dict(checkpoint)
        self.assertEqual(set(state.keys()), {"weight", "bias"})

    def test_extract_model_state_dict_rejects_invalid_format(self) -> None:
        with self.assertRaises(ValueError):
            extract_model_state_dict({"foo": "bar"})


if __name__ == "__main__":
    unittest.main()
