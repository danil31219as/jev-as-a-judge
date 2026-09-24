from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from train_judge import ensure_halo_source_path, load_train_config, parse_args


ROOT = Path(__file__).resolve().parents[1]


class TrainingConfigTests(unittest.TestCase):
    def test_halo_clone_root_is_added_to_import_path(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "src" / "configs" / "classification_config.py"
            marker.parent.mkdir(parents=True)
            marker.touch()
            with patch.dict(os.environ, {"HALO_REPO_DIR": directory}):
                original_path = sys.path.copy()
                try:
                    ensure_halo_source_path()
                    self.assertEqual(sys.path[0], str(Path(directory).resolve()))
                finally:
                    sys.path[:] = original_path

    def test_repository_configs_parse_and_share_full_preparation(self):
        ce = parse_args(["--config", str(ROOT / "configs/full_h100.toml")])
        rl = parse_args(["--config", str(ROOT / "configs/full_h100_rlcd.toml")])
        hard_rl = parse_args(["--config", str(ROOT / "configs/full_h100_rlcd_hard.toml")])
        four_b = parse_args(["--config", str(ROOT / "configs/full_h100_4b.toml")])
        self.assertTrue(ce.full_dataset and rl.full_dataset and hard_rl.full_dataset)
        self.assertTrue(four_b.full_dataset)
        self.assertEqual(four_b.model, "Qwen/Qwen3.5-4B")
        self.assertNotEqual(four_b.prepared_data_dir, ce.prepared_data_dir)
        self.assertNotEqual(four_b.output_dir, four_b.prepared_data_dir)
        self.assertEqual((four_b.per_device_train_batch_size,
                          four_b.gradient_accumulation_steps), (1, 16))
        self.assertFalse(four_b.laya_rl or four_b.hard_labels)
        self.assertEqual(ce.prepared_data_dir, rl.prepared_data_dir)
        self.assertEqual(ce.prepared_data_dir, hard_rl.prepared_data_dir)
        self.assertEqual((ce.per_device_train_batch_size, ce.gradient_accumulation_steps), (2, 8))
        self.assertFalse(ce.laya_rl)
        self.assertTrue(rl.laya_rl)
        self.assertTrue(hard_rl.laya_rl and hard_rl.hard_labels)
        self.assertFalse(ce.hard_labels or rl.hard_labels)
        self.assertNotEqual(ce.output_dir, rl.output_dir)
        self.assertNotEqual(rl.output_dir, hard_rl.output_dir)
        for name in ("sample.toml", "sample_rlcd.toml"):
            self.assertFalse(parse_args(["--config", str(ROOT / "configs" / name)]).full_dataset)

    def test_config_defaults_can_be_overridden_and_bad_keys_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.toml"
            path.write_text("full_dataset = true\nper_device_train_batch_size = 2\n", encoding="utf-8")
            args = parse_args(["--config", str(path), "--per-device-train-batch-size", "1"])
            self.assertEqual(args.per_device_train_batch_size, 1)
            path.write_text("unknown_setting = 4\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_train_config(path)
            path.write_text('epochs = "three"\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_train_config(path)


if __name__ == "__main__":
    unittest.main()
