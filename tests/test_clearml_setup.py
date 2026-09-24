import contextlib
from io import StringIO
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from train_judge import init_clearml, report_clearml_training_logs


class FakeTask:
    init_kwargs = None

    @classmethod
    def init(cls, **kwargs):
        cls.init_kwargs = kwargs
        return cls()

    def get_output_log_web_page(self):
        return "https://clearml.example/task"


class ClearMlSetupTests(unittest.TestCase):
    def test_opt_in_creates_task_without_automatic_uploads_or_streams(self):
        args = SimpleNamespace(
            clearml=True, clearml_project="judge-tests", clearml_task_name=None,
            laya_rl=True, samples=100,
        )
        with patch.dict(os.environ, {"RANK": "0"}, clear=True), contextlib.redirect_stdout(StringIO()):
            task = init_clearml(args, FakeTask)
        self.assertIsInstance(task, FakeTask)
        self.assertEqual(FakeTask.init_kwargs["project_name"], "judge-tests")
        self.assertIn("RLCD", FakeTask.init_kwargs["task_name"])
        self.assertFalse(FakeTask.init_kwargs["reuse_last_task_id"])
        self.assertFalse(FakeTask.init_kwargs["auto_connect_arg_parser"])
        self.assertFalse(FakeTask.init_kwargs["auto_connect_frameworks"])
        self.assertFalse(FakeTask.init_kwargs["auto_connect_streams"])
        self.assertFalse(FakeTask.init_kwargs["auto_resource_monitoring"])

    def test_non_primary_process_does_not_create_task(self):
        args = SimpleNamespace(clearml=True)
        with patch.dict(os.environ, {"RANK": "1"}, clear=True):
            self.assertIsNone(init_clearml(args, FakeTask))

    def test_default_run_does_not_require_clearml(self):
        args = SimpleNamespace(clearml=False)
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(init_clearml(args))

    def test_prepare_only_does_not_create_clearml_task(self):
        args = SimpleNamespace(clearml=True, prepare_only=True)
        with patch.dict(os.environ, {"RANK": "0"}, clear=True):
            self.assertIsNone(init_clearml(args, FakeTask))

    def test_only_numeric_training_logs_are_reported(self):
        logger = SimpleNamespace(report_scalar=lambda **kwargs: scalars.append(kwargs),
                                 report_text=lambda *args, **kwargs: texts.append((args, kwargs)))
        scalars = []
        texts = []
        report_clearml_training_logs(logger, {
            "loss": 0.5, "learning_rate": 1e-5, "eval_mae": 0.25,
            "eval_predicted_unique_scores": 3,
            "selection": "prepared/train.jsonl", "bad": float("nan"),
        }, 12)
        self.assertEqual([entry["series"] for entry in scalars],
                         ["loss", "learning_rate", "eval_mae", "eval_predicted_unique_scores"])
        self.assertEqual([entry["title"] for entry in scalars],
                         ["train", "train", "test", "test"])
        self.assertEqual(len(texts), 1)
        self.assertFalse(texts[0][1]["print_console"])


if __name__ == "__main__":
    unittest.main()
