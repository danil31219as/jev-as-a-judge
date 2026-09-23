import contextlib
from io import StringIO
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from train_judge import init_clearml


class FakeTask:
    init_kwargs = None

    @classmethod
    def init(cls, **kwargs):
        cls.init_kwargs = kwargs
        return cls()

    def connect(self, params, **kwargs):
        self.params = params
        self.connect_kwargs = kwargs

    def get_output_log_web_page(self):
        return "https://clearml.example/task"


class ClearMlSetupTests(unittest.TestCase):
    def test_opt_in_creates_distinct_task_and_connects_arguments(self):
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
        self.assertEqual(task.params["samples"], 100)
        self.assertTrue(task.connect_kwargs["ignore_remote_overrides"])

    def test_non_primary_process_does_not_create_task(self):
        args = SimpleNamespace(clearml=True)
        with patch.dict(os.environ, {"RANK": "1"}, clear=True):
            self.assertIsNone(init_clearml(args, FakeTask))

    def test_default_run_does_not_require_clearml(self):
        args = SimpleNamespace(clearml=False)
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(init_clearml(args))


if __name__ == "__main__":
    unittest.main()
