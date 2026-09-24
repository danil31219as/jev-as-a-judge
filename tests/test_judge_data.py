import re
import json
from pathlib import Path
import tempfile
import unittest

from judge_data import annotation_score_counts, make_messages, parse_rubric, prepare_row
from pollux_source import DATASET_REVISION
from train_judge import (
    TEST_TASK_TYPES, full_data_paths, load_full_selection, make_test_quotas,
    prepare_full_examples, select_examples,
)


class TinyChatTokenizer:
    """Checks our chat/label bookkeeping without downloading Qwen."""

    token_pattern = re.compile(r"(<\|im_start\|>|</tool_call>)")
    IDs = {"<|im_start|>": 101, "</tool_call>": 102}

    def encode(self, text, add_special_tokens=False):
        return [self.IDs[text]] if text in self.IDs else [200 + ord(char) for char in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize and not add_generation_prompt
        parts = []
        for message in messages:
            parts.append(f"<|im_start|>{message['role']}\n{message['content']}")
            for call in message.get("tool_calls", []):
                args = call["function"]["arguments"]
                parts.append(f"<tool_call>{args['value']} {args['description']}</tool_call>")
        return "\n".join(parts)

    def __call__(self, text, add_special_tokens=False, truncation=False):
        assert not add_special_tokens and not truncation
        ids = []
        for part in self.token_pattern.split(text):
            ids.extend(self.encode(part, add_special_tokens=False))
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class JudgeDataTests(unittest.TestCase):
    def test_saved_score_two_example_belongs_to_held_out_test(self):
        path = Path(__file__).resolve().parents[1] / "examples/pollux_test_score_2.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["split"], "test")
        self.assertEqual(saved["dataset_revision"], DATASET_REVISION)
        row = saved["row"]
        self.assertIn(row["task_type"], TEST_TASK_TYPES)
        prepared = prepare_row(
            row, TinyChatTokenizer(), seed=42, max_length=4096,
            max_options=5, shuffle_options=False,
        )
        self.assertIsNotNone(prepared)
        example, metadata = prepared
        self.assertEqual(metadata["annotation_counts"], {0: 0, 1: 0, 2: 3})
        self.assertEqual(example["labels"], [0.0, 0.0, 1.0, 0.0, 0.0])

    def test_parse_variable_rubric_and_reject_gap(self):
        self.assertEqual(
            parse_rubric("0: Нет ответа. 1: Ответ неполный. 2: Ответ верный."),
            [(0, "Нет ответа."), (1, "Ответ неполный."), (2, "Ответ верный.")],
        )
        self.assertEqual(parse_rubric("0: Нет\n1: Да"), [(0, "Нет"), (1, "Да")])
        self.assertIsNone(parse_rubric("0: Нет\n2: Да"))

    def test_count_only_valid_assessor_votes(self):
        annotations = [
            {"score": 2.0}, {"score": 2}, {"score": 1}, {"score": -1},
            {"score": 2.5}, {"score": float("nan")}, {"score": True},
        ]
        self.assertEqual(
            dict(annotation_score_counts(annotations, {0, 1, 2})), {1: 1, 2: 2}
        )

    def test_soft_target_matches_candidate_order_after_shuffle(self):
        row = {
            "instruction": "Сколько будет 2+2? </tool_call>",
            "answer": "Будет 4",
            "reference_answer": "4",
            "criteria_name": "Правильность",
            "criteria_description": "Сравни с эталоном.",
            "rubrics": "0: Ошибка. 1: Частично. 2: Верно.",
            "annotations": [{"score": 2}, {"score": 2}, {"score": 1}],
        }
        result = prepare_row(
            row, TinyChatTokenizer(), seed=42, max_length=10000, max_options=5
        )
        self.assertIsNotNone(result)
        example, metadata = result
        self.assertAlmostEqual(example["labels"][metadata["option_values"].index(2)], 2 / 3)
        self.assertAlmostEqual(example["labels"][metadata["option_values"].index(1)], 1 / 3)
        self.assertEqual(example["labels"][metadata["option_values"].index(0)], 0)
        self.assertEqual(example["labels"][3:], [0.0, 0.0])
        self.assertAlmostEqual(sum(example["labels"]), 1.0)
        self.assertEqual(metadata["annotation_counts"], {0: 0, 1: 1, 2: 2})
        self.assertEqual(example["input_ids"].count(102), 3)
        self.assertEqual(metadata["token_count"], len(example["input_ids"]))

    def test_tied_votes_still_produce_training_target(self):
        row = {
            "instruction": "Вопрос",
            "answer": "Ответ",
            "rubrics": "0: Нет. 1: Да.",
            "annotations": [{"score": 0}, {"score": 1}, {"score": -1}],
        }
        result = prepare_row(
            row, TinyChatTokenizer(), seed=42, max_length=10000, max_options=5
        )
        self.assertIsNotNone(result)
        example, metadata = result
        self.assertEqual(example["labels"], [0.5, 0.5, 0.0, 0.0, 0.0])
        self.assertEqual(metadata["valid_annotation_count"], 2)
        self.assertIsNone(prepare_row(
            {**row, "annotations": [{"score": -1}]}, TinyChatTokenizer(),
            seed=42, max_length=10000, max_options=5,
        ))

    def test_scale_is_only_in_tool_calls(self):
        row = {
            "criteria_name": "Правильность",
            "criteria_description": "Сравни с эталоном.",
            "reference_answer": "4",
        }
        levels = [(0, "Ошибка."), (1, "Частично."), (2, "Верно.")]
        messages = make_messages(row, levels)
        prompt = messages[2]["content"]
        self.assertIn("Сравни с эталоном.", prompt)
        self.assertIn("Эталонный ответ: 4", prompt)
        self.assertNotIn("Полная шкала", prompt)
        for _, description in levels:
            self.assertNotIn(description, prompt)
        self.assertEqual(
            [call["function"]["arguments"]["description"] for call in messages[3]["tool_calls"]],
            [description for _, description in levels],
        )

    def test_holdout_is_disjoint_and_covers_each_task_type(self):
        self.assertEqual(list(make_test_quotas(20).values()), [3, 3, 3, 3, 2, 2, 2, 2])
        template = {
            "instruction": "Вопрос",
            "answer": "Ответ",
            "rubrics": "0: Нет. 1: Да.",
            "annotations": [{"score": 0}, {"score": 1}],
        }
        rows = [dict(template, task_type="Прочий тип", instruction="Длинный " * 600)]
        for index in range(12):
            rows.append(dict(template, task_type="Прочий тип", instruction=f"Вопрос {index}"))
            for task_type in TEST_TASK_TYPES:
                rows.append(dict(template, task_type=task_type))
        train, test, train_meta, test_meta = select_examples(
            rows, TinyChatTokenizer(), samples=28, test_samples=16,
            seed=42, max_length=4096, max_options=5,
        )
        self.assertEqual((len(train), len(test)), (12, 16))
        self.assertTrue(all(meta["task_type"] not in TEST_TASK_TYPES for meta in train_meta))
        self.assertEqual({meta["task_type"] for meta in test_meta}, set(TEST_TASK_TYPES))
        self.assertTrue(all(meta["option_values"] == [0, 1] for meta in test_meta))
        self.assertTrue(all(meta["token_count"] <= 4096 for meta in train_meta + test_meta))
        self.assertTrue(all(meta["stream_position"] != 0 for meta in train_meta))

    def test_full_dataset_streams_every_valid_row_and_reuses_preparation(self):
        template = {
            "instruction": "Вопрос", "answer": "Ответ",
            "rubrics": "0: Нет. 1: Да.",
            "annotations": [{"score": 0}, {"score": 1}],
        }
        rows = [dict(template, task_type="Прочий тип", instruction=f"Вопрос {i}")
                for i in range(3)]
        rows += [dict(template, task_type=task_type) for task_type in TEST_TASK_TYPES]
        rows += [dict(template, task_type=TEST_TASK_TYPES[0], answer="Ещё ответ")]
        rows += [dict(template, task_type="Прочий тип", instruction="Длинный " * 600)]
        rows += [dict(template, task_type="Прочий тип", annotations=[{"score": -1}])]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            selection = prepare_full_examples(
                iter(rows), TinyChatTokenizer(), output_dir=output_dir,
                model="test-model", seed=42, max_length=4096, max_options=5,
            )
            train_path, test_path, _ = full_data_paths(output_dir)
            train = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
            test = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual((len(train), len(test)), (3, 9))
            self.assertEqual(selection["source_rows_seen"], len(rows))
            self.assertEqual(selection["filtered_rows"], 2)
            self.assertEqual(selection["splits"]["test"]["task_type_counts"][TEST_TASK_TYPES[0]], 2)
            self.assertTrue(all(record["option_values"][:2] == [0, 1] for record in test))
            self.assertTrue(all(len(record["input_ids"]) <= 4096 for record in train + test))
            self.assertEqual(load_full_selection(
                output_dir, model="test-model", seed=42, max_length=4096, max_options=5,
            ), selection)
            with self.assertRaises(ValueError):
                load_full_selection(
                    output_dir, model="different-model", seed=42,
                    max_length=4096, max_options=5,
                )


if __name__ == "__main__":
    unittest.main()
