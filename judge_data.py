"""Prepare rubric-conditioned candidate classification examples from POLLUX.

This module deliberately has no training dependencies, so its label and prompt
logic can be checked before downloading the model.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import math
import random
import re
from typing import Any


# POLLUX rubrics use headings such as "0: ... 1: ...".  A heading can begin a
# new line or follow the previous description on the same line.
_LEVEL = re.compile(r"(?<![\w\d])([0-4])\s*:\s+")
_MODEL_MARKUP = (
    ("<|im_start|>", "[im_start]"),
    ("<|im_end|>", "[im_end]"),
    ("<tool_call>", "[tool_call]"),
    ("</tool_call>", "[/tool_call]"),
)
MAX_INPUT_TOKENS = 4096


def _plain(value: Any) -> str:
    """Keep dataset text from creating fake chat or candidate markers."""
    result = "" if value is None else str(value)
    for original, safe in _MODEL_MARKUP:
        result = result.replace(original, safe)
    return result


def parse_rubric(rubrics: Any) -> list[tuple[int, str]] | None:
    """Read a complete, contiguous 0..K-1 score scale; reject ambiguous rows."""
    if not isinstance(rubrics, str):
        return None
    matches = list(_LEVEL.finditer(rubrics))
    values = [int(match.group(1)) for match in matches]
    if not 2 <= len(values) <= 5 or values != list(range(len(values))):
        return None
    levels = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(rubrics)
        description = _plain(rubrics[match.end() : end]).strip()
        if not description:
            return None
        levels.append((values[index], description))
    return levels


def annotation_score_counts(annotations: Any, valid_values: set[int]) -> Counter[int]:
    """Count assessor votes within the rubric, ignoring abstentions such as -1."""
    counts: Counter[int] = Counter()
    if not isinstance(annotations, list):
        return counts
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        value = annotation.get("score")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
            continue
        score = int(value)
        if score in valid_values:
            counts[score] += 1
    return counts


def make_messages(
    row: dict[str, Any], ordered_levels: list[tuple[int, str]]
) -> list[dict[str, Any]]:
    """Render the exact four-turn user/assistant/score-candidates conversation."""
    criterion = _plain(row.get("criteria_name")).strip()
    definition = _plain(row.get("criteria_description")).strip()
    reference = _plain(row.get("reference_answer")).strip()
    judge_request = (
        f"Оцени предыдущий ответ по критерию «{criterion}».\n"
        f"Описание критерия: {definition}\n"
    )
    if reference:
        judge_request += f"Эталонный ответ: {reference}\n"
    judge_request += "Выбери одну оценку из вариантов ниже."
    calls = [
        {
            "type": "function",
            "function": {
                "name": "score",
                "arguments": {"value": value, "description": description},
            },
        }
        for value, description in ordered_levels
    ]
    return [
        {"role": "user", "content": _plain(row.get("instruction"))},
        {"role": "assistant", "content": _plain(row.get("answer"))},
        {"role": "user", "content": judge_request},
        {"role": "assistant", "content": "", "tool_calls": calls},
    ]


def marker_ids(tokenizer: Any) -> tuple[int, int]:
    """Qwen's tool close is an added *single* token, despite special=False."""
    token_ids = []
    for text in ("<|im_start|>", "</tool_call>"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"{text!r} must be exactly one tokenizer token: {encoded}")
        token_ids.append(encoded[0])
    return token_ids[0], token_ids[1]


def prepare_row(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    seed: int,
    max_length: int,
    max_options: int,
    shuffle_options: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Create one tokenized example and small provenance metadata, or skip it."""
    levels = parse_rubric(row.get("rubrics"))
    if (
        levels is None
        or len(levels) > max_options
        or not _plain(row.get("instruction")).strip()
        or not _plain(row.get("answer")).strip()
    ):
        return None
    counts = annotation_score_counts(
        row.get("annotations"), {value for value, _ in levels}
    )
    vote_count = sum(counts.values())
    if vote_count == 0:
        return None

    ordered_levels = levels.copy()
    if shuffle_options:
        random.Random(seed).shuffle(ordered_levels)
    messages = make_messages(row, ordered_levels)
    # No `tools=`: Qwen would prepend a system example containing an extra
    # </tool_call>.  The assistant message already supplies the tool_calls.
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokens = tokenizer(text, add_special_tokens=False, truncation=False)
    input_ids = tokens["input_ids"]
    if len(input_ids) > min(max_length, MAX_INPUT_TOKENS):
        return None  # Never truncate away one of the scoring markers.

    start_id, end_id = marker_ids(tokenizer)
    assistant_starts = [i for i, token_id in enumerate(input_ids) if token_id == start_id]
    if not assistant_starts:
        raise ValueError("Chat template did not produce an assistant message")
    closing_positions = [
        i for i, token_id in enumerate(input_ids)
        if token_id == end_id and i > assistant_starts[-1]
    ]
    if len(closing_positions) != len(levels):
        raise ValueError(
            f"Expected {len(levels)} closing markers, got {len(closing_positions)}; "
            "check Qwen's chat template and dataset text"
        )

    option_values = [value for value, _ in ordered_levels]
    target_probabilities = [counts[value] / vote_count for value in option_values]
    example = {
        "input_ids": input_ids,
        "attention_mask": tokens.get("attention_mask", [1] * len(input_ids)),
        "labels": target_probabilities + [0.0] * (max_options - len(levels)),
        "option_values": option_values + [-1] * (max_options - len(levels)),
    }
    source_text = "\n".join(
        _plain(row.get(field))
        for field in ("instruction", "answer", "criteria_name", "rubrics")
    )
    metadata = {
        "sha256": sha256(source_text.encode("utf-8")).hexdigest(),
        "annotation_counts": {value: counts[value] for value, _ in levels},
        "valid_annotation_count": vote_count,
        "option_values": option_values,
        "target_probabilities": target_probabilities,
        "token_count": len(input_ids),
    }
    return example, metadata
