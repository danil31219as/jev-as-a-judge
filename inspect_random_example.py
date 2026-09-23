"""Save one random POLLUX row and decode its exact Qwen judge input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import sys
from urllib.parse import urlencode
from urllib.request import urlopen

from judge_data import marker_ids, prepare_row


MODEL_ID = "Qwen/Qwen3.5-0.8B"
DATASET_ID = "ai-forever/POLLUX"
BASE_URL = "https://datasets-server.huggingface.co/rows"


def fetch_row(offset: int) -> tuple[dict, int]:
    query = urlencode({
        "dataset": DATASET_ID,
        "config": "default",
        "split": "test",
        "offset": offset,
        "length": 1,
    })
    with urlopen(f"{BASE_URL}?{query}", timeout=30) as response:
        payload = json.load(response)
    return payload["rows"][0]["row"], int(payload["num_rows_total"])


def choose_sample(tokenizer, max_length: int, max_display_chars: int):
    random_source = secrets.SystemRandom()
    _, total = fetch_row(0)
    for _ in range(100):
        offset = random_source.randrange(total)
        row, _ = fetch_row(offset)
        try:
            prepared = prepare_row(
                row, tokenizer, seed=42, max_length=max_length, max_options=5
            )
        except ValueError:
            continue
        if prepared is None:
            continue
        example, _ = prepared
        decoded = tokenizer.decode(
            example["input_ids"], skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if len(decoded) <= max_display_chars:
            return offset, row
    raise RuntimeError("Could not find a short, unambiguous random POLLUX example")


def main() -> None:
    # Windows pipes can otherwise replace Cyrillic characters on print().
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="replace saved example with a random row")
    parser.add_argument("--tokenizer-dir", default=MODEL_ID)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-display-chars", type=int, default=4000)
    parser.add_argument("--sample-file", type=Path, default=Path("examples/pollux_random_example.json"))
    parser.add_argument("--decoded-file", type=Path, default=Path("examples/pollux_decoded_input.txt"))
    parser.add_argument("--summary-file", type=Path, default=Path("examples/pollux_inspection.json"))
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    if args.fetch or not args.sample_file.exists():
        offset, row = choose_sample(tokenizer, args.max_length, args.max_display_chars)
        args.sample_file.parent.mkdir(parents=True, exist_ok=True)
        args.sample_file.write_text(
            json.dumps({"dataset": DATASET_ID, "split": "test", "row_idx": offset, "row": row},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    saved = json.loads(args.sample_file.read_text(encoding="utf-8"))
    prepared = prepare_row(
        saved["row"], tokenizer, seed=42, max_length=args.max_length, max_options=5
    )
    if prepared is None:
        raise ValueError("The saved row no longer passes the training data filter")
    example, metadata = prepared
    start_id, close_id = marker_ids(tokenizer)
    final_assistant_start = max(
        index for index, token_id in enumerate(example["input_ids"])
        if token_id == start_id
    )
    closing_positions = [
        index for index, token_id in enumerate(example["input_ids"])
        if token_id == close_id and index > final_assistant_start
    ]
    decoded = tokenizer.decode(
        example["input_ids"], skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    args.decoded_file.parent.mkdir(parents=True, exist_ok=True)
    args.decoded_file.write_text(decoded, encoding="utf-8")
    summary = {
        "dataset": DATASET_ID,
        "split": "test",
        "row_idx": saved["row_idx"],
        "model_tokenizer": MODEL_ID,
        "token_count": len(example["input_ids"]),
        "tool_close_token_id": close_id,
        "tool_close_positions": closing_positions,
        "option_values": metadata["option_values"],
        "annotation_counts": metadata["annotation_counts"],
        "valid_annotation_count": metadata["valid_annotation_count"],
        "target_probabilities_in_call_order": metadata["target_probabilities"],
        "padded_training_target": example["labels"],
    }
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"POLLUX test row: {saved['row_idx']}")
    print(f"Token count: {len(example['input_ids'])}")
    print(f"Candidate scores in call order: {metadata['option_values']}")
    print(f"</tool_call> token positions: {closing_positions}")
    print(f"Valid assessor votes by score: {metadata['annotation_counts']}")
    print(f"Target probabilities in call order: {metadata['target_probabilities']}")
    print(f"Decoded input saved to: {args.decoded_file}")
    print("\n--- Decoded Qwen input (skip_special_tokens=False) ---\n")
    print(decoded)


if __name__ == "__main__":
    main()
