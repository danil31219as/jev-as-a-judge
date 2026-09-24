"""Predict score probabilities for one rubric example with Transformers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from judge_data import MAX_INPUT_TOKENS, make_messages, marker_ids, parse_rubric
from judge_model import ToolCallJudge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="final/ directory saved by train_judge.py")
    parser.add_argument("--example", type=Path,
                        default=Path("examples/pollux_random_example.json"),
                        help="JSON with a POLLUX row, optionally wrapped in {'row': ...}")
    args = parser.parse_args()

    saved = json.loads(args.example.read_text(encoding="utf-8"))
    row = saved.get("row", saved)
    if not isinstance(row, dict):
        raise ValueError("Example JSON must contain one object or a {'row': object} wrapper")
    levels = parse_rubric(row.get("rubrics"))
    if levels is None:
        raise ValueError("Example needs a complete contiguous rubric in the rubrics field")
    if not str(row.get("instruction") or "").strip() or not str(row.get("answer") or "").strip():
        raise ValueError("Example needs instruction and answer")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    messages = make_messages(row, levels)
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    encoded = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    if encoded["input_ids"].shape[1] > MAX_INPUT_TOKENS:
        raise ValueError(f"Example exceeds {MAX_INPUT_TOKENS} tokens")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = ToolCallJudge.from_pretrained(args.checkpoint, dtype=dtype, attn_implementation="sdpa")
    model.to(device).eval()
    start_id, close_id = marker_ids(tokenizer)
    if (model.config.judge_im_start_id, model.config.judge_tool_close_id) != (start_id, close_id):
        raise ValueError("Checkpoint marker IDs do not match its tokenizer")
    with torch.inference_mode():
        probabilities = model.predict_proba(**{
            name: tensor.to(device) for name, tensor in encoded.items()
            if name in ("input_ids", "attention_mask")
        })[0, :len(levels)].cpu().tolist()

    options = [
        {"value": value, "probability": probability, "description": description}
        for (value, description), probability in zip(levels, probabilities)
    ]
    selected = max(options, key=lambda option: (option["probability"], -option["value"]))
    print(json.dumps({
        "predicted_score": selected["value"],
        "scores": options,
        "token_count": int(encoded["input_ids"].shape[1]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
