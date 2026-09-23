"""Train a score-option judge with Halo and evaluate on held-out POLLUX task types."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any, Iterable

from judge_data import MAX_INPUT_TOKENS, marker_ids, prepare_row


TEST_TASK_TYPES = (
    "ИИ как персонаж (бытовая ситуация)",
    "ИИ как персонаж (экспертная ситуация)",
    "Прикладной брейншторминг",
    "Дать рекомендации",
    "Написать художественный текст",
    "Стайл-трансфер",
    "Придумать вопрос к тексту",
    "Изменить код",
)


def make_test_quotas(test_samples: int) -> dict[str, int]:
    """Give each held-out task type at least one test row, as evenly as possible."""
    if test_samples < len(TEST_TASK_TYPES):
        raise ValueError(f"test_samples must be at least {len(TEST_TASK_TYPES)}")
    each, extra = divmod(test_samples, len(TEST_TASK_TYPES))
    return {
        task_type: each + (index < extra)
        for index, task_type in enumerate(TEST_TASK_TYPES)
    }


def select_examples(
    source: Iterable[dict[str, Any]],
    tokenizer: Any,
    *,
    samples: int,
    test_samples: int,
    seed: int,
    max_length: int,
    max_options: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Select disjoint train/test rows by task_type from one shuffled stream."""
    train_goal = samples - test_samples
    quotas = make_test_quotas(test_samples)
    train_records: list[dict] = []
    test_records: list[dict] = []
    train_manifest: list[dict] = []
    test_manifest: list[dict] = []
    test_counts: Counter[str] = Counter()

    for source_position, row in enumerate(source):
        task_type = str(row.get("task_type") or "").strip()
        if not task_type:
            continue
        held_out = task_type in quotas
        if held_out and test_counts[task_type] >= quotas[task_type]:
            continue
        if not held_out and len(train_records) >= train_goal:
            continue
        prepared = prepare_row(
            row,
            tokenizer,
            seed=seed + source_position,
            max_length=max_length,
            max_options=max_options,
            shuffle_options=not held_out,
        )
        if prepared is None:
            continue
        record, metadata = prepared
        metadata.update({
            "stream_position": source_position,
            "task_type": task_type,
            "split": "test" if held_out else "train",
        })
        if held_out:
            if metadata["option_values"] != list(range(len(metadata["option_values"]))):
                raise ValueError("Test candidates must be in numeric score order")
            test_records.append(record)
            test_manifest.append(metadata)
            test_counts[task_type] += 1
        else:
            train_records.append(record)
            train_manifest.append(metadata)

        if len(train_records) == train_goal and len(test_records) == test_samples:
            return train_records, test_records, train_manifest, test_manifest

    raise RuntimeError(
        f"POLLUX was exhausted: train {len(train_records)}/{train_goal}, "
        f"test {len(test_records)}/{test_samples}; "
        f"test type counts {dict(test_counts)}"
    )


def summarize_split(manifest: list[dict]) -> dict:
    assessor_counts = Counter()
    for entry in manifest:
        assessor_counts.update(entry["annotation_counts"])
    return {
        "count": len(manifest),
        "task_type_counts": dict(sorted(Counter(x["task_type"] for x in manifest).items())),
        "assessor_score_counts": dict(sorted(assessor_counts.items())),
        "rows_with_assessor_disagreement": sum(
            sum(count > 0 for count in entry["annotation_counts"].values()) > 1
            for entry in manifest
        ),
        "scale_size_counts": dict(sorted(Counter(len(x["option_values"]) for x in manifest).items())),
        "rows": manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output-dir", default="checkpoints/pollux-tool-call-judge")
    parser.add_argument("--samples", type=int, default=100, help="total train + test rows")
    parser.add_argument("--test-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=MAX_INPUT_TOKENS)
    parser.add_argument("--max-options", type=int, default=5)
    parser.add_argument("--shuffle-buffer", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--laya-rl", action="store_true", help="add Laya-style RLCD to soft CE")
    parser.add_argument("--clearml", action="store_true", help="log this run to ClearML")
    parser.add_argument("--clearml-project", default="jev-as-a-judge")
    parser.add_argument("--clearml-task-name", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.samples <= args.test_samples or args.test_samples < len(TEST_TASK_TYPES):
        parser.error("samples must exceed test-samples and test-samples must cover all 8 task types")
    if not 1 <= args.max_length <= MAX_INPUT_TOKENS or args.max_options < 2:
        parser.error(f"max-length must be 1..{MAX_INPUT_TOKENS}; max-options must be >= 2")
    if args.shuffle_buffer < 1 or args.epochs <= 0 or args.learning_rate <= 0:
        parser.error("shuffle-buffer, epochs and learning-rate must be positive")
    return args


def is_primary_process() -> bool:
    """Only the global rank zero process should create or upload a ClearML task."""
    for name in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "LOCAL_RANK"):
        if name in os.environ:
            try:
                return int(os.environ[name]) in (-1, 0)
            except ValueError:
                continue
    return True


def init_clearml(args: argparse.Namespace, task_class: Any = None) -> Any:
    """Initialize before importing training libraries so ClearML sees the full run."""
    if not args.clearml or not is_primary_process():
        return None
    if task_class is None:
        try:
            from clearml import Task
        except ImportError as exc:
            raise RuntimeError("Install clearml and run clearml-init before using --clearml") from exc
        task_class = Task
    task_name = args.clearml_task_name or (
        "Qwen3.5-0.8B POLLUX judge RLCD" if args.laya_rl
        else "Qwen3.5-0.8B POLLUX judge CE"
    )
    task = task_class.init(
        project_name=args.clearml_project,
        task_name=task_name,
        reuse_last_task_id=False,
        auto_connect_arg_parser=False,
        auto_connect_frameworks=False,
    )
    task.connect(dict(vars(args)), name="run", ignore_remote_overrides=True)
    print(f"ClearML task: {task.get_output_log_web_page()}")
    return task


def run(args: argparse.Namespace, clearml_task: Any = None) -> None:
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        raise ValueError("Qwen tokenizer has no padding token")
    start_id, close_id = marker_ids(tokenizer)

    source = load_dataset("ai-forever/POLLUX", split="test", streaming=True)
    source = source.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    train_records, test_records, train_manifest, test_manifest = select_examples(
        source,
        tokenizer,
        samples=args.samples,
        test_samples=args.test_samples,
        seed=args.seed,
        max_length=args.max_length,
        max_options=args.max_options,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "dataset": "ai-forever/POLLUX",
        "source_split": "test",
        "selected_examples": args.samples,
        "seed": args.seed,
        "max_length": args.max_length,
        "max_options": args.max_options,
        "test_task_types": list(TEST_TASK_TYPES),
        "test_type_quotas": make_test_quotas(args.test_samples),
        "laya_rl": args.laya_rl,
        "splits": {
            "train": summarize_split(train_manifest),
            "test": summarize_split(test_manifest),
        },
    }
    selection_path = output_dir / "selection.json"
    selection_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if clearml_task is not None:
        clearml_task.upload_artifact(
            name="selection", artifact_object=str(selection_path.resolve()), wait_on_upload=True
        )
        logger = clearml_task.get_logger()
        logger.report_single_value(name="data/train_examples", value=len(train_records))
        logger.report_single_value(name="data/test_examples", value=len(test_records))
    print(f"Prepared {len(train_records)} train and {len(test_records)} test examples")
    if args.prepare_only:
        return

    import torch
    from transformers import DataCollatorWithPadding, TrainerCallback
    from src.configs.classification_config import ClassificationConfig
    from src.distributed.parallelism_config import ParallelismConfig
    from src.trainers.reward.classification import ClassificationTrainer
    from judge_metrics import compute_judge_metrics
    from judge_model import ToolCallJudge

    class ToolCallClassificationTrainer(ClassificationTrainer):
        """Pass score values through Halo's loss path for the optional RLCD reward."""

        def _compute_loss_inner(self, model, inputs, return_outputs):
            if self._loss_fn is not None:
                raise ValueError("ToolCallJudge requires Halo's model-computed loss")
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                option_values=inputs["option_values"],
                return_dict=True,
            )
            if return_outputs:
                return outputs.loss, {"logits": outputs.logits}
            return outputs.loss

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires a CUDA GPU with bfloat16 support")
    model = ToolCallJudge.from_pretrained(
        args.model,
        num_labels=args.max_options,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.judge_tool_close_id = close_id
    model.config.judge_im_start_id = start_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.get_text_config().pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.judge_rlcd_enabled = args.laya_rl
    model.config.judge_rlcd_group_size = 4
    model.config.judge_rlcd_sigma = 0.4

    training_args = ClassificationConfig(
        output_dir=str(output_dir),
        max_length=args.max_length,
        loss_type="cross_entropy",  # Halo delegates this loss to ToolCallJudge.
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        report_to="clearml" if args.clearml else "none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = ToolCallClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_records),
        eval_dataset=Dataset.from_list(test_records),
        data_collator=DataCollatorWithPadding(tokenizer, padding="longest"),
        processing_class=tokenizer,
        compute_metrics=compute_judge_metrics,
        is_binary=False,
        label_names_list=[str(i) for i in range(args.max_options)],
        parallelism_config=ParallelismConfig(),
    )

    if args.laya_rl:
        class LayaSigmaCallback(TrainerCallback):
            def on_epoch_begin(self, _args, state, control, **_kwargs):
                progress = min(1.0, max(0.0, (state.epoch or 0.0) / max(args.epochs - 1, 1)))
                model.config.judge_rlcd_sigma = 0.4 + (0.1 - 0.4) * progress
                return control

        trainer.add_callback(LayaSigmaCallback())

    trainer.train()
    metrics = trainer.evaluate(metric_key_prefix="test")
    metrics_path = output_dir / "test_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if clearml_task is not None:
        logger = clearml_task.get_logger()
        for metric_name in ("mae", "rmse", "f1_macro"):
            key = f"test_{metric_name}"
            if key in metrics:
                logger.report_single_value(name=f"test/{metric_name}", value=float(metrics[key]))
        clearml_task.upload_artifact(
            name="test_metrics", artifact_object=str(metrics_path.resolve()), wait_on_upload=True
        )
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Test metrics: {metrics}")
    print(f"Saved judge checkpoint to {final_dir}")


def main() -> None:
    args = parse_args()
    clearml_task = init_clearml(args)
    try:
        run(args, clearml_task)
    except BaseException as exc:
        if clearml_task is not None:
            try:
                clearml_task.mark_failed(status_message=f"{type(exc).__name__}: {exc}")
            except Exception:
                pass  # Preserve the original training error.
        raise
    else:
        if clearml_task is not None:
            clearml_task.close()


if __name__ == "__main__":
    main()
