"""Train a score-option judge with Halo and evaluate on held-out POLLUX task types."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from numbers import Real
import os
from pathlib import Path
import tomllib
from typing import Any, Iterable

from judge_data import MAX_INPUT_TOKENS, marker_ids, prepare_row
from pollux_source import DATASET_REVISION, load_pollux_source


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


def full_data_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    data_dir = output_dir / "prepared"
    return data_dir / "train.jsonl", data_dir / "test.jsonl", output_dir / "selection.json"


def _full_split_summary(count: int, task_types: Counter, scores: Counter,
                        disagreement: int, scale_sizes: Counter) -> dict:
    return {
        "count": count,
        "task_type_counts": dict(sorted(task_types.items())),
        "assessor_score_counts": dict(sorted(scores.items())),
        "rows_with_assessor_disagreement": disagreement,
        "scale_size_counts": dict(sorted(scale_sizes.items())),
    }


def prepare_full_examples(
    source: Iterable[dict[str, Any]], tokenizer: Any, *, output_dir: Path,
    model: str, seed: int, max_length: int, max_options: int,
) -> dict[str, Any]:
    """Stream every eligible POLLUX row into disk-backed train/test JSONL files."""
    train_path, test_path, selection_path = full_data_paths(output_dir)
    if any(path.exists() for path in (train_path, test_path, selection_path)):
        raise FileExistsError(
            f"Full-data output already exists in {output_dir}; use a fresh --output-dir"
        )
    train_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {split: Counter() for split in ("train", "test")}
    task_counts = {split: Counter() for split in ("train", "test")}
    score_counts = {split: Counter() for split in ("train", "test")}
    scale_counts = {split: Counter() for split in ("train", "test")}
    disagreement = Counter()
    source_rows = 0
    try:
        with train_path.open("x", encoding="utf-8") as train_file, test_path.open(
            "x", encoding="utf-8"
        ) as test_file:
            for source_position, row in enumerate(source):
                source_rows += 1
                if source_rows % 5000 == 0:
                    print(
                        f"Scanned {source_rows} rows: train={counts['train']['rows']}, "
                        f"test={counts['test']['rows']}", flush=True,
                    )
                task_type = str(row.get("task_type") or "").strip()
                if not task_type:
                    continue
                split = "test" if task_type in TEST_TASK_TYPES else "train"
                prepared = prepare_row(
                    row, tokenizer, seed=seed + source_position,
                    max_length=max_length, max_options=max_options,
                    shuffle_options=split == "train",
                )
                if prepared is None:
                    continue
                record, metadata = prepared
                if split == "test" and metadata["option_values"] != list(
                    range(len(metadata["option_values"]))
                ):
                    raise ValueError("Test candidates must be in numeric score order")
                target_file = test_file if split == "test" else train_file
                target_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split]["rows"] += 1
                task_counts[split][task_type] += 1
                score_counts[split].update(metadata["annotation_counts"])
                scale_counts[split][len(metadata["option_values"])] += 1
                disagreement[split] += sum(
                    count > 0 for count in metadata["annotation_counts"].values()
                ) > 1
        print(
            f"Finished scanning {source_rows} rows: train={counts['train']['rows']}, "
            f"test={counts['test']['rows']}", flush=True,
        )
        missing = set(TEST_TASK_TYPES) - set(task_counts["test"])
        if not counts["train"]["rows"] or missing:
            raise RuntimeError(
                f"Full POLLUX split is incomplete: train={counts['train']['rows']}, "
                f"missing test types={sorted(missing)}"
            )
        provenance = {
            "dataset": "ai-forever/POLLUX",
            "dataset_revision": DATASET_REVISION,
            "source_split": "test",
            "mode": "full",
            "model": model,
            "seed": seed,
            "max_length": max_length,
            "max_options": max_options,
            "source_rows_seen": source_rows,
            "filtered_rows": source_rows - counts["train"]["rows"] - counts["test"]["rows"],
            "test_task_types": list(TEST_TASK_TYPES),
            "splits": {
                split: _full_split_summary(
                    counts[split]["rows"], task_counts[split], score_counts[split],
                    disagreement[split], scale_counts[split],
                ) for split in ("train", "test")
            },
        }
        selection_path.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return json.loads(selection_path.read_text(encoding="utf-8"))
    except BaseException:
        train_path.unlink(missing_ok=True)
        test_path.unlink(missing_ok=True)
        selection_path.unlink(missing_ok=True)
        raise


def load_full_selection(output_dir: Path, *, model: str, seed: int,
                        max_length: int, max_options: int) -> dict[str, Any]:
    train_path, test_path, selection_path = full_data_paths(output_dir)
    if not all(path.is_file() for path in (train_path, test_path, selection_path)):
        raise FileNotFoundError(
            f"Full data is not prepared in {output_dir}; run --full-dataset --prepare-only first"
        )
    provenance = json.loads(selection_path.read_text(encoding="utf-8"))
    expected = {
        "dataset": "ai-forever/POLLUX", "source_split": "test", "mode": "full",
        "model": model, "seed": seed, "max_length": max_length,
        "max_options": max_options, "test_task_types": list(TEST_TASK_TYPES),
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError(
            f"Prepared data in {output_dir} uses different settings; "
            "use matching arguments or a fresh --output-dir"
        )
    return provenance


CONFIG_TYPES: dict[str, type] = {
    "model": str,
    "output_dir": str,
    "prepared_data_dir": str,
    "source_dir": str,
    "full_dataset": bool,
    "samples": int,
    "test_samples": int,
    "seed": int,
    "max_length": int,
    "max_options": int,
    "shuffle_buffer": int,
    "epochs": float,
    "learning_rate": float,
    "per_device_train_batch_size": int,
    "per_device_eval_batch_size": int,
    "gradient_accumulation_steps": int,
    "dataloader_num_workers": int,
    "gradient_checkpointing": bool,
    "logging_steps": int,
    "save_total_limit": int,
    "laya_rl": bool,
    "rl_group_size": int,
    "rl_sigma_start": float,
    "rl_sigma_end": float,
    "clearml": bool,
    "clearml_project": str,
    "clearml_task_name": str,
}


def load_train_config(path: str | Path) -> dict[str, Any]:
    """Read typed training defaults from TOML before parsing CLI overrides."""
    with Path(path).open("rb") as config_file:
        values = tomllib.load(config_file)
    unknown = set(values) - set(CONFIG_TYPES)
    if unknown:
        raise ValueError(f"Unknown training config keys: {', '.join(sorted(unknown))}")
    for key, value in values.items():
        expected = CONFIG_TYPES[key]
        if expected is float:
            valid = type(value) in (int, float)
        else:
            valid = type(value) is expected
        if not valid:
            raise ValueError(f"Training config {key!r} must be {expected.__name__}")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str)
    config_args, _ = config_parser.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, help="TOML file with training and data options")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output-dir", default="checkpoints/pollux-tool-call-judge")
    parser.add_argument(
        "--prepared-data-dir", default=None,
        help="reuse full-dataset preparation from another run directory",
    )
    parser.add_argument(
        "--source-dir", default="checkpoints/pollux-source",
        help="local cache of pinned POLLUX Parquet shards",
    )
    parser.add_argument(
        "--full-dataset", action="store_true",
        help="use every valid POLLUX row; hold out all rows of the 8 test task types",
    )
    parser.add_argument("--samples", type=int, default=100, help="total train + test rows")
    parser.add_argument("--test-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=MAX_INPUT_TOKENS)
    parser.add_argument("--max-options", type=int, default=5)
    parser.add_argument("--shuffle-buffer", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--laya-rl", action="store_true", help="add Laya-style RLCD to soft CE")
    parser.add_argument("--rl-group-size", type=int, default=4)
    parser.add_argument("--rl-sigma-start", type=float, default=0.4)
    parser.add_argument("--rl-sigma-end", type=float, default=0.1)
    parser.add_argument("--clearml", action="store_true", help="log this run to ClearML")
    parser.add_argument("--clearml-project", default="jev-as-a-judge")
    parser.add_argument("--clearml-task-name", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    if config_args.config:
        try:
            parser.set_defaults(**load_train_config(config_args.config))
        except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
            parser.error(f"cannot load --config: {exc}")
    args = parser.parse_args(argv)
    if args.prepared_data_dir and not args.full_dataset:
        parser.error("--prepared-data-dir requires --full-dataset")
    if not args.full_dataset and (
        args.samples <= args.test_samples or args.test_samples < len(TEST_TASK_TYPES)
    ):
        parser.error("samples must exceed test-samples and test-samples must cover all 8 task types")
    if not 1 <= args.max_length <= MAX_INPUT_TOKENS or args.max_options < 2:
        parser.error(f"max-length must be 1..{MAX_INPUT_TOKENS}; max-options must be >= 2")
    if args.shuffle_buffer < 1 or args.epochs <= 0 or args.learning_rate <= 0:
        parser.error("shuffle-buffer, epochs and learning-rate must be positive")
    if (args.per_device_train_batch_size < 1 or args.per_device_eval_batch_size < 1
            or args.gradient_accumulation_steps < 1 or args.dataloader_num_workers < 0):
        parser.error("batch sizes and gradient-accumulation-steps must be positive; workers >= 0")
    if args.logging_steps < 1 or args.save_total_limit < 1:
        parser.error("logging-steps and save-total-limit must be positive")
    if args.rl_group_size < 2 or args.rl_sigma_start <= 0 or args.rl_sigma_end <= 0:
        parser.error("RL group size must be >= 2 and sigmas must be positive")
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
    """Create a training-only task with automatic uploads and stream capture off."""
    if not args.clearml or getattr(args, "prepare_only", False) or not is_primary_process():
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
        auto_connect_streams=False,
        auto_resource_monitoring=False,
    )
    print(f"ClearML task: {task.get_output_log_web_page()}")
    return task


def report_clearml_training_logs(logger: Any, logs: dict[str, Any], step: int) -> None:
    """Send only numeric trainer logs as scalars and text, without files or data."""
    values = {
        name: float(value) for name, value in logs.items()
        if isinstance(value, Real) and not isinstance(value, bool)
        and math.isfinite(float(value))
    }
    if not values:
        return
    for name, value in values.items():
        group = "test" if name.startswith("test_") else "train"
        logger.report_scalar(title=group, series=name, value=value, iteration=step)
    logger.report_text(
        f"step {step}: " + ", ".join(f"{name}={value:g}" for name, value in values.items()),
        print_console=False,
    )


def run(args: argparse.Namespace, clearml_task: Any = None) -> None:
    from datasets import Dataset
    from transformers import AutoTokenizer, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        raise ValueError("Qwen tokenizer has no padding token")
    start_id, close_id = marker_ids(tokenizer)

    output_dir = Path(args.output_dir)
    if not (args.full_dataset and args.prepare_only):
        output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "selection.json"
    if args.full_dataset:
        data_dir = Path(args.prepared_data_dir) if args.prepared_data_dir else output_dir
        train_path, test_path, selection_path = full_data_paths(data_dir)
        if args.prepare_only and not any(
            path.exists() for path in (train_path, test_path, selection_path)
        ):
            if int(os.environ.get("WORLD_SIZE", "1")) > 1:
                raise RuntimeError("Prepare full data with one Python process before torchrun")
            source = load_pollux_source(Path(args.source_dir))
            provenance = prepare_full_examples(
                source, tokenizer, output_dir=data_dir, model=args.model,
                seed=args.seed, max_length=args.max_length, max_options=args.max_options,
            )
        else:
            provenance = load_full_selection(
                data_dir, model=args.model, seed=args.seed,
                max_length=args.max_length, max_options=args.max_options,
            )
        train_count = provenance["splits"]["train"]["count"]
        test_count = provenance["splits"]["test"]["count"]
    else:
        source = load_pollux_source(Path(args.source_dir))
        source = source.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
        train_records, test_records, train_manifest, test_manifest = select_examples(
            source, tokenizer, samples=args.samples, test_samples=args.test_samples,
            seed=args.seed, max_length=args.max_length, max_options=args.max_options,
        )
        provenance = {
            "dataset": "ai-forever/POLLUX", "dataset_revision": DATASET_REVISION,
            "source_split": "test", "mode": "sample",
            "selected_examples": args.samples, "seed": args.seed,
            "max_length": args.max_length, "max_options": args.max_options,
            "test_task_types": list(TEST_TASK_TYPES),
            "test_type_quotas": make_test_quotas(args.test_samples),
            "laya_rl": args.laya_rl,
            "splits": {
                "train": summarize_split(train_manifest),
                "test": summarize_split(test_manifest),
            },
        }
        selection_path.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        train_count, test_count = len(train_records), len(test_records)
    print(f"Prepared {train_count} train and {test_count} test examples", flush=True)
    if args.prepare_only:
        return

    if args.full_dataset:
        train_dataset = Dataset.from_json(str(train_path))
        test_dataset = Dataset.from_json(str(test_path))
    else:
        train_dataset = Dataset.from_list(train_records)
        test_dataset = Dataset.from_list(test_records)

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
    model.config.judge_rlcd_group_size = args.rl_group_size
    model.config.judge_rlcd_sigma = args.rl_sigma_start

    training_args = ClassificationConfig(
        output_dir=str(output_dir),
        max_length=args.max_length,
        loss_type="cross_entropy",  # Halo delegates this loss to ToolCallJudge.
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = ToolCallClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorWithPadding(tokenizer, padding="longest"),
        processing_class=tokenizer,
        compute_metrics=compute_judge_metrics,
        is_binary=False,
        label_names_list=[str(i) for i in range(args.max_options)],
        parallelism_config=ParallelismConfig(),
    )

    if clearml_task is not None:
        class ClearMLTrainingCallback(TrainerCallback):
            def on_log(self, _args, state, control, logs=None, **_kwargs):
                if state.is_world_process_zero:
                    report_clearml_training_logs(
                        clearml_task.get_logger(), logs or {}, state.global_step
                    )
                return control

        trainer.add_callback(ClearMLTrainingCallback())

    if args.laya_rl:
        class LayaSigmaCallback(TrainerCallback):
            def on_epoch_begin(self, _args, state, control, **_kwargs):
                progress = min(1.0, max(0.0, (state.epoch or 0.0) / max(args.epochs - 1, 1)))
                model.config.judge_rlcd_sigma = (
                    args.rl_sigma_start + (args.rl_sigma_end - args.rl_sigma_start) * progress
                )
                return control

        trainer.add_callback(LayaSigmaCallback())

    trainer.train()
    metrics = trainer.evaluate(metric_key_prefix="test")
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    if trainer.is_world_process_zero():
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
