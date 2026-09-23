"""Held-out metrics for ordered score candidates and soft assessor targets."""

from __future__ import annotations

import math
from typing import Any, Sequence


def score_metrics(
    logits: Sequence[Sequence[float]], targets: Sequence[Sequence[float]]
) -> dict[str, float]:
    """Compare argmax scores with the assessors' majority score."""
    if len(logits) != len(targets) or not len(logits):
        raise ValueError("Metrics need equally sized, nonempty predictions and targets")

    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    predicted_classes: list[int] = []
    target_classes: list[int] = []
    for row_logits, row_targets in zip(logits, targets):
        if len(row_logits) != len(row_targets) or not len(row_logits):
            raise ValueError("Logit and target widths must match")
        predicted_score = max(range(len(row_logits)), key=lambda i: float(row_logits[i]))
        majority_score = max(range(len(row_targets)), key=lambda i: float(row_targets[i]))
        error = predicted_score - majority_score
        absolute_errors.append(abs(error))
        squared_errors.append(error * error)
        predicted_classes.append(predicted_score)
        target_classes.append(majority_score)

    classes = set(predicted_classes) | set(target_classes)
    class_f1 = []
    for value in classes:
        true_positive = sum(
            predicted == target == value
            for predicted, target in zip(predicted_classes, target_classes)
        )
        false_positive = sum(
            predicted == value and target != value
            for predicted, target in zip(predicted_classes, target_classes)
        )
        false_negative = sum(
            predicted != value and target == value
            for predicted, target in zip(predicted_classes, target_classes)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        class_f1.append(2 * true_positive / denominator if denominator else 0.0)

    return {
        "mae": sum(absolute_errors) / len(absolute_errors),
        "rmse": math.sqrt(sum(squared_errors) / len(squared_errors)),
        "f1_macro": sum(class_f1) / len(class_f1),
    }


def compute_judge_metrics(eval_pred: Any) -> dict[str, float]:
    """Halo/Hugging Face Trainer callback; test calls are ordered 0..K-1."""
    predictions = eval_pred.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    return score_metrics(predictions, eval_pred.label_ids)
