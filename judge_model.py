"""A Qwen3.5 backbone with one scalar logit at each score tool-call close."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3_5ForConditionalGeneration
from transformers.modeling_outputs import SequenceClassifierOutput


def laya_rlcd_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_options: torch.Tensor,
    option_values: torch.Tensor,
    *,
    sigma: float,
    group_size: int = 4,
) -> torch.Tensor:
    """Laya's centered Gaussian policy gradient with an ordinal score reward."""
    if sigma <= 0 or group_size < 2:
        raise ValueError("RLCD needs sigma > 0 and at least two noise samples")
    if option_values.shape != logits.shape:
        raise ValueError("RLCD option_values must have shape [batch, num_labels]")
    option_values = option_values.to(device=logits.device)
    if ((option_values < 0) & valid_options).any():
        raise ValueError("Valid candidates need nonnegative numeric scores")

    scores = logits.float()
    valid_float = valid_options.float()
    counts = valid_float.sum(dim=-1)
    noise = torch.randn(
        (group_size, *scores.shape), device=scores.device, dtype=scores.dtype
    ) * sigma * valid_float.unsqueeze(0)
    noise = (noise - noise.sum(dim=-1, keepdim=True) / counts[None, :, None])
    noise = noise * valid_float.unsqueeze(0)
    sampled_logits = scores.detach().unsqueeze(0) + noise
    q = F.softmax(sampled_logits.masked_fill(~valid_options.unsqueeze(0), -1e4), dim=-1)

    # The tool calls are shuffled, but ranked probability score needs the
    # numeric order 0, 1, ..., K-1. Padding is sorted after all real options.
    order = option_values.masked_fill(~valid_options, scores.shape[-1]).argsort(dim=-1)
    sorted_mask = valid_options.gather(-1, order)
    sorted_q = q.gather(-1, order.unsqueeze(0).expand(group_size, -1, -1))
    sorted_targets = targets.gather(-1, order)
    with torch.no_grad():
        log_score = (
            targets.unsqueeze(0) * q.clamp_min(1e-12).log().clamp_min(-9.21)
        ).sum(dim=-1)
        spherical = (targets.unsqueeze(0) * q).sum(dim=-1) / q.norm(
            dim=-1
        ).clamp_min(1e-9)
        rps = (
            ((sorted_q.cumsum(-1) - sorted_targets.unsqueeze(0).cumsum(-1)) ** 2)
            * sorted_mask.unsqueeze(0)
        ).sum(dim=-1) / (counts - 1).unsqueeze(0)
        reward = log_score + 0.75 * spherical - rps
        advantage = reward - reward.mean(dim=0, keepdim=True)
        advantage = advantage / (advantage.std() + 1e-6)

    log_density = -(
        ((sampled_logits - scores.unsqueeze(0)) ** 2) * valid_float.unsqueeze(0)
    ).sum(dim=-1) / (2 * sigma**2)
    return -(advantage * log_density).mean()


def normalize_soft_targets(labels: torch.Tensor, valid_options: torch.Tensor) -> torch.Tensor:
    """Validate assessor distributions, allowing only low-precision rounding."""
    targets = labels.float()
    if not torch.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("Soft labels must be finite nonnegative probabilities")
    if (targets.masked_select(~valid_options) != 0).any():
        raise ValueError("Padded candidates must have zero target probability")
    totals = targets.sum(dim=1, keepdim=True)
    # Data may be rounded before reaching this forward, even when the collator
    # later promotes it back to float32. One percent covers BF16 rounding only.
    if not torch.allclose(totals, torch.ones_like(totals), atol=0.01, rtol=0):
        raise ValueError(
            "Each soft-label distribution must sum to one; "
            f"got sums {totals.flatten().tolist()} with dtype {labels.dtype}"
        )
    return targets / totals


class ToolCallJudge(Qwen3_5ForConditionalGeneration):
    """Categorical score distribution over a variable count of candidates.

    ``config.num_labels`` is the padded output width.  The token-level head is
    always ``nn.Linear(hidden_size, 1)``; only closing tool-call positions are
    selected for the softmax and soft-target cross-entropy.
    """

    def __init__(self, config):
        super().__init__(config)
        self.score = nn.Linear(config.get_text_config().hidden_size, 1)
        self._init_weights(self.score)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.FloatTensor | None = None,
        option_values: torch.LongTensor | None = None,
        return_dict: bool = True,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids are required to find </tool_call> positions")
        close_id = getattr(self.config, "judge_tool_close_id", None)
        start_id = getattr(self.config, "judge_im_start_id", None)
        if close_id is None or start_id is None:
            raise ValueError("Checkpoint has no judge marker token IDs")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        hidden = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
        token_scores = self.score(hidden).squeeze(-1)

        sequence_positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        valid = attention_mask.bool()
        starts = (input_ids == start_id) & valid
        last_start = torch.where(starts, sequence_positions[None, :], -1).amax(dim=1)
        markers = (
            (input_ids == close_id)
            & valid
            & (sequence_positions[None, :] > last_start[:, None])
        )
        counts = markers.sum(dim=1)
        width = int(self.config.num_labels)
        if (last_start < 0).any() or (counts < 2).any() or (counts > width).any():
            raise ValueError("Expected 2..num_labels closing markers in final assistant turn")

        # The column is the candidate's position in this row, not its numeric
        # score.  Missing choices get zero probability after softmax.
        ranks = markers.long().cumsum(dim=1) - 1
        batch_idx, token_idx = markers.nonzero(as_tuple=True)
        logits = token_scores.new_full((input_ids.shape[0], width), -1e4)
        logits[batch_idx, ranks[batch_idx, token_idx]] = token_scores[batch_idx, token_idx]
        loss = None
        if labels is not None:
            if labels.shape != logits.shape:
                raise ValueError("Soft labels must have shape [batch, num_labels]")
            padding = torch.arange(width, device=logits.device)[None, :] >= counts[:, None]
            targets = normalize_soft_targets(labels.to(device=logits.device), ~padding)
            loss = F.cross_entropy(logits.float(), targets)
            if self.training and getattr(self.config, "judge_rlcd_enabled", False):
                if option_values is None:
                    raise ValueError("RLCD training requires option_values")
                loss = loss + laya_rlcd_loss(
                    logits, targets, ~padding, option_values,
                    sigma=float(self.config.judge_rlcd_sigma),
                    group_size=int(self.config.judge_rlcd_group_size),
                )

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        return SequenceClassifierOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def predict_proba(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None = None):
        """Return softmax probabilities in tool-call order (zero for padding)."""
        logits = self.forward(input_ids=input_ids, attention_mask=attention_mask).logits
        return F.softmax(logits.float(), dim=-1)
