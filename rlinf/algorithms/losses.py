# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional

import torch

from rlinf.algorithms.registry import register_policy_loss
from rlinf.algorithms.utils import huber_loss
from rlinf.utils.utils import masked_mean, masked_mean_ratio


def _masked_values(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        selected = values.reshape(-1)
    else:
        if mask.shape != values.shape:
            mask = mask.expand_as(values)
        selected = values.masked_select(mask)
    if selected.numel() == 0:
        selected = values.new_zeros(1)
    return selected.float()


def compute_decoupled_ppo_actor_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    advantages: torch.Tensor,
    proximal_logprobs: Optional[torch.Tensor] = None,
    versions: Optional[torch.Tensor] = None,
    current_version: Optional[float] = None,
    loss_mask: Optional[torch.Tensor] = None,
    clip_ratio_c: Optional[float] = None,
    loss_agg_func: Optional[Callable[..., torch.Tensor]] = masked_mean,
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    critic_warmup: Optional[bool] = False,
    behave_weight_threshold: Optional[float] = None,
    debug_metrics: Optional[dict] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """Compute actor loss for decoupled PPO with optional proximal policy anchor."""
    assert logprobs.dtype == torch.float32, (
        "logprobs must be float32 to keep numerical stability"
    )
    assert old_logprobs.dtype == torch.float32, (
        "old_logprobs must be float32 to keep numerical stability"
    )
    assert advantages.dtype == torch.float32, (
        "advantages must be float32 to keep numerical stability"
    )

    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs).bool()

    loss_mask_ratio = None
    if (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    ):
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    if proximal_logprobs is None:
        if versions is None or current_version is None:
            proximal_logprobs = old_logprobs.detach()
        else:
            v_behav = versions.float()
            v_theta = float(current_version)
            v_prox = v_theta - 1.0

            version_diff = v_theta - v_behav
            version_gap = v_prox - v_behav
            generated_tokens_mask = versions >= 0
            alpha = torch.where(
                (version_diff > 0) & generated_tokens_mask,
                version_gap / version_diff,
                torch.zeros_like(v_behav),
            )
            while alpha.dim() < logprobs.dim():
                alpha = alpha.unsqueeze(-1)
            alpha = torch.clamp(alpha, 0.0, 1.0)
            proximal_logprobs = (
                old_logprobs + alpha * (logprobs - old_logprobs)
            ).detach()

    assert proximal_logprobs.dtype == torch.float32, (
        "proximal_logprobs must be float32 to keep numerical stability"
    )

    loss_mask_count = loss_mask.count_nonzero() or 1
    proximal_ratio = torch.where(
        loss_mask, torch.exp(logprobs - proximal_logprobs), 0.0
    )
    clipped_proximal_ratio = torch.clamp(
        proximal_ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high
    )

    pg_loss1 = -advantages * proximal_ratio
    pg_loss2 = -advantages * clipped_proximal_ratio
    pg_loss = torch.max(pg_loss1, pg_loss2)

    if clip_ratio_c is not None:
        assert clip_ratio_c > 1.0, clip_ratio_c
        pg_loss3 = torch.sign(advantages) * clip_ratio_c * advantages
        dual_clip_mask = pg_loss3.detach() < pg_loss.detach()
        pg_loss = torch.min(pg_loss, pg_loss3)
    else:
        dual_clip_mask = torch.zeros_like(pg_loss, dtype=torch.bool)

    behav_weight = torch.exp(proximal_logprobs - old_logprobs)
    behav_mask = (
        (behav_weight <= behave_weight_threshold).logical_and(loss_mask)
        if behave_weight_threshold is not None
        else loss_mask
    )
    behav_mask_count = behav_mask.count_nonzero() or 1

    pg_loss = loss_agg_func(pg_loss * behav_weight, behav_mask, loss_mask_ratio)
    if critic_warmup:
        pg_loss = torch.tensor(0.0, device=pg_loss.device)

    with torch.no_grad():
        clip_fraction = (pg_loss1 < pg_loss2).logical_and(
            loss_mask
        ).count_nonzero() / loss_mask_count
        dual_clip_fraction = (
            dual_clip_mask.logical_and(loss_mask).count_nonzero() / loss_mask_count
        )
        proximal_approx_kl = (
            -torch.where(loss_mask, logprobs - proximal_logprobs, 0.0).sum()
            / loss_mask_count
        )
        behav_approx_kl = (
            -torch.where(behav_mask, proximal_logprobs - old_logprobs, 0.0).sum()
            / behav_mask_count
        )
        behav_clip_fraction = 1.0 - (behav_mask_count / loss_mask_count)

        masked_logprobs = _masked_values(logprobs.detach(), loss_mask)
        masked_old_logprobs = _masked_values(old_logprobs.detach(), loss_mask)
        masked_proximal_logprobs = _masked_values(proximal_logprobs.detach(), loss_mask)
        masked_advantages = _masked_values(advantages.detach(), loss_mask)
        masked_versions = (
            _masked_values(versions.detach(), loss_mask) if versions is not None else None
        )

    metrics_data = {
        "actor/policy_loss": pg_loss.detach(),
        "actor/proximal_ratio": masked_mean(proximal_ratio.detach(), loss_mask),
        "actor/clipped_proximal_ratio": masked_mean(
            clipped_proximal_ratio.detach(), loss_mask
        ),
        "actor/clip_fraction": clip_fraction,
        "actor/dual_clip_fraction": dual_clip_fraction,
        "actor/behav_clip_fraction": behav_clip_fraction,
        "actor/proximal_approx_kl": proximal_approx_kl,
        "actor/behav_approx_kl": behav_approx_kl,
    }
    if (
        versions is not None
        and current_version is not None
        and versions.shape == loss_mask.shape
        and loss_mask.any()
    ):
        metrics_data["actor/average_version"] = versions[loss_mask].float().mean()
        metrics_data["actor/current_version"] = torch.tensor(
            float(current_version), device=logprobs.device
        )

    if debug_metrics is not None:
        for key in (
            "actor/debug_logprob_prox_gap_mean",
            "actor/debug_prox_old_gap_mean",
        ):
            if key in debug_metrics:
                metrics_data[key] = debug_metrics[key]

    return pg_loss, metrics_data


def compute_ppo_actor_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    advantages: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    clip_ratio_c: Optional[float] = None,
    loss_agg_func: Optional[Callable[..., torch.Tensor]] = masked_mean,
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    critic_warmup: Optional[bool] = False,
    clip_log_ratio_min: Optional[float] = None,
    clip_log_ratio_max: Optional[float] = None,
    fast_path_zero_loss_mask: Optional[bool] = False,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO actor loss function.

    Args:
        logprobs (torch.FloatTensor): Log probabilities of actions.
        old_logprobs (torch.FloatTensor): Old log probabilities of actions.
        clip_ratio_low (float): Lower bound of clipping ratio.
        clip_ratio_high (float): Upper bound of clipping ratio.
        advantages (torch.FloatTensor): GAE (normalized) advantages.
        loss_mask (Optional[torch.BoolTensor], optional): Mask for valid entries. Defaults to None.
        clip_ratio_c (Optional[float], optional): Optional clipping coefficient. Defaults to None.
        loss_agg_func (callable, optional): Aggregation function (e.g., masked_mean). Defaults to None.
        max_episode_steps (Optional[int], optional): Max episode length for normalization. Defaults to None.

    Returns:
        Tuple[torch.Tensor, Dict]: (actor_loss, metrics_dict)
    """
    if fast_path_zero_loss_mask and (
        loss_mask is not None and loss_mask[0].sum() == 0.0
    ):
        return torch.tensor(0.0, device=logprobs.device), {
            "actor/token_num": torch.tensor(0.0, device=logprobs.device),
            "actor/policy_loss": torch.tensor(0.0, device=logprobs.device),
            "actor/policy_loss_mbs_mean": torch.tensor(0.0, device=logprobs.device),
            "actor/policy_loss_abs": torch.tensor(0.0, device=logprobs.device),
            "actor/ratio": torch.tensor(0.0, device=logprobs.device),
            "actor/clipped_ratio": torch.tensor(0.0, device=logprobs.device),
            "actor/dual_cliped_ratio": torch.tensor(0.0, device=logprobs.device),
            "actor/approx_kl": torch.tensor(0.0, device=logprobs.device),
            "actor/clip_fraction": torch.tensor(0.0, device=logprobs.device),
            "actor/log_ratio_mean": torch.tensor(0.0, device=logprobs.device),
            "actor/log_ratio_min": torch.tensor(0.0, device=logprobs.device),
            "actor/log_ratio_max": torch.tensor(0.0, device=logprobs.device),
            "actor/ratio_min": torch.tensor(0.0, device=logprobs.device),
            "actor/ratio_max": torch.tensor(0.0, device=logprobs.device),
            "actor/clip_lower_fraction": torch.tensor(0.0, device=logprobs.device),
            "actor/clip_upper_fraction": torch.tensor(0.0, device=logprobs.device),
            "actor/advantage_mean": torch.tensor(0.0, device=logprobs.device),
            "actor/advantage_positive_fraction": torch.tensor(
                0.0, device=logprobs.device
            ),
        }

    loss_mask_ratio = None

    if (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    ):
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs).bool()

    assert logprobs.dtype == torch.float32, (
        "logprobs must be float32 to keep numerical stability"
    )
    assert old_logprobs.dtype == torch.float32, (
        "old_logprobs must be float32 to keep numerical stability"
    )
    assert advantages.dtype == torch.float32, (
        "advantages must be float32 to keep numerical stability"
    )

    loss_mask_count = loss_mask.count_nonzero() or 1
    # For numerical stability.
    log_ratio = logprobs - old_logprobs
    if clip_log_ratio_min is not None:
        log_ratio = torch.clamp(log_ratio, min=clip_log_ratio_min)
    if clip_log_ratio_max is not None:
        log_ratio = torch.clamp(log_ratio, max=clip_log_ratio_max)
    ratio = torch.where(loss_mask, torch.exp(log_ratio), 0)
    approx_kl = torch.where(loss_mask, log_ratio.detach(), 0.0)

    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    policy_loss1 = -advantages * ratio
    policy_loss2 = -advantages * clipped_ratio

    clip_mask = policy_loss1.detach() < policy_loss2.detach()

    policy_loss = torch.max(policy_loss1, policy_loss2)
    if clip_ratio_c is not None:
        assert clip_ratio_c > 1.0, "clip_ratio_c must be greater than 1.0"
        policy_loss3 = torch.sign(advantages) * clip_ratio_c * advantages
        dual_clip_mask = policy_loss3.detach() < policy_loss.detach()
        policy_loss = torch.min(policy_loss, policy_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_mask)

    metric_policy_loss_abs = loss_agg_func(
        policy_loss.abs(), loss_mask, loss_mask_ratio
    )
    policy_loss = loss_agg_func(
        policy_loss, loss_mask, loss_mask_ratio
    )  # default max_episode_steps is None

    clip_mask = policy_loss1.detach() < policy_loss2.detach()
    dual_clip_mask = (dual_clip_mask * loss_mask).bool()

    clip_fraction = (clip_mask * loss_mask).sum() / float(loss_mask_count)
    approx_kl = -torch.sum(approx_kl) / float(loss_mask_count)

    dual_cliped_ratio = torch.where(dual_clip_mask, ratio, 0)

    if critic_warmup:
        policy_loss = torch.tensor(0.0, device=policy_loss.device)

    # Compile metrics for logging
    loss_mask_for_metrics = loss_mask
    ratio_for_metrics = ratio.detach()
    ratio_abs_for_metrics = (ratio - 1).abs().detach()
    clipped_ratio_for_metrics = clipped_ratio.detach()
    dual_cliped_ratio_for_metrics = dual_cliped_ratio.detach()
    log_ratio_for_metrics = log_ratio.detach()
    advantages_for_metrics = advantages.detach()

    # Only broadcast when ratio has action_dim dimension and loss_mask's last dim is 1
    # This handles token_level mode: ratio [bsz, num_chunks, action_dim], loss_mask [bsz, num_chunks, 1]
    if len(ratio.shape) > 2 and loss_mask.shape[-1] == 1 and ratio.shape[-1] > 1:
        # Broadcast loss_mask to match ratio's shape for metrics computation
        loss_mask_for_metrics = loss_mask.expand_as(ratio)

    selected_ratio = _masked_values(ratio_for_metrics, loss_mask_for_metrics)
    selected_log_ratio = _masked_values(log_ratio_for_metrics, loss_mask_for_metrics)
    selected_advantages = _masked_values(advantages_for_metrics, loss_mask_for_metrics)

    metrics_data = {
        "actor/token_num": torch.tensor(
            float(loss_mask_count), device=logprobs.device, dtype=torch.float32
        ),
        "actor/policy_loss": policy_loss.detach(),
        "actor/policy_loss_abs": metric_policy_loss_abs.detach(),
        "actor/ratio": masked_mean(ratio_for_metrics, loss_mask_for_metrics),
        "actor/ratio_abs": masked_mean(ratio_abs_for_metrics, loss_mask_for_metrics),
        "actor/clipped_ratio": masked_mean(
            clipped_ratio_for_metrics, loss_mask_for_metrics
        ),
        "actor/dual_cliped_ratio": masked_mean(
            dual_cliped_ratio_for_metrics, loss_mask_for_metrics
        ),
        "actor/approx_kl": approx_kl.detach(),
        "actor/clip_fraction": clip_fraction.detach(),
        "actor/log_ratio_mean": selected_log_ratio.mean(),
        "actor/log_ratio_min": selected_log_ratio.min(),
        "actor/log_ratio_max": selected_log_ratio.max(),
        "actor/ratio_min": selected_ratio.min(),
        "actor/ratio_max": selected_ratio.max(),
        "actor/clip_lower_fraction": (
            (selected_ratio < (1.0 - clip_ratio_low)).float().mean()
        ),
        "actor/clip_upper_fraction": (
            (selected_ratio > (1.0 + clip_ratio_high)).float().mean()
        ),
        "actor/advantage_mean": selected_advantages.mean(),
        "actor/advantage_positive_fraction": (
            (selected_advantages > 0).float().mean()
        ),
    }
    return policy_loss, metrics_data


def compute_ppo_critic_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    prev_values: torch.Tensor,
    value_clip: float,
    huber_delta: float,
    loss_mask: Optional[torch.Tensor] = None,
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO critic loss function.

    Args:
        values (torch.Tensor): Current value predictions.
        returns (torch.Tensor): Return values.
        prev_values (torch.Tensor): Previous value predictions.
        value_clip (float): Value clipping threshold.
        huber_delta (float): Huber loss delta parameter.

    Returns:
        Tuple[torch.Tensor, Dict]: (critic_loss, metrics_dict)
    """
    loss_mask_ratio = None
    loss_agg_func = masked_mean

    if (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    ):
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    value_pred_clipped = prev_values + (values - prev_values).clamp(
        -value_clip, value_clip
    )  # [bsz, ] | [bsz, chunk-step]

    value_loss_original = huber_loss(
        returns - values, huber_delta
    )  # [bsz, ] | [bsz, chunk-step]
    value_loss_clipped = huber_loss(
        returns - value_pred_clipped, huber_delta
    )  # [bsz, ] | [bsz, chunk-step]
    value_loss = torch.max(value_loss_original, value_loss_clipped)
    value_loss = loss_agg_func(value_loss, loss_mask, loss_mask_ratio)

    value_clip_indicator = (value_pred_clipped - prev_values).abs() > value_clip
    value_clip_ratio = value_clip_indicator.float().mean()

    # explained variance
    if loss_mask is not None:
        masked_returns = returns[loss_mask]
        masked_values = values[loss_mask]
    else:
        masked_returns = returns
        masked_values = values

    var_returns = torch.var(masked_returns)
    if torch.isnan(var_returns) or var_returns == 0:
        explained_variance = torch.tensor(float("nan"), device=returns.device)
    else:
        var_diff = torch.var(masked_returns - masked_values)
        if torch.isnan(var_diff):
            explained_variance = torch.tensor(float("nan"), device=returns.device)
        else:
            explained_variance = 1 - var_diff / var_returns

    # Compile metrics for logging
    metrics_data = {
        "critic/value_loss": value_loss.detach(),
        "critic/value_clip_ratio": value_clip_ratio.detach(),
        "critic/explained_variance": explained_variance.detach(),
    }
    return value_loss, metrics_data


@register_policy_loss("decoupled_actor_critic")
def compute_decoupled_ppo_actor_critic_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    """Compute decoupled PPO actor+critic loss."""
    metrics_data = {}
    actor_loss, actor_metrics_data = compute_decoupled_ppo_actor_loss(**kwargs)
    critic_loss, critic_metrics_data = compute_ppo_critic_loss(**kwargs)

    loss = actor_loss + critic_loss
    metrics_data.update(actor_metrics_data)
    metrics_data.update(critic_metrics_data)
    return loss, metrics_data


@register_policy_loss("actor_critic")
def compute_ppo_actor_critic_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO actor loss function.

    Args:
        logprobs (torch.Tensor): Log probabilities of actions
        values (torch.Tensor): Current value predictions
        old_log_prob (torch.Tensor): Previous log probabilities
        advantages (torch.Tensor): Advantage values
        returns (torch.Tensor): Return values
        prev_values (torch.Tensor): Previous value predictions
        clip_ratio_low (float): Lower clipping ratio for PPO
        clip_ratio_high (float): Upper clipping ratio for PPO
        value_clip (float): Value clipping threshold
        huber_delta (float): Huber loss delta parameter

    Returns:
        Tuple[torch.Tensor, Dict]: Loss and metrics dictionary
    """
    metrics_data = {}
    actor_loss, actor_metrics_data = compute_ppo_actor_loss(**kwargs)
    critic_loss, critic_metrics_data = compute_ppo_critic_loss(**kwargs)

    loss = actor_loss + critic_loss
    metrics_data.update(actor_metrics_data)
    metrics_data.update(critic_metrics_data)

    return loss, metrics_data


@register_policy_loss("actor")
def compute_grpo_actor_loss_fn(**kwargs) -> tuple[torch.Tensor, dict]:
    """
    Compute actor loss for Group Relative Policy Optimization (GRPO).

    This function implements the PPO-style actor loss with clipping for GRPO.
    Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppotrainer.py#L1122

    Args:
        log_prob (torch.Tensor): Current log probabilities
        old_log_prob (torch.Tensor): Previous log probabilities
        advantages (torch.Tensor): Advantage values of shape
        clip_ratio_high (float): Upper clipping ratio for PPO
        clip_ratio_low (float): Lower clipping ratio for PPO
        loss_mask (Optional[torch.Tensor]): Mask tensor of shape to apply to the loss

    Returns:
        Tuple[torch.Tensor, Dict]: Policy gradient loss and metrics dictionary containing:
            - actor/loss: Total actor loss
            - actor/policy_loss: Policy gradient loss
            - actor/clip_fraction: Fraction of clipped policy gradient loss
            - actor/ppo_kl: Approximate KL divergence
    """
    metrics_data = {}
    actor_loss, actor_metrics_data = compute_ppo_actor_loss(**kwargs)
    metrics_data.update(actor_metrics_data)

    return actor_loss, metrics_data


@register_policy_loss("gspo")
def compute_gspo_actor_loss_fn(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
    kl_beta: float = 0.02,
    log_ratio_clip: float = 20.0,
    loss_agg_func: Optional[Callable[..., torch.Tensor]] = masked_mean,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs).bool()

    if logprobs.ndim == 3 and loss_mask.shape[-1] == 1:
        loss_mask = loss_mask.expand_as(logprobs)

    reduce_dims = tuple(range(1, logprobs.ndim))
    log_diff = torch.where(
        loss_mask,
        logprobs - old_logprobs,
        torch.zeros_like(logprobs),
    )

    token_count = loss_mask.sum(dim=reduce_dims, keepdim=True).clamp(min=1)
    seq_log_ratio = log_diff.sum(dim=reduce_dims, keepdim=True) / token_count
    seq_log_ratio = torch.clamp(
        seq_log_ratio,
        -log_ratio_clip,
        log_ratio_clip,
    )
    ratio = torch.exp(seq_log_ratio)

    if advantages.ndim != logprobs.ndim:
        advantages = advantages.expand_as(logprobs)

    adv_masked = torch.where(
        loss_mask,
        advantages,
        torch.zeros_like(advantages),
    )
    seq_adv = adv_masked.sum(dim=reduce_dims, keepdim=True) / token_count

    clipped_ratio = torch.clamp(
        ratio,
        1.0 - clip_ratio_low,
        1.0 + clip_ratio_high,
    )
    policy_loss1 = -seq_adv * ratio
    policy_loss2 = -seq_adv * clipped_ratio
    policy_loss = torch.max(policy_loss1, policy_loss2)

    seq_mask = token_count > 0
    policy_loss_mean = loss_agg_func(policy_loss, seq_mask)

    seq_log_ratio_sum = log_diff.sum(dim=reduce_dims, keepdim=True)
    approx_kl = 0.5 * loss_agg_func(
        seq_log_ratio_sum.pow(2),
        seq_mask,
    )

    pure_policy_loss = policy_loss_mean.clone()
    policy_loss_mean = policy_loss_mean + kl_beta * approx_kl

    with torch.no_grad():
        clip_mask = (ratio > 1.0 + clip_ratio_high) | (ratio < 1.0 - clip_ratio_low)
        clip_fraction = loss_agg_func(
            clip_mask.float(),
            seq_mask,
        )
        metrics_data = {
            "actor/policy_loss": pure_policy_loss,
            "actor/total_loss": policy_loss_mean.detach(),
            "actor/ratio": loss_agg_func(ratio, seq_mask).detach(),
            "actor/clipped_ratio": loss_agg_func(clipped_ratio, seq_mask).detach(),
            "actor/approx_kl": approx_kl.detach(),
            "actor/clip_fraction": clip_fraction.detach(),
            "actor/seq_log_ratio_mean": loss_agg_func(
                seq_log_ratio,
                seq_mask,
            ).detach(),
            "actor/seq_adv_mean": loss_agg_func(
                seq_adv,
                seq_mask,
            ).detach(),
        }

    return policy_loss_mean, metrics_data
