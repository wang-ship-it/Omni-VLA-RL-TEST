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

import builtins
import os
from typing import Callable, Optional

import torch

from rlinf.algorithms.registry import register_policy_loss
from rlinf.algorithms.utils import huber_loss
from rlinf.utils.utils import masked_mean, masked_mean_ratio


_DEBUG_VERBOSE = (
    os.getenv("OMNI_VLA_DEBUG_LOG", "0") == "1"
    or os.getenv("OMNI_VLA_DEBUG_LOSS", "0") == "1"
)
_DEBUG_EVERY = max(1, int(os.getenv("OMNI_VLA_DEBUG_EVERY", "1")))
_DEBUG_COUNTER = 0


def print(*args, **kwargs):
    global _DEBUG_COUNTER
    if not _DEBUG_VERBOSE:
        return
    _DEBUG_COUNTER += 1
    if _DEBUG_COUNTER % _DEBUG_EVERY == 0:
        builtins.print(*args, **kwargs)


def _debug_tensor_health(name: str, tensor: Optional[torch.Tensor], topk: int = 5) -> None:
    """Print concise NaN/Inf diagnostics for a tensor."""
    if tensor is None:
        print(f"[DEBUG NAN CHECK] {name}: None")
        return

    t = tensor.detach()
    nan_mask = torch.isnan(t)
    inf_mask = torch.isinf(t)
    nan_count = nan_mask.sum().item()
    inf_count = inf_mask.sum().item()
    total_count = t.numel()
    print(
        f"[DEBUG NAN CHECK] {name}: shape={tuple(t.shape)}, total={total_count}, "
        f"nan_count={nan_count}, inf_count={inf_count}"
    )

    bad_mask = nan_mask | inf_mask
    if bad_mask.any():
        bad_indices = torch.nonzero(bad_mask, as_tuple=False)
        show_count = min(topk, bad_indices.shape[0])
        print(f"[DEBUG NAN CHECK] {name}: first {show_count} bad entries:")
        for i in range(show_count):
            idx = tuple(bad_indices[i].tolist())
            print(f"[DEBUG NAN CHECK]   {name}{idx} = {t[idx]}")


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
    print("\n" + "="*80)
    print("[DEBUG PPO ACTOR] ====== ENTERING compute_ppo_actor_loss ======")
    print(f"[DEBUG PPO ACTOR] logprobs.shape: {logprobs.shape}")
    print(f"[DEBUG PPO ACTOR] logprobs stats: min={logprobs.min().item():.6f}, max={logprobs.max().item():.6f}, mean={logprobs.mean().item():.6f}")
    print(f"[DEBUG PPO ACTOR] old_logprobs.shape: {old_logprobs.shape}")
    print(f"[DEBUG PPO ACTOR] old_logprobs stats: min={old_logprobs.min().item():.6f}, max={old_logprobs.max().item():.6f}, mean={old_logprobs.mean().item():.6f}")
    print(f"[DEBUG PPO ACTOR] advantages.shape: {advantages.shape}")
    print(f"[DEBUG PPO ACTOR] advantages stats: min={advantages.min().item():.6f}, max={advantages.max().item():.6f}, mean={advantages.mean().item():.6f}")
    print(f"[DEBUG PPO ACTOR] clip_ratio_low: {clip_ratio_low}, clip_ratio_high: {clip_ratio_high}")
    if clip_ratio_c is not None:
        print(f"[DEBUG PPO ACTOR] clip_ratio_c: {clip_ratio_c}")
    print(f"[DEBUG PPO ACTOR] loss_mask.shape: {loss_mask.shape if loss_mask is not None else None}")
    print(f"[DEBUG PPO ACTOR] critic_warmup: {critic_warmup}")
    if clip_log_ratio_min is not None:
        print(f"[DEBUG PPO ACTOR] clip_log_ratio_min: {clip_log_ratio_min}")
    if clip_log_ratio_max is not None:
        print(f"[DEBUG PPO ACTOR] clip_log_ratio_max: {clip_log_ratio_max}")

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

    assert logprobs.dtype == torch.float32
    assert old_logprobs.dtype == torch.float32
    assert advantages.dtype == torch.float32

    loss_mask_count = loss_mask.count_nonzero() or 1
    log_ratio = logprobs - old_logprobs
    print(f"[DEBUG PPO ACTOR] log_ratio stats: min={log_ratio.min().item():.6f}, max={log_ratio.max().item():.6f}, mean={log_ratio.mean().item():.6f}")
    
    if clip_log_ratio_min is not None:
        log_ratio = torch.clamp(log_ratio, min=clip_log_ratio_min)
        print(f"[DEBUG PPO ACTOR] log_ratio AFTER clip_log_ratio_min: min={log_ratio.min().item():.6f}")
    if clip_log_ratio_max is not None:
        log_ratio = torch.clamp(log_ratio, max=clip_log_ratio_max)
        print(f"[DEBUG PPO ACTOR] log_ratio AFTER clip_log_ratio_max: max={log_ratio.max().item():.6f}")
        
    ratio = torch.where(loss_mask, torch.exp(log_ratio), 0)
    print(f"[DEBUG PPO ACTOR] ratio stats: min={ratio.min().item():.6f}, max={ratio.max().item():.6f}, mean={ratio.mean().item():.6f}")
    print(f"[DEBUG PPO ACTOR] ratio (1.0 - ratio) stats: min={(1.0-ratio).min().item():.6f}, max={(1.0-ratio).max().item():.6f}")
    
    approx_kl = torch.where(loss_mask, log_ratio.detach(), 0.0)

    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    print(f"[DEBUG PPO ACTOR] clipped_ratio stats: min={clipped_ratio.min().item():.6f}, max={clipped_ratio.max().item():.6f}")
    
    policy_loss1 = -advantages * ratio
    policy_loss2 = -advantages * clipped_ratio
    print(f"[DEBUG PPO ACTOR] policy_loss1 stats: min={policy_loss1.min().item():.6f}, max={policy_loss1.max().item():.6f}")
    print(f"[DEBUG PPO ACTOR] policy_loss2 stats: min={policy_loss2.min().item():.6f}, max={policy_loss2.max().item():.6f}")

    clip_mask = policy_loss1.detach() < policy_loss2.detach()
    print(f"[DEBUG PPO ACTOR] clip_mask true count: {clip_mask.sum().item()}")

    policy_loss = torch.max(policy_loss1, policy_loss2)
    print(f"[DEBUG PPO ACTOR] policy_loss (after max) stats: min={policy_loss.min().item():.6f}, max={policy_loss.max().item():.6f}")
    
    if clip_ratio_c is not None:
        assert clip_ratio_c > 1.0, clip_ratio_c
        policy_loss3 = torch.sign(advantages) * clip_ratio_c * advantages
        dual_clip_mask = policy_loss3.detach() < policy_loss.detach()
        print(f"[DEBUG PPO ACTOR] policy_loss3 stats: min={policy_loss3.min().item():.6f}, max={policy_loss3.max().item():.6f}")
        print(f"[DEBUG PPO ACTOR] dual_clip_mask true count: {dual_clip_mask.sum().item()}")
        policy_loss = torch.min(policy_loss, policy_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_mask)

    metric_policy_loss_abs = loss_agg_func(
        policy_loss.abs(), loss_mask, loss_mask_ratio
    )
    policy_loss = loss_agg_func(
        policy_loss, loss_mask, loss_mask_ratio
    )
    print(f"[DEBUG PPO ACTOR] Final policy_loss (after aggregation): {policy_loss.item():.6f}")

    clip_mask = policy_loss1.detach() < policy_loss2.detach()
    dual_clip_mask = (dual_clip_mask * loss_mask).bool()

    clip_fraction = (clip_mask * loss_mask).sum() / float(loss_mask_count)
    approx_kl = -torch.sum(approx_kl) / float(loss_mask_count)

    dual_cliped_ratio = torch.where(dual_clip_mask, ratio, 0)

    if critic_warmup:
        policy_loss = torch.tensor(0.0, device=policy_loss.device)
        print("[DEBUG PPO ACTOR] CRITIC WARMUP MODE - policy_loss set to 0")

    print(f"[DEBUG PPO ACTOR] clip_fraction: {clip_fraction.item():.6f}")
    print(f"[DEBUG PPO ACTOR] approx_kl: {approx_kl.item():.6f}")
    print("[DEBUG PPO ACTOR] ====== EXITING ======" + "\n" + "="*80 + "\n")

    # Compile metrics for logging
    loss_mask_for_metrics = loss_mask
    ratio_for_metrics = ratio.detach()
    ratio_abs_for_metrics = (ratio - 1).abs().detach()
    clipped_ratio_for_metrics = clipped_ratio.detach()
    dual_cliped_ratio_for_metrics = dual_cliped_ratio.detach()

    # Only broadcast when ratio has action_dim dimension and loss_mask's last dim is 1
    # This handles token_level mode: ratio [bsz, num_chunks, action_dim], loss_mask [bsz, num_chunks, 1]
    if len(ratio.shape) > 2 and loss_mask.shape[-1] == 1 and ratio.shape[-1] > 1:
        # Broadcast loss_mask to match ratio's shape for metrics computation
        loss_mask_for_metrics = loss_mask.expand_as(ratio)

    metrics_data = {
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
    print("\n" + "="*80)
    print("[DEBUG PPO CRITIC LOSS] ====== ENTERING compute_ppo_critic_loss ======")
    print(f"[DEBUG] values.shape: {values.shape}, dtype: {values.dtype}")
    print(f"[DEBUG] values stats: min={values.min().item():.6f}, max={values.max().item():.6f}, mean={values.mean().item():.6f}, std={values.std().item():.6f}")
    print(f"[DEBUG] values has_nan: {torch.isnan(values).any().item()}, has_inf: {torch.isinf(values).any().item()}")
    print(f"[DEBUG] returns.shape: {returns.shape}, dtype: {returns.dtype}")
    print(f"[DEBUG] returns stats: min={returns.min().item():.6f}, max={returns.max().item():.6f}, mean={returns.mean().item():.6f}, std={returns.std().item():.6f}")
    print(f"[DEBUG] returns has_nan: {torch.isnan(returns).any().item()}, has_inf: {torch.isinf(returns).any().item()}")
    print(f"[DEBUG] prev_values.shape: {prev_values.shape if prev_values is not None else None}")
    if prev_values is not None:
        print(f"[DEBUG] prev_values stats: min={prev_values.min().item():.6f}, max={prev_values.max().item():.6f}, mean={prev_values.mean().item():.6f}")
        print(f"[DEBUG] prev_values has_nan: {torch.isnan(prev_values).any().item()}, has_inf: {torch.isinf(prev_values).any().item()}")
    print(f"[DEBUG] loss_mask.shape: {loss_mask.shape if loss_mask is not None else None}")
    print(f"[DEBUG] value_clip: {value_clip}, huber_delta: {huber_delta}")
    
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
    print(f"[DEBUG] value_pred_clipped.shape: {value_pred_clipped.shape}")
    print(f"[DEBUG] (values - prev_values).shape: {(values - prev_values).shape}")
    print(f"[DEBUG] (values - prev_values) stats: min={(values - prev_values).min().item():.6f}, max={(values - prev_values).max().item():.6f}")

    value_loss_original = huber_loss(
        returns - values, huber_delta
    )  # [bsz, ] | [bsz, chunk-step]
    print(f"[DEBUG] value_loss_original.shape: {value_loss_original.shape}")
    print(f"[DEBUG] value_loss_original stats: min={value_loss_original.min().item():.6f}, max={value_loss_original.max().item():.6f}")
    print(f"[DEBUG] (returns - values) stats: min={(returns - values).min().item():.6f}, max={(returns - values).max().item():.6f}")
    
    value_loss_clipped = huber_loss(
        returns - value_pred_clipped, huber_delta
    )  # [bsz, ] | [bsz, chunk-step]
    value_loss = torch.max(value_loss_original, value_loss_clipped)
    value_loss = loss_agg_func(value_loss, loss_mask, loss_mask_ratio)
    print(f"[DEBUG] value_loss (after max & aggregation): {value_loss.item():.6f}")

    value_clip_indicator = (value_pred_clipped - prev_values).abs() > value_clip
    value_clip_ratio = value_clip_indicator.float().mean()
    print(f"[DEBUG] value_clip_ratio: {value_clip_ratio.item():.6f}")

    # explained variance
    print("\n[DEBUG] ====== Computing explained_variance ======")
    if loss_mask is not None:
        masked_returns = returns[loss_mask]
        masked_values = values[loss_mask]
        print(f"[DEBUG] loss_mask true count: {loss_mask.sum().item()}")
    else:
        masked_returns = returns
        masked_values = values
        print(f"[DEBUG] No loss_mask, using all data")
    
    print(f"[DEBUG] masked_returns.shape: {masked_returns.shape}")
    if masked_returns.numel() > 0:
        print(
            f"[DEBUG] masked_returns stats: min={masked_returns.min().item():.6f}, "
            f"max={masked_returns.max().item():.6f}, mean={masked_returns.mean().item():.6f}"
        )
    else:
        print("[DEBUG] masked_returns is empty after loss_mask")
    print(f"[DEBUG] masked_values.shape: {masked_values.shape}")
    if masked_values.numel() > 0:
        print(
            f"[DEBUG] masked_values stats: min={masked_values.min().item():.6f}, "
            f"max={masked_values.max().item():.6f}, mean={masked_values.mean().item():.6f}"
        )
    else:
        print("[DEBUG] masked_values is empty after loss_mask")

    # Explicitly log NaN/Inf source to locate why explained_variance turns NaN.
    _debug_tensor_health("masked_returns", masked_returns)
    _debug_tensor_health("masked_values", masked_values)
    _debug_tensor_health("masked_returns_minus_values", masked_returns - masked_values)

    ev_returns = masked_returns
    ev_values = masked_values
    valid_mask = torch.isfinite(ev_returns) & torch.isfinite(ev_values)
    ev_returns = ev_returns[valid_mask]
    ev_values = ev_values[valid_mask]

    var_eps = 1e-8
    if ev_returns.numel() < 2:
        explained_variance = torch.tensor(0.0, device=returns.device)
        ev_valid = False
    else:
        var_returns = torch.var(ev_returns, unbiased=False)

        if torch.isfinite(var_returns) and var_returns > var_eps:
            var_diff = torch.var(ev_returns - ev_values, unbiased=False)

            if torch.isfinite(var_diff):
                explained_variance = 1 - var_diff / var_returns
                ev_valid = True
            else:
                explained_variance = torch.tensor(0.0, device=returns.device)
                ev_valid = False
        else:
            explained_variance = torch.tensor(0.0, device=returns.device)
            ev_valid = False

    explained_variance_for_log = torch.nan_to_num(
        explained_variance, nan=0.0, posinf=0.0, neginf=0.0
    )
    print(
        f"[DEBUG] Final explained_variance(raw): {explained_variance.item() if torch.isfinite(explained_variance).item() else 'NaN'}"
    )
    print(
        f"[DEBUG] Final explained_variance(logged): {explained_variance_for_log.item():.6f}, valid={ev_valid}"
    )
    print("[DEBUG PPO CRITIC LOSS] ====== EXITING ======" + "\n" + "="*80)

    # Compile metrics for logging
    metrics_data = {
        "critic/value_loss": value_loss.detach().item(),
        "critic/value_clip_ratio": value_clip_ratio.detach().item(),
        "critic/explained_variance": explained_variance_for_log.detach().item(),
        "critic/explained_variance_valid": float(ev_valid),
    }
    return value_loss, metrics_data


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
    log_ratio_clip: float = 20.0,  # 防止exp爆炸
    loss_agg_func: Optional[Callable[..., torch.Tensor]] = masked_mean,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Strict Sequence-level PPO (GSPO version)

    - ratio = exp(sum_t (logpi - logpi_old))
    - seq_adv = sum_t advantage
    - PPO clip on sequence ratio
    - KL = 0.5 * (log_ratio^2)
    """

    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs).bool()

    # 处理 broadcast
    if logprobs.ndim == 3 and loss_mask.shape[-1] == 1:
        loss_mask = loss_mask.expand_as(logprobs)

    reduce_dims = tuple(range(1, logprobs.ndim))

    # -------------------------------------------------
    # 1️⃣ 计算 sequence-level log-ratio (MEAN, per paper Eq.14: 1/|A| * sum)
    # -------------------------------------------------
    log_diff = torch.where(
        loss_mask,
        logprobs - old_logprobs,
        torch.zeros_like(logprobs),
    )

    token_count = loss_mask.sum(dim=reduce_dims, keepdim=True).clamp(min=1)
    seq_log_ratio = log_diff.sum(dim=reduce_dims, keepdim=True) / token_count

    # 数值稳定
    seq_log_ratio = torch.clamp(
        seq_log_ratio,
        -log_ratio_clip,
        log_ratio_clip,
    )

    ratio = torch.exp(seq_log_ratio)

    # -------------------------------------------------
    # 2️⃣ 计算 sequence-level advantage (MEAN, advantage 已是 sequence-level 常量)
    # -------------------------------------------------
    if advantages.ndim != logprobs.ndim:
        advantages = advantages.expand_as(logprobs)

    adv_masked = torch.where(
        loss_mask,
        advantages,
        torch.zeros_like(advantages),
    )

    seq_adv = adv_masked.sum(dim=reduce_dims, keepdim=True) / token_count

    # -------------------------------------------------
    # 3️⃣ PPO Clipping (sequence-level)
    # -------------------------------------------------
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - clip_ratio_low,
        1.0 + clip_ratio_high,
    )

    policy_loss1 = -seq_adv * ratio
    policy_loss2 = -seq_adv * clipped_ratio

    policy_loss = torch.max(policy_loss1, policy_loss2)

    # 有效序列 mask
    seq_mask = token_count > 0

    policy_loss_mean = loss_agg_func(policy_loss, seq_mask)

    # -------------------------------------------------
    # 4️⃣ KL penalty (稳定版)
    # -------------------------------------------------
    # approx KL ≈ 0.5 * (log_ratio)^2 (基于 sum，参考论文 Eq 18)
    seq_log_ratio_sum = log_diff.sum(dim=reduce_dims, keepdim=True)
    approx_kl = 0.5 * loss_agg_func(
        seq_log_ratio_sum.pow(2),
        seq_mask,
    )

    pure_policy_loss = policy_loss_mean.clone()
    policy_loss_mean = policy_loss_mean + kl_beta * approx_kl

    # -------------------------------------------------
    # 5️⃣ Metrics
    # -------------------------------------------------
    with torch.no_grad():

        clip_mask = (ratio > 1.0 + clip_ratio_high) | (
            ratio < 1.0 - clip_ratio_low
        )

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
