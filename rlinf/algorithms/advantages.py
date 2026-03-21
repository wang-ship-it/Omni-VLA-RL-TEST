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
from typing import Optional

import torch

from rlinf.algorithms.registry import register_advantage
from rlinf.algorithms.utils import kl_penalty, safe_normalize
from rlinf.utils.utils import masked_mean


_DEBUG_VERBOSE = (
    os.getenv("OMNI_VLA_DEBUG_LOG", "0") == "1"
    or os.getenv("OMNI_VLA_DEBUG_ADV", "0") == "1"
)
_DEBUG_EVERY = max(1, int(os.getenv("OMNI_VLA_DEBUG_EVERY", "1")))
_DEBUG_COUNTER = 0
_DEBUG_INCLUDE = [
    s.strip() for s in os.getenv("OMNI_VLA_DEBUG_INCLUDE", "").split(",") if s.strip()
]


def print(*args, **kwargs):
    global _DEBUG_COUNTER
    if not _DEBUG_VERBOSE:
        return
    if _DEBUG_INCLUDE:
        msg = " ".join(str(a) for a in args)
        if not any(k in msg for k in _DEBUG_INCLUDE):
            return
    _DEBUG_COUNTER += 1
    if _DEBUG_COUNTER % _DEBUG_EVERY == 0:
        builtins.print(*args, **kwargs)


@register_advantage("gae")
def compute_gae_advantages_and_returns(
    rewards: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    values: Optional[torch.Tensor] = None,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate advantages and returns for Proximal Policy Optimization (PPO).
    NOTE: currently this function does not support auto-reset.

    This function implements Generalized Advantage Estimation (GAE) to compute
    advantages and returns for PPO training. The advantages are normalized
    using mean and standard deviation for stable training.

    Args:
        rewards (torch.Tensor): Rewards per timestep. Shape: [seq_len, bsz].
        values (torch.Tensor): Value function estimates. Shape: [seq_len, bsz].
        dones (torch.Tensor): Done flags (1 if episode ended, else 0).
        gamma (float, optional): Discount factor. Defaults to 1.0.
        gae_lambda (float, optional): GAE smoothing factor. Defaults to 1.0.
        normalize_advantages (bool, optional): Whether to normalize advantages. Defaults to True.
        normalize_returns (bool, optional): Whether to normalize returns. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns)
    """
    print("\n" + "="*80)
    print("[DEBUG GAE] ====== ENTERING compute_gae_advantages_and_returns ======")
    print(f"[DEBUG GAE] rewards.shape: {rewards.shape}, dtype: {rewards.dtype}")
    print(f"[DEBUG GAE] rewards stats: min={rewards.min().item():.6f}, max={rewards.max().item():.6f}, mean={rewards.mean().item():.6f}")
    print(f"[DEBUG GAE] rewards has_nan: {torch.isnan(rewards).any().item()}, has_inf: {torch.isinf(rewards).any().item()}")
    print(f"[DEBUG GAE] gamma: {gamma}, gae_lambda: {gae_lambda}")
    if values is not None:
        print(f"[DEBUG GAE] values.shape: {values.shape}")
        print(f"[DEBUG GAE] values stats: min={values.min().item():.6f}, max={values.max().item():.6f}, mean={values.mean().item():.6f}")
        print(f"[DEBUG GAE] values has_nan: {torch.isnan(values).any().item()}, has_inf: {torch.isinf(values).any().item()}")
    else:
        print("[DEBUG GAE] values is None (critic-free mode)")
    if dones is not None:
        print(f"[DEBUG GAE] dones.shape: {dones.shape}")
        print(f"[DEBUG GAE] dones stats: sum={dones.sum().item()}, mean={dones.float().mean().item():.6f}")
    else:
        print("[DEBUG GAE] dones is None")
    if loss_mask is not None:
        print(f"[DEBUG GAE] loss_mask.shape: {loss_mask.shape}")
        print(f"[DEBUG GAE] loss_mask true count: {loss_mask.sum().item()}")
    else:
        print("[DEBUG GAE] loss_mask is None")
        
    T = rewards.shape[0]
    print(f"[DEBUG GAE] T (sequence length): {T}")
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0

    critic_free = values is None
    if critic_free:
        gae_lambda = 1
        gamma = 1
        print("[DEBUG GAE] Running in CRITIC-FREE mode (values is None)")
    else:
        print("[DEBUG GAE] Running in CRITIC mode with value function")

    print(f"[DEBUG GAE] Starting GAE backward computation from step {T-1} to 0")
    for step in reversed(range(T)):
        if critic_free:
            delta = rewards[step]
        else:
            delta = (
                rewards[step]
                + gamma * values[step + 1] * (~dones[step + 1])
                - values[step]
            )
        
        gae = delta + gamma * gae_lambda * (~dones[step + 1]) * gae
        returns[step] = gae if critic_free else gae + values[step]
        
        if step == T-1 or step == 0:
            delta_str = f"{delta.mean().item():.6f}" if delta.numel() > 1 else f"{delta.item():.6f}"
            gae_str = f"{gae.mean().item():.6f}" if gae.numel() > 1 else f"{gae.item():.6f}"
            returns_step_str = f"{returns[step].mean().item():.6f}" if returns[step].numel() > 1 else f"{returns[step].item():.6f}"
            print(f"[DEBUG GAE] Step {step}: delta={delta_str}, gae={gae_str}, returns[{step}]={returns_step_str}")

    advantages = returns - values[:-1] if not critic_free else returns
    
    print(f"[DEBUG GAE] advantages.shape: {advantages.shape}")
    print(f"[DEBUG GAE] advantages stats BEFORE normalization: min={advantages.min().item():.6f}, max={advantages.max().item():.6f}, mean={advantages.mean().item():.6f}")
    print(f"[DEBUG GAE] returns.shape: {returns.shape}")
    print(f"[DEBUG GAE] returns stats: min={returns.min().item():.6f}, max={returns.max().item():.6f}, mean={returns.mean().item():.6f}")

    if normalize_advantages:
        print(f"[DEBUG GAE] Normalizing advantages...")
        advantages = safe_normalize(advantages, loss_mask=loss_mask)
        print(f"[DEBUG GAE] advantages stats AFTER normalization: min={advantages.min().item():.6f}, max={advantages.max().item():.6f}, mean={advantages.mean().item():.6f}")
    if normalize_returns:
        print(f"[DEBUG GAE] Normalizing returns...")
        returns = safe_normalize(returns, loss_mask=loss_mask)
    
    print(f"[DEBUG GAE] Final advantages stats: min={advantages.min().item():.6f}, max={advantages.max().item():.6f}, mean={advantages.mean().item():.6f}")
    print(f"[DEBUG GAE] Final returns stats: min={returns.min().item():.6f}, max={returns.max().item():.6f}, mean={returns.mean().item():.6f}")
    print("[DEBUG GAE] ====== EXITING ======" + "\n" + "="*80 + "\n")

    return advantages, returns


@register_advantage("grpo")
def compute_grpo_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    **kwargs,
):
    """
    Compute GRPO advantages.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        group_size (int): Group size for advantage computation.

    Returns:
        torch.Tensor: advantages
    """
    grouped_rewards = rewards.view(-1, group_size)

    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )

    advantages = grouped_rewards - grouped_reward_mean
    advantages = advantages / (grouped_reward_std + 1e-6)

    advantages = (torch.zeros_like(loss_mask) + advantages.view(1, -1)) * loss_mask

    return advantages, None


@register_advantage("grpo_dynamic")
def compute_grpo_dynamic_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    idx_to_traj: list[int],
    advantage_mode: str = "turn",
    **kwargs,
):
    """
    Compute GRPO advantages for multi-turn multi-agent scenarios.

    IMPORTANT: This function computes advantages PER QUESTION, not globally.
    - idx_to_traj maps turn_idx -> global_traj_idx (e.g., [0,0,1,1,2,2,3,3,4,4,...,15,15])
    - Trajectories 0-3 belong to question 0, 4-7 to question 1, etc.
    - We must compute GRPO separately for each question's group_size trajectories

    One advantage computation modes:

    1. "turn": Turn-level GRPO
       - Compute mean/std over all turns within each question
       - Example: Q0 has 4 trajs with 1,2,3,4 turns = 10 turns total.
                  Compute GRPO over these 10 turn rewards (currently all same within traj).
       - Future-proof: works when turns have different rewards within same trajectory

    Args:
        rewards: Shape [num_sequence, 1] after preprocessing (num_sequence = total turns)
        loss_mask: Shape [seq_len, num_sequence] after preprocessing
        group_size: Number of trajectories per question (e.g., 4)
        idx_to_traj: List mapping turn_idx -> global_traj_idx
        advantage_mode: "turn"

    Returns:
        advantages: Shape [seq_len, num_sequence]
    """
    num_sequence = len(idx_to_traj)

    # Handle rewards shape - squeeze if needed
    if rewards.ndim == 2:
        rewards_flat = rewards.squeeze(-1)  # [num_sequence, 1] -> [num_sequence]
    else:
        rewards_flat = rewards  # Already [num_sequence]

    assert rewards_flat.numel() == num_sequence, (
        f"Rewards size mismatch: {rewards_flat.numel()} != {num_sequence}"
    )

    # Determine number of questions
    num_trajectories = max(idx_to_traj) + 1
    num_questions = num_trajectories // group_size
    assert num_trajectories % group_size == 0, (
        f"num_trajectories {num_trajectories} not divisible by group_size {group_size}"
    )

    # Initialize advantage tensor
    turn_advantages = torch.zeros(
        num_sequence, dtype=rewards.dtype, device=rewards.device
    )
    if advantage_mode == "turn":
        # For each question, compute GRPO over all its turns

        # Step 1: Map each turn to its question
        turn_to_question = torch.tensor(
            [idx_to_traj[i] // group_size for i in range(num_sequence)],
            dtype=torch.long,
            device=rewards.device,
        )

        # Step 2: Compute per-question statistics over all turns
        for question_idx in range(num_questions):
            # Get all turns belonging to this question
            question_mask = turn_to_question == question_idx
            question_turn_rewards = rewards_flat[question_mask]

            # Compute statistics for this question's turns
            question_mean = question_turn_rewards.mean()
            question_std = question_turn_rewards.std()

            # Normalize turns in this question
            normalized_question_rewards = (question_turn_rewards - question_mean) / (
                question_std + 1e-6
            )

            # Assign back to turn_advantages
            turn_advantages[question_mask] = normalized_question_rewards

    else:
        raise ValueError(f"Invalid advantage_mode: {advantage_mode}. Must be 'turn'")

    # Broadcast advantages to match loss_mask shape [seq_len, num_sequence]
    # turn_advantages is [num_sequence], we broadcast to [seq_len, num_sequence]
    advantages = torch.zeros_like(
        loss_mask, dtype=rewards.dtype
    ) + turn_advantages.view(1, -1)
    advantages = advantages * loss_mask

    return advantages, None


@register_advantage("gspo")
def compute_gspo_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages for GSPO (Group-level Sequence Policy Optimization).
    Uses pairwise ranking: A_i = mean(r_i - r_j) for all j != i in the group.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups * group_size] or [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [seq_len, num_sequences]
        group_size (int): Number of sequences per group.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (advantages, None)
    """
    # Ensure rewards is 1D before reshaping
    if rewards.ndim == 2:
        rewards = rewards.reshape(-1)

    grouped_rewards = rewards.view(-1, group_size)  # [num_groups, group_size]

    # Eq. 16: normalize by mean and std over all G trajectories in the group
    mean = grouped_rewards.mean(dim=-1, keepdim=True)  # [num_groups, 1]
    std  = grouped_rewards.std(dim=-1, keepdim=True)   # [num_groups, 1]

    advantages = (grouped_rewards - mean) / (std + 1e-6)  # [num_groups, group_size]
    advantages = advantages.view(-1)  # Flatten to [num_groups * group_size]

    # Broadcast sequence-level advantage to all tokens in each sequence
    # loss_mask shape: [seq_len, num_sequences]
    # advantages shape: [num_sequences]
    seq_len = loss_mask.shape[0]
    advantages = (torch.zeros(seq_len, advantages.shape[0], dtype=loss_mask.dtype, device=loss_mask.device) + advantages.unsqueeze(0)) * loss_mask

    return advantages, None


@register_advantage("reinpp")
def compute_reinpp_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    use_reinpp_baseline: bool = False,
    kl_beta: float = 0.0,
    logprob=None,
    ref_logprob=None,
    kl_penalty_type: str = "",
    **kwargs,
):
    """
    Compute advantages for reinforce++ and reinforce++ baseline.

    Args:
        rewards (torch.Tensor): The reward or score values.
        loss_mask (torch.Tensor): The loss mask for valid entries.
        group_size (int): The group size for advantage computation.
        use_reinpp_baseline (bool, optional): Whether to use reinforce++ baseline.
        kl_beta (float, optional): KL penalty coefficient.
        logprob (optional): Log probability of current policy.
        ref_logprob (optional): Log probability of reference policy.
        kl_penalty_type (str, optional): Type of KL penalty.

    Returns:
        torch.Tensor: advantages
    """
    # first group baseline for reinforce++ baseline
    if use_reinpp_baseline:
        grouped_rewards = rewards.view(-1, group_size)  # [num_prompt, group_size]
        grouped_rewards -= grouped_rewards.mean(dim=1, keepdims=True)
        rewards = grouped_rewards.view(-1)  # [B]

    # build the reward matrix
    r_matrix = torch.zeros_like(loss_mask).float()  # [L, B]
    seq_length = loss_mask.size(0)
    mask_flipped = loss_mask.long().fliplr()
    eos_positions = mask_flipped.argmax(
        dim=0, keepdim=True
    )  # position of last True in original mask
    eos_indices = seq_length - 1 - eos_positions  # [1, B]

    r_matrix = r_matrix.scatter_(dim=0, index=eos_indices, src=rewards)  # [L, B]

    # add kl penalty
    if kl_beta > 0:
        kld = kl_penalty(logprob, ref_logprob, kl_penalty=kl_penalty_type)  # [L, B]
        r_matrix -= kl_beta * kld

    # compute return
    ret_matrix = torch.cumsum(r_matrix.flip(dims=[0]), dim=0).flip(dims=[0])

    # normalize
    advantages = ret_matrix.clone()

    mean = masked_mean(advantages, loss_mask)
    var = masked_mean((advantages - mean).pow(2), loss_mask)
    rstd = var.clamp(min=1e-8).rsqrt()

    advantages = (advantages - mean) * rstd

    return advantages, None
