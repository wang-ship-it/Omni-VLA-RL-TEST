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

from typing import Optional

import torch


def huber_loss(error: torch.Tensor, delta: float) -> torch.Tensor:
    return torch.where(
        error.abs() < delta, 0.5 * error**2, delta * (error.abs() - 0.5 * delta)
    )


def kl_penalty(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    """
    Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def preprocess_embodied_advantages_inputs(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    """
    Preprocess inputs before computing advantages & returns.
    Unify names & formats, align with math interfaces.
    """
    print("\n" + "="*80)
    print("[DEBUG EMBODIED PREPROCESS] ====== ENTERING preprocess_embodied_advantages_inputs ======")
    print(f"[DEBUG EMBODIED] reward_type: {kwargs.get('reward_type', 'N/A')}")
    print(f"[DEBUG EMBODIED] rewards.shape (INPUT): {rewards.shape}")
    print(f"[DEBUG EMBODIED] rewards stats: min={rewards.min().item():.6f}, max={rewards.max().item():.6f}, mean={rewards.mean().item():.6f}")
    print(f"[DEBUG EMBODIED] dones.shape (INPUT): {dones.shape}")
    if values is not None:
        print(f"[DEBUG EMBODIED] values.shape (INPUT): {values.shape}")
        print(f"[DEBUG EMBODIED] values stats: min={values.min().item():.6f}, max={values.max().item():.6f}, mean={values.mean().item():.6f}")
    else:
        print("[DEBUG EMBODIED] values is None")
    if loss_mask is not None:
        print(f"[DEBUG EMBODIED] loss_mask.shape (INPUT): {loss_mask.shape}, dtype: {loss_mask.dtype}")
    if loss_mask_sum is not None:
        print(f"[DEBUG EMBODIED] loss_mask_sum.shape (INPUT): {loss_mask_sum.shape}, dtype: {loss_mask_sum.dtype}")
        
    if kwargs["reward_type"] == "chunk_level":
        print("[DEBUG EMBODIED] reward_type == chunk_level, applying sum/max preprocessing...")
        rewards = rewards.sum(dim=-1, keepdim=True)
        print(f"[DEBUG EMBODIED] rewards.shape AFTER sum: {rewards.shape}")
        dones = dones.max(dim=-1, keepdim=True)[0]
        print(f"[DEBUG EMBODIED] dones.shape AFTER max: {dones.shape}")
        if loss_mask is not None:
            loss_mask = loss_mask.max(dim=-1, keepdim=True)[0]
            print(f"[DEBUG EMBODIED] loss_mask.shape AFTER max: {loss_mask.shape}")
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.max(dim=-1, keepdim=True)[0]
            print(f"[DEBUG EMBODIED] loss_mask_sum.shape AFTER max: {loss_mask_sum.shape}")

    num_chunk, bsz, chunk_size = rewards.shape
    n_steps = num_chunk * chunk_size
    print(f"[DEBUG EMBODIED] num_chunk: {num_chunk}, bsz: {bsz}, chunk_size: {chunk_size}")
    print(f"[DEBUG EMBODIED] n_steps (total steps): {n_steps}")
    kwargs.update(
        {
            "num_chunk": num_chunk,
            "batch_size": bsz,
            "chunk_size": chunk_size,
            "n_steps": n_steps,
        }
    )

    # Transpose(1, 2) -> [num-chunk, chunk-size, bsz]
    # Reshape -> [n_steps, bsz]
    # Rewards [n_steps, bsz]
    rewards = rewards.transpose(1, 2).reshape(n_steps, bsz)
    print(f"[DEBUG EMBODIED] rewards.shape AFTER transpose+reshape: {rewards.shape}")
    print(f"[DEBUG EMBODIED] rewards stats: min={rewards.min().item():.6f}, max={rewards.max().item():.6f}, mean={rewards.mean().item():.6f}")

    # Loss Mask (T steps) [bsz, n_steps]
    if loss_mask is not None:
        loss_mask = loss_mask.transpose(1, 2).reshape(n_steps, bsz)
        print(f"[DEBUG EMBODIED] loss_mask.shape AFTER transpose+reshape: {loss_mask.shape}")

    # Dones (T+1 steps) [num-chunk+1, bsz, chunk-size]
    flattened_dones_full = dones.transpose(1, 2).reshape(
        (num_chunk + 1) * chunk_size, bsz
    )
    dones = flattened_dones_full[-(n_steps + 1) :]
    print(f"[DEBUG EMBODIED] dones.shape AFTER flatten: {dones.shape}")
    print(f"[DEBUG EMBODIED] dones[-1] sum (episode ends): {dones[-1].sum().item()}")

    if kwargs["adv_type"] == "gae":
        flattened_values_full = values.transpose(1, 2).reshape(
            (num_chunk + 1) * chunk_size, bsz
        )
        values = flattened_values_full[: n_steps + 1]
        print(f"[DEBUG EMBODIED] values.shape AFTER flatten: {values.shape}")
        print(f"[DEBUG EMBODIED] values stats: min={values.min().item():.6f}, max={values.max().item():.6f}, mean={values.mean().item():.6f}")

    kwargs.update(
        {
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
        }
    )
    print("[DEBUG EMBODIED] ====== EXITING ======" + "\n" + "="*80 + "\n")
    return kwargs


def calculate_scores(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    **kwargs,
) -> dict:
    scores = torch.zeros(kwargs["batch_size"])
    for step in reversed(range(kwargs["n_steps"])):
        scores = scores * ~dones[step + 1]
        scores += rewards[step]
    scores = scores.reshape(-1, kwargs["group_size"])

    kwargs.update(
        {
            "rewards": scores,
            "dones": dones,
        }
    )

    return kwargs


def postprocess_embodied_advantages_outputs(
    advantages: torch.Tensor,
    num_chunk: int,
    chunk_size: int,
    returns: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    """
    Post-process results for Embodiment tasks; unflatten tensors.
    """
    print("\n" + "="*80)
    print("[DEBUG POSTPROCESS ADV] ====== ENTERING postprocess_embodied_advantages_outputs ======")
    print(f"[DEBUG POSTPROCESS ADV] num_chunk: {num_chunk}, chunk_size: {chunk_size}")
    print(f"[DEBUG POSTPROCESS ADV] advantages.shape (input): {advantages.shape}")
    if returns is not None:
        print(f"[DEBUG POSTPROCESS ADV] returns.shape (input): {returns.shape}")
    
    res = {}

    advantages = advantages.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
    print(f"[DEBUG POSTPROCESS ADV] advantages.shape (after reshape+transpose): {advantages.shape}")
    print(f"[DEBUG POSTPROCESS ADV] advantages stats: min={advantages.min().item():.6f}, max={advantages.max().item():.6f}, mean={advantages.mean().item():.6f}")
    res.update({"advantages": advantages})

    if returns is not None:
        returns = returns.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
        print(f"[DEBUG POSTPROCESS ADV] returns.shape (after reshape+transpose): {returns.shape}")
        print(f"[DEBUG POSTPROCESS ADV] returns stats: min={returns.min().item():.6f}, max={returns.max().item():.6f}, mean={returns.mean().item():.6f}")
        res.update({"returns": returns})

    print(f"[DEBUG POSTPROCESS ADV] Final res keys: {list(res.keys())}")
    print("[DEBUG POSTPROCESS ADV] ====== EXITING ======" + "\n" + "="*80 + "\n")
    return res


def preprocess_reasoning_advantages_inputs(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    logprob: Optional[torch.Tensor] = None,
    ref_logprob: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    # NOTE: to align with embodied inputs, we transpose loss mask and rewards when needed.

    bsz, seq_len = loss_mask.shape
    loss_mask = loss_mask.transpose(0, 1)  # [seq_len, bsz]

    # Track actual batch size (may be expanded for grouped algorithms)
    actual_bsz = bsz

    assert rewards.ndim == 1, f"Unsupported reward shape {rewards.shape}"

    if kwargs["adv_type"] == "gae":
        expanded_rewards = torch.zeros(
            (seq_len, bsz), dtype=rewards.dtype, device=rewards.device
        )
        expanded_rewards[-1] = rewards  # only last token has reward
        kwargs.update({"rewards": expanded_rewards})

    elif kwargs["adv_type"] == "grpo":
        grouped_rewards = rewards.reshape(-1, kwargs["group_size"]).contiguous()
        kwargs.update(
            {
                "rewards": grouped_rewards,
            }
        )

    elif kwargs["adv_type"] == "gspo":
        grouped_rewards = rewards.reshape(-1, kwargs["group_size"]).contiguous()
        kwargs.update(
            {
                "rewards": grouped_rewards,
            }
        )

    elif kwargs["adv_type"] == "grpo_dynamic":
        grouped_rewards = (
            rewards.reshape(-1, kwargs["num_sequence"]).transpose(0, 1).contiguous()
        )
        kwargs.update(
            {
                "rewards": grouped_rewards,
            }
        )

    elif kwargs["adv_type"] == "reinpp":
        kwargs.update({"rewards": rewards.unsqueeze(0)})

    if values is not None:  # [bsz, seq_len]
        assert values.ndim == 2, f"Unsupported values shape {values.shape}"
        values = values.transpose(0, 1)  # [seq_len, bsz]
        # pad values with zeros at the end for bootstrapping
        values = torch.cat(
            [
                values,
                torch.zeros(
                    (1, values.shape[-1]), dtype=values.dtype, device=values.device
                ),
            ],
            dim=0,
        )  # [seq_len+1, bsz]

        kwargs.update({"values": values})

    if logprob is not None:
        logprob = logprob.transpose(0, 1)
        kwargs.update({"logprob": logprob})

    if ref_logprob is not None:
        ref_logprob = ref_logprob.transpose(0, 1)
        kwargs.update({"ref_logprob": ref_logprob})

    # Create done flags (episode ends at the last token)
    # Use actual_bsz which may be expanded for grouped algorithms like gspo
    dones = torch.zeros(seq_len + 1, actual_bsz, dtype=torch.bool, device=rewards.device)
    dones[-1] = True
    kwargs.update(
        {
            "dones": dones,
            "loss_mask": loss_mask,
        }
    )

    return kwargs


def postprocess_reasoning_advantages_outputs(
    advantages: torch.Tensor,
    returns: Optional[torch.Tensor] = None,
) -> dict:
    """
    Post-process results for Reasoning tasks; transpose tensors back.
    """

    advantages = advantages.transpose(0, 1)  # [bsz, seq_len]
    if returns is not None:
        returns = returns.transpose(0, 1)  # [bsz, seq_len]

    return advantages, returns


def preprocess_loss_inputs(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    logprob_type: Optional[str] = None,
    single_action_dim: Optional[int] = None,
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    values: Optional[torch.Tensor] = None,
    prev_values: Optional[torch.Tensor] = None,
    returns: Optional[torch.Tensor] = None,
    reward_type: Optional[str] = None,
    **kwargs,
) -> dict:
    print("\n" + "="*80)
    print("[DEBUG PREPROCESS] ====== ENTERING preprocess_loss_inputs ======")
    print(f"[DEBUG PREPROCESS] reward_type: {reward_type}, logprob_type: {logprob_type}")
    print(f"[DEBUG PREPROCESS] logprobs.shape: {logprobs.shape}, dtype: {logprobs.dtype}")
    print(f"[DEBUG PREPROCESS] old_logprobs.shape: {old_logprobs.shape}")
    print(f"[DEBUG PREPROCESS] advantages.shape BEFORE flatten: {advantages.shape}")
    if loss_mask is not None:
        print(f"[DEBUG PREPROCESS] loss_mask.shape BEFORE flatten: {loss_mask.shape}, dtype: {loss_mask.dtype}")
    if values is not None:
        print(f"[DEBUG PREPROCESS] values.shape BEFORE flatten: {values.shape}")
    if prev_values is not None:
        print(f"[DEBUG PREPROCESS] prev_values.shape BEFORE flatten: {prev_values.shape}")
    if returns is not None:
        print(f"[DEBUG PREPROCESS] returns.shape BEFORE flatten: {returns.shape}")
    print(f"[DEBUG PREPROCESS] single_action_dim: {single_action_dim}")
    
    if reward_type == "chunk_level":
        print("[DEBUG PREPROCESS] reward_type == chunk_level, flattening tensors...")
        advantages = advantages.flatten()
        print(f"[DEBUG PREPROCESS] advantages.shape AFTER flatten: {advantages.shape}")
        if loss_mask is not None:
            loss_mask = loss_mask.flatten()
            print(f"[DEBUG PREPROCESS] loss_mask.shape AFTER flatten: {loss_mask.shape}")
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.flatten()
        if values is not None:
            values = values.flatten()
            print(f"[DEBUG PREPROCESS] values.shape AFTER flatten: {values.shape}")
        if prev_values is not None:
            prev_values = prev_values.flatten()
            print(f"[DEBUG PREPROCESS] prev_values.shape AFTER flatten: {prev_values.shape}")
        if returns is not None:
            returns = returns.flatten()
            print(f"[DEBUG PREPROCESS] returns.shape AFTER flatten: {returns.shape}")

    bsz = logprobs.shape[0]
    print(f"[DEBUG PREPROCESS] bsz (batch size): {bsz}")
    
    if logprob_type == "token_level":
        print("[DEBUG PREPROCESS] logprob_type == token_level")
        logprobs = logprobs.reshape(bsz, -1, single_action_dim)
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim)
        advantages = advantages.unsqueeze(-1)
        if loss_mask is not None:
            loss_mask = loss_mask.unsqueeze(-1)
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.unsqueeze(-1)

    elif logprob_type == "action_level":
        print("[DEBUG PREPROCESS] logprob_type == action_level")
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)

    elif logprob_type == "chunk_level":
        print("[DEBUG PREPROCESS] logprob_type == chunk_level")
        logprobs_reshaped = logprobs.reshape(bsz, -1, single_action_dim)
        num_elements = logprobs_reshaped.shape[1] * logprobs_reshaped.shape[2]
        print(f"[DEBUG PREPROCESS] num_elements (action_chunks * action_dim): {num_elements}")
        logprobs = logprobs_reshaped.sum(dim=[1, 2]) / num_elements
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2]) / num_elements
        print(f"[DEBUG PREPROCESS] logprobs.shape AFTER chunk_level: {logprobs.shape}")
        print(f"[DEBUG PREPROCESS] logprobs stats: min={logprobs.min().item():.6f}, max={logprobs.max().item():.6f}, mean={logprobs.mean().item():.6f}")
        print(f"[DEBUG PREPROCESS] old_logprobs stats: min={old_logprobs.min().item():.6f}, max={old_logprobs.max().item():.6f}, mean={old_logprobs.mean().item():.6f}")

    target_shape = logprobs.shape
    print(f"[DEBUG PREPROCESS] target_shape for expansion: {target_shape}")
    advantages = expand_to_target_dim(advantages, target_shape)
    loss_mask = expand_to_target_dim(loss_mask, target_shape)
    loss_mask_sum = expand_to_target_dim(loss_mask_sum, target_shape)
    values = expand_to_target_dim(values, target_shape)
    prev_values = expand_to_target_dim(prev_values, target_shape)
    returns = expand_to_target_dim(returns, target_shape)
    
    print(f"[DEBUG PREPROCESS] Final shapes after expansion:")
    print(f"[DEBUG PREPROCESS]   logprobs: {logprobs.shape}")
    print(f"[DEBUG PREPROCESS]   old_logprobs: {old_logprobs.shape}")
    print(f"[DEBUG PREPROCESS]   advantages: {advantages.shape}")
    print(f"[DEBUG PREPROCESS]   loss_mask: {loss_mask.shape if loss_mask is not None else None}")
    print(f"[DEBUG PREPROCESS]   values: {values.shape if values is not None else None}")
    print(f"[DEBUG PREPROCESS]   prev_values: {prev_values.shape if prev_values is not None else None}")
    print(f"[DEBUG PREPROCESS]   returns: {returns.shape if returns is not None else None}")
    print("[DEBUG PREPROCESS] ====== EXITING ======" + "\n" + "="*80 + "\n")

    kwargs.update(
        {
            "logprobs": logprobs,
            "old_logprobs": old_logprobs,
            "advantages": advantages,
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "values": values,
            "prev_values": prev_values,
            "returns": returns,
        }
    )

    return kwargs


def postprocess_loss_metric(metrics_data: dict) -> dict:
    for k, v in metrics_data.items():
        if isinstance(v, torch.Tensor):
            metrics_data[k] = v.detach().item()
        elif isinstance(v, (float, int)):
            metrics_data[k] = v
    return metrics_data


def expand_to_target_dim(tensor, target_shape):
    if tensor is None:
        return None
    if tensor.shape != target_shape:
        while len(tensor.shape) < len(target_shape):
            tensor = tensor.unsqueeze(-1)
    return tensor


def safe_normalize(array, loss_mask):
    valid_array = array[loss_mask]
    if len(valid_array) > 0:
        mean = valid_array.mean()
        std = valid_array.std()
        array = (array - mean) / (std + 1e-5)

    return array
