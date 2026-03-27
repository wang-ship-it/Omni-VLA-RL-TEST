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

import os
from typing import Any, Optional

import numpy as np
import torch

from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.config import SupportedModel
from rlinf.utils.chain_trace import (
    format_block,
    get_chain_trace_config,
    rowwise_abs_mean,
    select_topk_indices,
    should_enable_chain_trace,
    tensor_stats_str,
    trace_id_to_str,
)
from rlinf.utils.distributed import all_reduce_dict, masked_normalization
from rlinf.utils.metric_utils import append_to_dict, compute_rollout_metrics
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.utils import (
    clear_memory,
    cpu_weight_swap,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
)
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def flatten_rollout_batch_for_train(
    nested_dict: dict, shuffle_id: Optional[torch.Tensor]
) -> dict:
    """Flatten [T, B, ...] rollout tensors to [T*B, ...] for actor training."""
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            if isinstance(value, torch.Tensor):
                value = value[:-1]

        if "env_info" in key:
            raise NotImplementedError("env_info nested dict is not supported here")

        if value is None:
            ret_dict[key] = None
            continue

        if isinstance(value, torch.Tensor):
            flat = value.reshape(-1, *value.shape[2:])
            ret_dict[key] = flat[shuffle_id] if shuffle_id is not None else flat
        elif isinstance(value, dict):
            ret_dict[key] = flatten_rollout_batch_for_train(value, shuffle_id)
        else:
            raise NotImplementedError(
                f"Unsupported value type in rollout batch: key={key}, type={type(value)}"
            )

    return ret_dict


class AsyncPPOEmbodiedFSDPActor(EmbodiedFSDPActor):
    """Embodied FSDP actor worker for async PPO / decoupled actor-critic training."""

    def init_worker(self) -> None:
        super().init_worker()
        self.previous_actor_state_dict = retrieve_model_state_dict_in_cpu(
            self.model, {}
        )
        self.previous_actor_offload_buffer = {}

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)
        self.previous_actor_state_dict = retrieve_model_state_dict_in_cpu(
            self.model, getattr(self, "previous_actor_state_dict", {})
        )
        if not hasattr(self, "previous_actor_offload_buffer"):
            self.previous_actor_offload_buffer = {}

    def _unwrap_model_for_debug(self):
        model = self.model
        while hasattr(model, "module"):
            model = model.module
        return model

    def _set_omni_vla_gradient_checkpointing(self, enabled: bool) -> bool:
        model = self._unwrap_model_for_debug()
        toggle_method = getattr(model, "gradient_checkpointing_enable", None)
        disable_method = getattr(model, "gradient_checkpointing_disable", None)
        reasoning_spatial_expert = getattr(model, "reasoning_spatial_expert", None)
        if (
            reasoning_spatial_expert is None
            or toggle_method is None
            or disable_method is None
        ):
            return False
        if enabled:
            toggle_method()
        else:
            disable_method()
        return True

    def _log_encoder_freeze_status(self) -> None:
        if not (self._rank == 0 and int(self.version) < 3):
            return

        model = self._unwrap_model_for_debug()
        reasoning_spatial_expert = getattr(model, "reasoning_spatial_expert", None)
        if reasoning_spatial_expert is None:
            self.logger.info(
                "[Async PPO debug] encoder freeze status unavailable: "
                "missing reasoning_spatial_expert on actor model"
            )
            return

        modules = {
            "vggt_encoder": getattr(reasoning_spatial_expert, "vggt_encoder", None),
            "vision_tower": getattr(
                getattr(reasoning_spatial_expert, "reasoning_expert", None),
                "vision_tower",
                None,
            ),
        }

        status_parts = []
        for name, module in modules.items():
            if module is None:
                status_parts.append(f"{name}=missing")
                continue

            params = list(module.parameters())
            total_params = sum(p.numel() for p in params)
            trainable_params = sum(p.numel() for p in params if p.requires_grad)
            frozen = trainable_params == 0
            status_parts.append(
                f"{name}(training={module.training}, frozen={frozen}, "
                f"trainable_params={trainable_params}, total_params={total_params})"
            )

        self.logger.info(
            "[Async PPO debug] encoder freeze status: " + ", ".join(status_parts)
        )

    def _chain_trace_enabled(self) -> bool:
        return should_enable_chain_trace(self.cfg, self._rank)

    def _metric_value(self, metrics_data: dict[str, Any], key: str) -> float | None:
        value = metrics_data.get(key, None)
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            return float(value.detach().float().mean().item())
        return float(value)

    def _train_chain_trace_triggered(self, metrics_data: dict[str, Any]) -> bool:
        trace_cfg = get_chain_trace_config(self.cfg)
        if not trace_cfg["enabled"]:
            return False
        if trace_cfg["mode"] != "anomaly":
            return True
        thresholds = trace_cfg["thresholds"]
        behav_kl = self._metric_value(metrics_data, "actor/behav_approx_kl")
        prox_old_gap = self._metric_value(
            metrics_data, "actor/prox_old_logprob_gap_abs_mean"
        )
        train_rollout_gap = self._metric_value(
            metrics_data, "actor/debug_train_rollout_logprob_gap_abs_mean"
        )
        return bool(
            (behav_kl is not None and behav_kl >= thresholds["behav_approx_kl"])
            or (
                prox_old_gap is not None
                and prox_old_gap >= thresholds["prox_old_gap_abs_mean"]
            )
            or (
                train_rollout_gap is not None
                and train_rollout_gap >= thresholds["train_rollout_gap_abs_mean"]
            )
        )

    def _format_version_tuple(
        self,
        *,
        versions: torch.Tensor | None,
        batch_idx: int,
        current_version: int | None = None,
    ) -> str:
        batch_version = None
        if versions is not None:
            batch_version = int(versions[batch_idx].detach().reshape(-1)[0].item())
        return (
            f"current_version={current_version}, "
            f"previous_actor_version={int(self.version)}, "
            f"batch_version={batch_version}"
        )

    def _log_proximal_chain_trace(
        self,
        *,
        flat_batch: dict[str, Any],
        proximal_logprobs_flat: torch.Tensor,
        recompute_mode: str,
        omni_semantic_overrides: dict[str, bool],
        sample_trace_context: dict[str, Any] | None,
    ) -> None:
        if not self._chain_trace_enabled():
            return

        prev_logprobs = flat_batch["prev_logprobs"]
        gap = proximal_logprobs_flat.float() - prev_logprobs.float()
        sample_gap = rowwise_abs_mean(gap)
        trace_cfg = get_chain_trace_config(self.cfg)
        threshold = trace_cfg["thresholds"]["prox_old_gap_abs_mean"]
        if trace_cfg["mode"] == "anomaly" and float(sample_gap.max().item()) < threshold:
            return

        forward_inputs = flat_batch.get("forward_inputs", {})
        trace_ids = forward_inputs.get("debug_trace_ids")
        sample_indices = forward_inputs.get("debug_sample_indices")
        versions = flat_batch.get("versions")
        topk_indices = select_topk_indices(sample_gap, trace_cfg["topk"])

        summary_lines = [
            (
                f"step={int(self.version)} recompute_mode={recompute_mode} "
                f"prox_old_gap_abs_mean={sample_gap.mean().item():.6f} "
                f"prox_old_gap_abs_max={sample_gap.max().item():.6f} "
                f"threshold={threshold:.6f}"
            ),
            f"trace_ids={[trace_id_to_str(trace_ids[idx]) for idx in topk_indices] if trace_ids is not None else []}",
        ]
        self.logger.info(format_block("STEP SUMMARY [PROXIMAL]", summary_lines))
        self.logger.info(
            format_block(
                "DIFF SUMMARY [PROXIMAL]",
                [
                    f"prox_old_gap_abs_mean={sample_gap.mean().item():.6f}",
                    f"prox_old_gap_abs_max={sample_gap.max().item():.6f}",
                ],
            )
        )

        for idx in topk_indices:
            lines = [
                (
                    f"step={int(self.version)} trace_id={trace_id_to_str(trace_ids[idx]) if trace_ids is not None else 'none'} "
                    f"sample_idx={int(sample_indices[idx].item()) if sample_indices is not None else idx} "
                    f"{self._format_version_tuple(versions=versions, batch_idx=idx)}"
                ),
                (
                    "semantic_flags="
                    f"recompute_mode={recompute_mode}, "
                    f"prefix_middle_no_grad_override={omni_semantic_overrides.get('prefix_middle_no_grad_override')}, "
                    f"clone_past_key_values_override={omni_semantic_overrides.get('clone_past_key_values_override')}, "
                    f"gradient_checkpointing_override={omni_semantic_overrides.get('gradient_checkpointing_override')}"
                ),
            ]
            if sample_trace_context:
                lines.append(
                    "runtime_context="
                    f"training={sample_trace_context.get('model_training')}, "
                    f"gc={sample_trace_context.get('gradient_checkpointing_enabled')}, "
                    f"reasoning_use_cache={sample_trace_context.get('reasoning_use_cache')}, "
                    f"prefix_cache_seq_len={sample_trace_context.get('prefix_cache_seq_len')}, "
                    f"middle_cache_seq_len={sample_trace_context.get('middle_cache_seq_len')}"
                )
            lines.extend(
                [
                    f"old_logprobs={tensor_stats_str(prev_logprobs[idx])}",
                    f"proximal_logprobs={tensor_stats_str(proximal_logprobs_flat[idx])}",
                    f"abs(old-prox)={tensor_stats_str(gap[idx].abs())}",
                ]
            )
            self.logger.info(format_block("PROXIMAL RECOMPUTE TRACE", lines))

    def _log_train_chain_trace(
        self,
        *,
        metrics_data: dict[str, Any],
        data: dict[str, Any],
        out: dict[str, Any],
        current_version: int,
    ) -> None:
        if not self._chain_trace_enabled():
            return

        behav_kl = self._metric_value(metrics_data, "actor/behav_approx_kl")
        prox_old_gap_mean = self._metric_value(
            metrics_data, "actor/prox_old_logprob_gap_abs_mean"
        )
        train_rollout_gap_mean = self._metric_value(
            metrics_data, "actor/debug_train_rollout_logprob_gap_abs_mean"
        )
        logprobs = out["logprobs"].detach().float()
        old_logprobs = data["prev_logprobs"].detach().float()
        proximal_logprobs = data.get("proximal_logprobs", None)
        proximal_logprobs = (
            proximal_logprobs.detach().float() if proximal_logprobs is not None else None
        )
        sample_score = rowwise_abs_mean(logprobs - old_logprobs)
        if proximal_logprobs is not None:
            sample_score = sample_score + rowwise_abs_mean(proximal_logprobs - old_logprobs)

        trace_cfg = get_chain_trace_config(self.cfg)
        topk_indices = select_topk_indices(sample_score, trace_cfg["topk"])
        forward_inputs = data["forward_inputs"]
        trace_ids = forward_inputs.get("debug_trace_ids")
        sample_indices = forward_inputs.get("debug_sample_indices")
        versions = data.get("versions", None)
        rollout_gap = forward_inputs.get("debug_old_vs_local_recompute_gap_abs_mean")
        if rollout_gap is not None:
            rollout_gap = rollout_gap.detach().float().reshape(-1)

        summary_lines = [
            (
                f"step={int(self.version)} current_version={current_version} "
                f"behav_approx_kl={behav_kl} "
                f"prox_old_gap_abs_mean={prox_old_gap_mean} "
                f"train_rollout_gap_abs_mean={train_rollout_gap_mean}"
            ),
            f"trace_ids={[trace_id_to_str(trace_ids[idx]) for idx in topk_indices] if trace_ids is not None else []}",
        ]
        self.logger.info(format_block("STEP SUMMARY [TRAIN]", summary_lines))
        self.logger.info(
            format_block(
                "DIFF SUMMARY [TRAIN]",
                [
                    f"behav_approx_kl={behav_kl}",
                    f"prox_old_gap_abs_mean={prox_old_gap_mean}",
                    f"train_rollout_gap_abs_mean={train_rollout_gap_mean}",
                ],
            )
        )

        clip_fraction = self._metric_value(metrics_data, "actor/clip_fraction")
        proximal_ratio = self._metric_value(metrics_data, "actor/proximal_ratio")
        clipped_proximal_ratio = self._metric_value(
            metrics_data, "actor/clipped_proximal_ratio"
        )
        for idx in topk_indices:
            rollout_training = forward_inputs.get("debug_rollout_model_training")
            rollout_gc = forward_inputs.get("debug_rollout_gradient_checkpointing_enabled")
            rollout_use_cache = forward_inputs.get("debug_rollout_reasoning_use_cache")
            rollout_prefix_cache_len = forward_inputs.get("debug_rollout_prefix_cache_seq_len")
            rollout_middle_cache_len = forward_inputs.get("debug_rollout_middle_cache_seq_len")
            lines = [
                (
                    f"step={int(self.version)} trace_id={trace_id_to_str(trace_ids[idx]) if trace_ids is not None else 'none'} "
                    f"sample_idx={int(sample_indices[idx].item()) if sample_indices is not None else idx} "
                    f"{self._format_version_tuple(versions=versions, batch_idx=idx, current_version=current_version)}"
                ),
                (
                    "rollout_semantic_flags="
                    f"training={int(rollout_training[idx].item()) if rollout_training is not None else 'na'}, "
                    f"gc={int(rollout_gc[idx].item()) if rollout_gc is not None else 'na'}, "
                    f"reasoning_use_cache={int(rollout_use_cache[idx].item()) if rollout_use_cache is not None else 'na'}, "
                    f"prefix_cache_seq_len={int(rollout_prefix_cache_len[idx].item()) if rollout_prefix_cache_len is not None else 'na'}, "
                    f"middle_cache_seq_len={int(rollout_middle_cache_len[idx].item()) if rollout_middle_cache_len is not None else 'na'}"
                ),
                f"old_logprobs={tensor_stats_str(old_logprobs[idx])}",
                (
                    f"proximal_logprobs={tensor_stats_str(proximal_logprobs[idx])}"
                    if proximal_logprobs is not None
                    else "proximal_logprobs=None"
                ),
                f"current_logprobs={tensor_stats_str(logprobs[idx])}",
                f"advantages={tensor_stats_str(data['advantages'][idx])}",
                (
                    f"behav_weight={tensor_stats_str(torch.exp((proximal_logprobs[idx] - old_logprobs[idx]).detach()))}"
                    if proximal_logprobs is not None
                    else "behav_weight=None"
                ),
                (
                    f"rollout_old_vs_local_gap_abs_mean={float(rollout_gap[idx].item()):.6f}"
                    if rollout_gap is not None
                    else "rollout_old_vs_local_gap_abs_mean=None"
                ),
                (
                    f"clip_fraction={clip_fraction}, "
                    f"proximal_ratio={proximal_ratio}, "
                    f"clipped_proximal_ratio={clipped_proximal_ratio}"
                ),
            ]
            self.logger.info(format_block("TRAIN CONSUME TRACE", lines))

    def _refresh_previous_actor_snapshot(self) -> None:
        if not hasattr(self, "previous_actor_state_dict"):
            self.previous_actor_state_dict = {}
        self.previous_actor_state_dict = retrieve_model_state_dict_in_cpu(
            self.model, self.previous_actor_state_dict
        )
        if not hasattr(self, "previous_actor_offload_buffer"):
            self.previous_actor_offload_buffer = {}

    @torch.inference_mode()
    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        proximal_values = self.rollout_batch.get("proximal_values", None)
        prev_values = self.rollout_batch.get("prev_values", None)

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": proximal_values if proximal_values is not None else prev_values,
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
        }

        adv_and_ret = calculate_adv_and_returns(**kwargs)
        self.rollout_batch.update(adv_and_ret)

        if kwargs["loss_mask"] is not None:
            self.rollout_batch["loss_mask"] = kwargs["loss_mask"]
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch["loss_mask_sum"] = kwargs["loss_mask_sum"]

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics

    @torch.inference_mode()
    def compute_proximal_logprobs(self) -> None:
        assert not self.is_weight_offloaded, (
            "Weight offloading is not supported when recomputing proximal logprobs."
        )
        assert hasattr(self, "previous_actor_state_dict"), (
            "previous_actor_state_dict must be initialized before recomputing "
            "proximal logprobs."
        )

        t_dim = self.rollout_batch["prev_logprobs"].shape[0]
        b_dim = self.rollout_batch["prev_logprobs"].shape[1]

        flat = flatten_rollout_batch_for_train(self.rollout_batch, shuffle_id=None)
        total = flat["prev_logprobs"].shape[0]
        micro_batch_size = self.cfg.actor.micro_batch_size
        num_splits = (total + micro_batch_size - 1) // micro_batch_size

        iterator = split_dict_to_chunk(flat, num_splits)

        prev_training_mode = self.model.training
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        recompute_mode = str(
            self.cfg.algorithm.get("proximal_recompute_model_mode", "train")
        ).lower()
        omni_semantic_overrides: dict[str, bool] = {}
        restore_omni_gradient_checkpointing = False
        if recompute_mode == "train":
            self.model.train()
        elif recompute_mode == "eval":
            if model_type == SupportedModel.OMNI_VLA:
                # Keep the global module in train mode so activation-memory paths remain valid,
                # while explicitly overriding the Omni-VLA behavior semantics to mimic rollout.
                self.model.train()
                restore_omni_gradient_checkpointing = (
                    self._set_omni_vla_gradient_checkpointing(False)
                )
                omni_semantic_overrides = {
                    "prefix_middle_no_grad_override": False,
                    "clone_past_key_values_override": True,
                    "gradient_checkpointing_override": False,
                }
            else:
                self.model.eval()
        else:
            raise ValueError(
                "algorithm.proximal_recompute_model_mode must be one of "
                "{'train', 'eval'}"
            )
        proximal_logprobs_list = []
        prox_trace_context = None

        try:
            with cpu_weight_swap(
                self.model,
                self.previous_actor_state_dict,
                self.previous_actor_offload_buffer,
            ):
                with torch.no_grad():
                    for micro_batch in iterator:
                        micro_batch = put_tensor_device(micro_batch, self.device)
                        forward_inputs = micro_batch.get("forward_inputs", None)
                        if forward_inputs is None:
                            raise ValueError(
                                "Missing forward_inputs in compute_proximal_logprobs. "
                                "This usually means batch splitting dropped nested dict fields."
                            )

                        model_kwargs = {}
                        if SupportedModel(self.cfg.actor.model.model_type) in [
                            SupportedModel.OPENVLA,
                            SupportedModel.OPENVLA_OFT,
                        ]:
                            model_kwargs["temperature"] = (
                                self.cfg.algorithm.sampling_params.temperature_train
                            )
                            model_kwargs["top_k"] = (
                                self.cfg.algorithm.sampling_params.top_k
                            )
                        elif (
                            SupportedModel(self.cfg.actor.model.model_type)
                            == SupportedModel.GR00T
                        ):
                            model_kwargs["prev_logprobs"] = micro_batch["prev_logprobs"]

                        out = self.model(
                            forward_inputs=forward_inputs,
                            compute_logprobs=True,
                            compute_entropy=False,
                            compute_values=False,
                            use_cache=False,
                            debug_chain_trace=self._chain_trace_enabled(),
                            **omni_semantic_overrides,
                            **model_kwargs,
                        )
                        proximal_logprobs_list.append(out["logprobs"].cpu())
                        if prox_trace_context is None:
                            prox_trace_context = out.get("debug_trace_context", None)
        finally:
            if restore_omni_gradient_checkpointing:
                self._set_omni_vla_gradient_checkpointing(True)
            self.model.train(prev_training_mode)

        proximal_logprobs = torch.cat(proximal_logprobs_list, dim=0).view(
            t_dim,
            b_dim,
            *self.rollout_batch["prev_logprobs"].shape[2:],
        )
        self.rollout_batch["proximal_logprobs"] = proximal_logprobs
        self._log_proximal_chain_trace(
            flat_batch=flat,
            proximal_logprobs_flat=torch.cat(proximal_logprobs_list, dim=0),
            recompute_mode=recompute_mode,
            omni_semantic_overrides=omni_semantic_overrides,
            sample_trace_context=prox_trace_context,
        )

        if self._rank == 0 and int(self.version) < 3:
            prox_stats = proximal_logprobs.float()
            prev_stats = self.rollout_batch["prev_logprobs"].float()
            self.logger.info(
                "[Async PPO debug] proximal_logprobs ready: "
                f"prev_shape={tuple(self.rollout_batch['prev_logprobs'].shape)}, "
                f"prox_shape={tuple(proximal_logprobs.shape)}, "
                f"recompute_mode={recompute_mode}, "
                f"loss_type={self.cfg.algorithm.loss_type}, "
                f"prox_mean={prox_stats.mean().item():.6f}, "
                f"prox_std={prox_stats.std(unbiased=False).item():.6f}, "
                f"prev_mean={prev_stats.mean().item():.6f}, "
                f"prev_std={prev_stats.std(unbiased=False).item():.6f}, "
                f"prox_prev_gap={torch.mean(torch.abs(prox_stats - prev_stats)).item():.6f}"
            )

    def run_training(self) -> dict[str, Any]:
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self._log_encoder_freeze_status()

        t_dim = int(self.rollout_batch["prev_logprobs"].shape[0])
        b_dim = int(self.rollout_batch["prev_logprobs"].shape[1])
        total_samples = t_dim * b_dim

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.cfg.actor.seed) + int(self._rank))
        shuffle_id = torch.randperm(total_samples, generator=generator)

        with torch.no_grad():
            self.rollout_batch = flatten_rollout_batch_for_train(
                self.rollout_batch, shuffle_id
            )

        if self.cfg.algorithm.normalize_advantages:
            self.rollout_batch["advantages"] = masked_normalization(
                self.rollout_batch["advantages"],
                self.rollout_batch.get("loss_mask", None),
            )

        self.model.train()

        world_size = int(self._world_size)
        global_batch_size = int(self.cfg.actor.global_batch_size)
        micro_batch_size = int(self.cfg.actor.micro_batch_size)

        assert global_batch_size % (micro_batch_size * world_size) == 0, (
            f"global_batch_size {global_batch_size} must be divisible by "
            f"micro_batch_size {micro_batch_size} * world_size {world_size}"
        )

        per_rank_batch_size = global_batch_size // world_size
        micro_per_rank = per_rank_batch_size // micro_batch_size
        self.gradient_accumulation = micro_per_rank

        flattened_rollout_size = int(self.rollout_batch["prev_logprobs"].shape[0])
        assert flattened_rollout_size % per_rank_batch_size == 0, (
            f"Flattened rollout size {flattened_rollout_size} must be divisible by "
            f"per-rank batch size {per_rank_batch_size}"
        )
        num_global_batches = flattened_rollout_size // per_rank_batch_size
        receive_stats = self.get_rollout_receive_stats()
        tracked_receive_stats = {
            "replay_channel_qsize_before_recv": receive_stats[
                "replay_channel_qsize_before_recv"
            ],
            "replay_channel_qsize_after_drain": receive_stats[
                "replay_channel_qsize_after_drain"
            ],
            "dropped_rollout_batches": receive_stats["dropped_rollout_batches"],
            "current_version_minus_batch_version_max": float(
            int(self.version) + 1 - receive_stats["received_rollout_version_max"]
            ),
        }

        metrics: dict[str, list] = {}
        update_epoch = int(self.cfg.algorithm.get("update_epoch", 1))
        chain_trace_logged = False

        for _ in range(update_epoch):
            global_batch_iter = split_dict_to_chunk(
                self.rollout_batch,
                num_global_batches,
            )

            for train_global_batch in global_batch_iter:
                train_global_batch_size = int(
                    train_global_batch["prev_logprobs"].shape[0]
                )
                assert train_global_batch_size == per_rank_batch_size, (
                    f"Expected per-rank global batch size {per_rank_batch_size}, "
                    f"got {train_global_batch_size}"
                )
                assert train_global_batch_size % micro_batch_size == 0

                micro_batch_iter = split_dict_to_chunk(
                    train_global_batch,
                    micro_per_rank,
                )

                self.optimizer.zero_grad()

                for mb_idx, data in enumerate(micro_batch_iter):
                    data = put_tensor_device(
                        data,
                        f"cuda:{int(os.environ['LOCAL_RANK'])}",
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(mb_idx + 1) == self.gradient_accumulation,
                    )

                    advantages = data["advantages"]
                    old_logprobs = data["prev_logprobs"]
                    returns = data.get("returns", None)
                    prev_values = data.get("prev_values", None)
                    loss_mask = data.get("loss_mask", None)
                    loss_mask_sum = data.get("loss_mask_sum", None)

                    versions = data.get("versions", None)
                    proximal_logprobs = data.get("proximal_logprobs", None)
                    proximal_values = data.get("proximal_values", None)
                    current_version = int(self.version) + 1

                    forward_inputs = data.get("forward_inputs", None)
                    if forward_inputs is None:
                        raise ValueError(
                            "Missing forward_inputs in run_training. "
                            "This usually means batch splitting dropped nested dict fields."
                        )

                    model_kwargs = {}
                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.OPENVLA,
                        SupportedModel.OPENVLA_OFT,
                    ]:
                        model_kwargs["temperature"] = (
                            self.cfg.algorithm.sampling_params.temperature_train
                        )
                        model_kwargs["top_k"] = self.cfg.algorithm.sampling_params.top_k
                    elif (
                        SupportedModel(self.cfg.actor.model.model_type)
                        == SupportedModel.GR00T
                    ):
                        model_kwargs["prev_logprobs"] = old_logprobs

                    compute_values = self.cfg.algorithm.adv_type == "gae"

                    with self.amp_context:
                        out = self.model(
                            forward_inputs=forward_inputs,
                            compute_logprobs=True,
                            compute_entropy=(self.cfg.algorithm.entropy_bonus > 0),
                            compute_values=compute_values,
                            use_cache=False,
                            debug_train_rollout_semantics_gap=self.cfg.algorithm.get(
                                "debug_train_rollout_semantics_gap", False
                            ),
                            **model_kwargs,
                        )

                    if (
                        SupportedModel(self.cfg.actor.model.model_type)
                        == SupportedModel.GR00T
                    ):
                        old_logprobs = out["prev_logprobs"]

                    loss_kwargs = {
                        "loss_type": self.cfg.algorithm.loss_type,
                        "logprob_type": self.cfg.algorithm.logprob_type,
                        "reward_type": self.cfg.algorithm.reward_type,
                        "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                        "logprobs": out["logprobs"],
                        "values": out.get("values", None),
                        "old_logprobs": old_logprobs,
                        "advantages": advantages,
                        "returns": returns,
                        "prev_values": proximal_values
                        if proximal_values is not None
                        else prev_values,
                        "proximal_logprobs": proximal_logprobs,
                        "versions": versions,
                        "current_version": current_version,
                        "behave_weight_threshold": self.cfg.algorithm.get(
                            "behave_weight_threshold", None
                        ),
                        "clip_ratio_c": self.cfg.algorithm.get("clip_ratio_c", 3.0),
                        "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                        "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                        "value_clip": self.cfg.algorithm.get("value_clip", None),
                        "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                        "loss_mask": loss_mask,
                        "loss_mask_sum": loss_mask_sum,
                        "max_episode_steps": self.cfg.env.train.max_episode_steps,
                        "task_type": self.cfg.runner.task_type,
                        "critic_warmup": self.optimizer_steps
                        < self.critic_warmup_steps,
                    }

                    loss, metrics_data = policy_loss(**loss_kwargs)
                    debug_metrics = out.get("debug_metrics", None)
                    if debug_metrics:
                        metrics_data.update(debug_metrics)
                    if (
                        not chain_trace_logged
                        and self._train_chain_trace_triggered(metrics_data)
                    ):
                        self._log_train_chain_trace(
                            metrics_data=metrics_data,
                            data=data,
                            out=out,
                            current_version=current_version,
                        )
                        chain_trace_logged = True

                    if (
                        self._rank == 0
                        and int(self.version) < 3
                        and mb_idx == 0
                        and self.cfg.algorithm.loss_type == "decoupled_actor_critic"
                    ):
                        prox_gap = None
                        if proximal_logprobs is not None:
                            prox_gap = torch.mean(
                                torch.abs(out["logprobs"].float() - proximal_logprobs.float())
                            ).item()
                        old_gap = torch.mean(
                            torch.abs(out["logprobs"].float() - old_logprobs.float())
                        ).item()
                        adv_stats = advantages.float()
                        loss_mask_count = (
                            int(loss_mask.count_nonzero().item())
                            if loss_mask is not None
                            else int(out["logprobs"].numel())
                        )
                        debug_keys = [
                            "actor/proximal_ratio",
                            "actor/proximal_approx_kl",
                            "actor/behav_approx_kl",
                            "actor/behav_weight_p95",
                            "actor/clip_fraction",
                            "actor/debug_logprob_prox_gap_mean",
                            "actor/debug_prox_old_gap_mean",
                            "actor/debug_versions_min",
                            "actor/debug_versions_max",
                        ]
                        debug_metrics = {
                            key: float(metrics_data[key])
                            for key in debug_keys
                            if key in metrics_data
                        }
                        prox_gap_msg = (
                            f"{prox_gap:.6f}" if prox_gap is not None else "None"
                        )
                        self.logger.info(
                            "[Async PPO debug] decoupled actor metrics: "
                            f"{debug_metrics}, "
                            f"pre_loss_logprob_old_gap={old_gap:.6f}, "
                            f"pre_loss_logprob_prox_gap={prox_gap_msg}"
                        )
                        self.logger.info(
                            "[Async PPO debug] decoupled actor inputs: "
                            f"advantages_mean={adv_stats.mean().item():.6f}, "
                            f"advantages_std={adv_stats.std(unbiased=False).item():.6f}, "
                            f"loss_mask_count={loss_mask_count}"
                        )

                    entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                    if (
                        self.cfg.algorithm.entropy_bonus > 0
                        and not loss_kwargs["critic_warmup"]
                    ):
                        entropy = out["entropy"]
                        entropy = reshape_entropy(
                            entropy,
                            entropy_type=self.cfg.algorithm.entropy_type,
                            action_dim=self.cfg.actor.model.get("action_dim", 7),
                            batch_size=out["logprobs"].shape[0],
                        )
                        entropy_loss = masked_mean(entropy, mask=loss_mask)
                        loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

                    loss = loss / self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    metrics_data["actor/entropy_loss"] = float(
                        entropy_loss.detach().item()
                    )
                    metrics_data["actor/total_loss"] = float(loss.detach().item())
                    for key, value in tracked_receive_stats.items():
                        metrics_data[f"actor/{key}"] = float(value)
                    append_to_dict(metrics, metrics_data)

                torch.cuda.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                extra_metrics = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    extra_metrics["critic/lr"] = lr_list[1]
                append_to_dict(metrics, extra_metrics)

        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        self._refresh_previous_actor_snapshot()
        clear_memory()

        mean_metric_dict = {k: float(np.mean(v)) for k, v in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict,
            op=torch.distributed.ReduceOp.AVG,
        )
        return mean_metric_dict
