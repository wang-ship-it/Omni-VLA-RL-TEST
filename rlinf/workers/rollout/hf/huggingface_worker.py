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

import copy
import gc
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.data.embodied_io_struct import (
    RolloutResult,
)
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.chain_trace import (
    format_block,
    get_chain_trace_config,
    get_pipeline_trace_config,
    rowwise_abs_mean,
    select_topk_indices,
    should_enable_chain_trace,
    should_enable_pipeline_trace,
    summarize_value,
    tensor_stats_str,
    trace_id_to_str,
)
from rlinf.utils.placement import HybridComponentPlacement


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        self.actor_group_name = cfg.actor.group_name
        self.device = self.torch_platform.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        actor_world_size = self.placement.get_world_size("actor")
        self.actor_weight_src_rank = self._rank % actor_world_size
        self.rollout_epoch = cfg.algorithm.get("rollout_epoch", 1)
        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.expert_model = None

        # Sync weight comm options
        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )
        self.total_num_train_envs = cfg.env.train.total_num_envs
        self.total_num_eval_envs = cfg.env.eval.total_num_envs
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num

        self.train_batch_size = (
            self.total_num_train_envs // self._world_size // self.num_pipeline_stages
        )
        self.eval_batch_size = (
            self.total_num_eval_envs // self._world_size // self.num_pipeline_stages
        )
        self.enable_cuda_graph = cfg.rollout.get("enable_cuda_graph", False)
        self.enable_eval = cfg.runner.val_check_interval > 0 or cfg.runner.only_eval

        self.n_train_chunk_steps = (
            cfg.env.train.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )
        self.n_eval_chunk_steps = (
            cfg.env.eval.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.version = 0
        self.applied_weight_version = 0
        self.requested_weight_version = 0
        self.finished_episodes = None
        self._debug_trace_batch_counter = 0
        self._pipeline_trace_enabled = should_enable_pipeline_trace(cfg, self._rank)
        self._pipeline_trace_cfg = get_pipeline_trace_config(cfg)
        self._pipeline_trace_counter = 0

    def _should_log_pipeline_trace(self) -> bool:
        if not self._pipeline_trace_enabled:
            return False
        self._pipeline_trace_counter += 1
        return (
            self._pipeline_trace_counter % self._pipeline_trace_cfg["log_every"] == 0
        )

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self.hf_model: BasePolicy = get_model(rollout_model_config)

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)

        if self.cfg.rollout.get("expert_model", None):
            expert_model_config = copy.deepcopy(self.cfg.actor.model)
            with open_dict(expert_model_config):
                expert_model_config.precision = self.cfg.rollout.expert_model.precision
                expert_model_config.model_path = (
                    self.cfg.rollout.expert_model.model_path
                )
            self.expert_model = get_model(expert_model_config)

            if self.cfg.runner.get("expert_ckpt_path", None):
                expert_model_dict = torch.load(self.cfg.runner.expert_ckpt_path)
                self.expert_model.load_state_dict(expert_model_dict)

        self.hf_model.eval()
        if self.expert_model is not None:
            self.expert_model.eval()

        if self.cfg.rollout.get("enable_torch_compile", False):
            mode = self.cfg.rollout.get(
                "torch_compile_mode", "max-autotune-no-cudagraphs"
            )
            self.hf_model.enable_torch_compile(mode=mode)
        if self.enable_cuda_graph and not self.enable_offload:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
            )

        self.dst_ranks = {
            "train": self._setup_dst_ranks(
                self.total_num_train_envs // self.num_pipeline_stages
            ),
        }
        self.src_ranks = {
            "train": self._setup_src_ranks(
                self.total_num_train_envs // self.num_pipeline_stages
            ),
        }
        if self.enable_eval:
            self.dst_ranks["eval"] = self._setup_dst_ranks(
                self.total_num_eval_envs // self.num_pipeline_stages
            )
            self.src_ranks["eval"] = self._setup_src_ranks(
                self.total_num_eval_envs // self.num_pipeline_stages
            )

        self.log_info(f"Rollout worker initialized with dst_ranks: {self.dst_ranks}")
        self.log_info(f"Rollout worker initialized with src_ranks: {self.src_ranks}")
        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_train"]
            if self._sampling_params["do_sample"]
            else 1.0,
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

        self._eval_sampling_params = {
            "do_sample": True
            if self._sampling_params.get("temperature_eval", -1) > 0
            else False,
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

        if self.expert_model is not None:
            self._dagger_sampling_params = {
                "beta": self.cfg.algorithm.get("dagger", {}).get("init_beta", 0.5),
                "beta_schedule": self.cfg.algorithm.get("dagger", {}).get(
                    "beta_schedule", "exponential"
                ),
                "beta_min": self.cfg.algorithm.get("dagger", {}).get("beta_min", 0.05),
                "beta_decay": self.cfg.algorithm.get("dagger", {}).get(
                    "beta_decay", 0.99
                ),
            }

    def update_dagger_beta(self):
        if self.expert_model is None:
            return

        if self._dagger_sampling_params["beta_schedule"] == "exponential":
            self._dagger_sampling_params["beta"] = max(
                self._dagger_sampling_params["beta_min"],
                self._dagger_sampling_params["beta"]
                * self._dagger_sampling_params["beta_decay"],
            )
        else:
            raise NotImplementedError(
                f"Beta schedule {self._dagger_sampling_params['beta_schedule']} is not implemented"
            )

    def _setup_dst_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """Compute env peer ranks for this rollout worker.

        This mapping supports both one-to-many and many-to-one env/rollout layouts.
        The returned ranks are used as communication counterparts for receiving env
        outputs and sending action chunks.

        Args:
            batch_size: Total env batch size per pipeline stage across all workers.

        Returns:
            Ordered ``(env_rank, batch_size)`` tuples this rollout worker should
            send action chunks to.
        """
        env_world_size = self.placement.get_world_size("env")
        rollout_world_size = self.placement.get_world_size("rollout")
        return CommMapper.get_dst_ranks(
            batch_size=batch_size,
            src_world_size=rollout_world_size,
            dst_world_size=env_world_size,
            src_rank=self._rank,
        )

    def _setup_src_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """Compute env source ranks and sizes for receiving env outputs."""
        env_world_size = self.placement.get_world_size("env")
        rollout_world_size = self.placement.get_world_size("rollout")
        return CommMapper.get_src_ranks(
            batch_size=batch_size,
            src_world_size=env_world_size,
            dst_world_size=rollout_world_size,
            dst_rank=self._rank,
        )

    @Worker.timer("predict")
    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "train"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.OMNI_VLA,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.CNN_POLICY,
        ]:
            if self.cfg.algorithm.loss_type == "embodied_dagger":
                kwargs = {"mode": "eval"}
            else:
                kwargs = {"mode": mode}

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.CNN_POLICY,
            SupportedModel.FLOW_POLICY,
            SupportedModel.MLP_POLICY,
        ]:
            kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        only_save_expert = self.cfg.algorithm.get("dagger", {}).get(
            "only_save_expert", True
        )

        if mode == "train" and self.expert_model is not None:
            # training with expert model. Beta-probability acting.
            use_expert = torch.rand(1).item() < self._dagger_sampling_params["beta"]
        else:
            use_expert = False

        with torch.no_grad():
            expert_label_flag = False
            # Decide which model to act via use_expert
            if use_expert:
                actions, result = self.expert_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )
                expert_label_flag = True
            else:
                actions, result = self.hf_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )

            # Decide re-label or not
            if (
                not only_save_expert  # only re-label in classic dagger mode
                and not use_expert  # only re-label if not using expert
                and self.expert_model is not None  # only re-label if expert exists
                and mode == "train"  # only re-label in train mode
            ):
                _, expert_result = self.expert_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )
                expert_forward_inputs = expert_result["forward_inputs"]
                expert_target = expert_forward_inputs.get(
                    "model_action", expert_forward_inputs.get("action")
                )
                if expert_target is not None:
                    result["forward_inputs"]["model_action"] = expert_target
                expert_label_flag = True

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        result["expert_label_flag"] = bool(expert_label_flag)
        if (
            mode == "train"
            and SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.OMNI_VLA
        ):
            self._attach_rollout_chain_trace(result)
        return actions, result

    def _attach_rollout_chain_trace(self, result: dict[str, Any]) -> None:
        forward_inputs = result.get("forward_inputs", {})
        prev_logprobs = result.get("prev_logprobs", None)
        if not forward_inputs or prev_logprobs is None:
            return

        batch_size = int(prev_logprobs.shape[0])
        trace_base = (
            int(self.applied_weight_version) * 1_000_000_000
            + int(self._rank) * 1_000_000
            + int(self._debug_trace_batch_counter) * 1_000
        )
        self._debug_trace_batch_counter += 1

        trace_ids = torch.arange(
            trace_base,
            trace_base + batch_size,
            dtype=torch.int64,
            device=prev_logprobs.device,
        )[:, None]
        sample_indices = torch.arange(
            batch_size, dtype=torch.int64, device=prev_logprobs.device
        )[:, None]

        forward_inputs["debug_trace_ids"] = trace_ids
        forward_inputs["debug_sample_indices"] = sample_indices
        forward_inputs["debug_rollout_step"] = torch.full(
            (batch_size, 1),
            int(self.version),
            dtype=torch.int64,
            device=prev_logprobs.device,
        )
        forward_inputs["debug_rollout_version"] = torch.full(
            (batch_size, 1),
            int(self.applied_weight_version),
            dtype=torch.int64,
            device=prev_logprobs.device,
        )
        forward_inputs["debug_rollout_rank"] = torch.full(
            (batch_size, 1),
            int(self._rank),
            dtype=torch.int64,
            device=prev_logprobs.device,
        )

        trace_enabled = should_enable_chain_trace(self.cfg, self._rank)
        if not trace_enabled:
            return

        recompute_out = self.hf_model(
            forward_inputs=forward_inputs,
            compute_logprobs=True,
            compute_entropy=False,
            compute_values=False,
            use_cache=False,
            debug_chain_trace=True,
            prefix_middle_no_grad_override=False,
            clone_past_key_values_override=True,
            gradient_checkpointing_override=False,
        )
        local_logprobs = recompute_out["logprobs"].detach()
        gap = prev_logprobs.detach().float() - local_logprobs.float()
        sample_gap_abs_mean = rowwise_abs_mean(gap)
        sample_gap_mean = gap.reshape(gap.shape[0], -1).mean(dim=1)

        forward_inputs["debug_old_vs_local_recompute_gap_abs_mean"] = (
            sample_gap_abs_mean[:, None]
        )
        forward_inputs["debug_old_vs_local_recompute_gap_mean"] = sample_gap_mean[:, None]

        trace_context = recompute_out.get("debug_trace_context", {})
        for key in (
            "model_training",
            "gradient_checkpointing_enabled",
            "reasoning_use_cache",
            "spatial_use_cache",
            "action_use_cache",
            "prefix_middle_no_grad",
            "clone_past_key_values",
            "prefix_cache_seq_len",
            "middle_cache_seq_len",
        ):
            value = trace_context.get(key, 0)
            dtype = torch.int64 if isinstance(value, bool) or isinstance(value, int) else torch.float32
            forward_inputs[f"debug_rollout_{key}"] = torch.full(
                (batch_size, 1),
                int(value) if dtype == torch.int64 else float(value),
                dtype=dtype,
                device=prev_logprobs.device,
            )

        trace_cfg = get_chain_trace_config(self.cfg)
        threshold = trace_cfg["thresholds"]["train_rollout_gap_abs_mean"]
        should_log = (
            trace_cfg["mode"] != "anomaly"
            or float(sample_gap_abs_mean.max().item()) >= threshold
        )
        if not should_log:
            return

        topk_indices = select_topk_indices(sample_gap_abs_mean, trace_cfg["topk"])
        summary_lines = [
            (
                f"step={int(self.version)} rollout_version={int(self.applied_weight_version)} "
                f"worker_rank={int(self._rank)} batch_size={batch_size} "
                f"old_vs_local_gap_abs_mean={sample_gap_abs_mean.mean().item():.6f} "
                f"old_vs_local_gap_abs_max={sample_gap_abs_mean.max().item():.6f} "
                f"threshold={threshold:.6f}"
            ),
            f"trace_ids={[trace_id_to_str(trace_ids[idx]) for idx in topk_indices]}",
        ]
        self.logger.info(format_block("STEP SUMMARY [ROLLOUT]", summary_lines))
        self.logger.info(
            format_block(
                "DIFF SUMMARY [ROLLOUT]",
                [
                    f"old_vs_local_gap_abs_mean={sample_gap_abs_mean.mean().item():.6f}",
                    f"old_vs_local_gap_abs_max={sample_gap_abs_mean.max().item():.6f}",
                ],
            )
        )

        chains = forward_inputs.get("chains")
        denoise_inds = forward_inputs.get("denoise_inds")
        for idx in topk_indices:
            denoise = denoise_inds[idx].detach().cpu()
            denoise_pos = int(denoise.reshape(-1)[0].item())
            chains_pre = chains[idx, denoise_pos]
            chains_next = chains[idx, denoise_pos + 1]
            lines = [
                (
                    f"step={int(self.version)} trace_id={trace_id_to_str(trace_ids[idx])} "
                    f"sample_idx={int(sample_indices[idx].item())} "
                    f"rollout_version={int(self.applied_weight_version)}"
                ),
                (
                    "semantic_flags="
                    f"training={bool(trace_context.get('model_training', False))}, "
                    f"gc={bool(trace_context.get('gradient_checkpointing_enabled', False))}, "
                    f"reasoning_use_cache={bool(trace_context.get('reasoning_use_cache', False))}, "
                    f"prefix_cache_seq_len={int(trace_context.get('prefix_cache_seq_len', 0))}, "
                    f"middle_cache_seq_len={int(trace_context.get('middle_cache_seq_len', 0))}"
                ),
                f"denoise_inds={tensor_stats_str(denoise_inds[idx])}",
                f"chains={tensor_stats_str(chains[idx])}",
                f"chains_pre={tensor_stats_str(chains_pre)}",
                f"chains_next={tensor_stats_str(chains_next)}",
                f"prev_logprobs={tensor_stats_str(prev_logprobs[idx])}",
                f"local_recompute_logprobs={tensor_stats_str(local_logprobs[idx])}",
                (
                    f"old_vs_local_gap={tensor_stats_str(gap[idx])} "
                    f"old_vs_local_gap_abs_mean={float(sample_gap_abs_mean[idx].item()):.6f}"
                ),
            ]
            self.logger.info(format_block("ROLLOUT TRACE", lines))

    def get_bootstrap_values(
        self, final_obs: dict[str, Any] | None
    ) -> torch.Tensor | None:
        if final_obs is None:
            return None
        if not (
            hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head")
        ):
            return None
        with torch.no_grad():
            actions, result = self.predict(final_obs)
            if "prev_values" in result and result["prev_values"] is not None:
                final_values = result["prev_values"]
            else:
                final_values = torch.zeros_like(actions[:, :1], dtype=torch.float32)
        return final_values[:, :1].cpu().contiguous()

    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""
        param_state_dict = await self.recv(
            self.actor_group_name,
            src_rank=self.actor_weight_src_rank,
            async_op=True,
            options=self._sync_weight_comm_options,
        ).async_wait()
        self.hf_model.load_state_dict(param_state_dict)
        # In the current runner order, rollout.set_global_step(step) is called
        # immediately before syncing weights for that same logical step. Record the
        # applied version only after load_state_dict has actually completed.
        self.applied_weight_version = int(self.version)
        self.requested_weight_version = float(self.version)

        del param_state_dict
        gc.collect()
        self.torch_platform.empty_cache()

    def get_weight_sync_metrics(self) -> dict[str, float]:
        return {
            "applied_weight_version": float(self.applied_weight_version),
            "applied_minus_runner_version": float(
                self.applied_weight_version - self.version
            ),
        }

    @Worker.timer("generate_one_epoch")
    async def generate_one_epoch(self, input_channel: Channel, output_channel: Channel):
        self.update_dagger_beta()
        for _ in range(self.n_train_chunk_steps):
            for _ in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)
                actions, result = self.predict(env_output["obs"])
                if self._should_log_pipeline_trace():
                    bootstrap_values = self.get_bootstrap_values(
                        env_output.get("final_obs", None)
                    )
                    self.logger.info(
                        format_block(
                            "PIPELINE TRACE [ROLLOUT]",
                            [
                                f"rank={self._rank} version={int(self.version)} applied_weight_version={int(self.applied_weight_version)}",
                                f"env_obs={summarize_value(env_output.get('obs'), max_items=self._pipeline_trace_cfg['max_items'])}",
                                f"actions={summarize_value(actions, max_items=self._pipeline_trace_cfg['max_items'])}",
                                f"prev_logprobs={summarize_value(result.get('prev_logprobs'), max_items=self._pipeline_trace_cfg['max_items'])}",
                                f"prev_values={summarize_value(result.get('prev_values'), max_items=self._pipeline_trace_cfg['max_items'])}",
                                f"bootstrap_values={summarize_value(bootstrap_values, max_items=self._pipeline_trace_cfg['max_items'])}",
                            ],
                        )
                    )

                save_flags = None
                if result.get("expert_label_flag", False):
                    save_flags = torch.full(
                        (actions.shape[0], self.cfg.actor.model.num_action_chunks),
                        True,
                        dtype=torch.bool,
                        device=actions.device,
                    )
                rollout_result = RolloutResult(
                    actions=actions,
                    prev_logprobs=result["prev_logprobs"]
                    if self.collect_prev_infos
                    else None,
                    prev_values=result["prev_values"]
                    if self.collect_prev_infos
                    else None,
                    bootstrap_values=self.get_bootstrap_values(
                        env_output.get("final_obs", None)
                    ),
                    save_flags=save_flags,
                    forward_inputs=result["forward_inputs"],
                    versions=torch.full_like(
                        result["prev_logprobs"],
                        float(self.applied_weight_version),
                        dtype=torch.float32,
                    ),
                )
                self.send_rollout_result(output_channel, rollout_result, mode="train")
        for _ in range(self.num_pipeline_stages):
            env_output = await self.recv_env_output(input_channel)
            actions, result = self.predict(env_output["obs"])

            rollout_result = RolloutResult(
                actions=actions,
                prev_values=result["prev_values"] if self.collect_prev_infos else None,
                bootstrap_values=self.get_bootstrap_values(
                    env_output.get("final_obs", None)
                ),
            )
            self.send_rollout_result(output_channel, rollout_result, mode="train")

    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
    ):
        if self.enable_offload:
            self.reload_model()

        for _ in tqdm(
            range(self.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            await self.generate_one_epoch(input_channel, output_channel)

        if self.enable_offload:
            self.offload_model()

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()
        for _ in tqdm(
            range(self.cfg.algorithm.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(self.n_eval_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel, mode="eval")
                    actions, _ = self.predict(env_output["obs"], mode="eval")
                    self.send_chunk_actions(output_channel, actions, mode="eval")

        if self.enable_offload:
            self.offload_model()

    def offload_model(self):
        if self.enable_cuda_graph:
            self.hf_model.release_cuda_graph()
        self.hf_model.to("cpu")
        self.torch_platform.empty_cache()

    def reload_model(self):
        self.hf_model.to(self.device)
        if self.enable_cuda_graph:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
            )

    async def recv_env_output(
        self, input_channel: Channel, mode: Literal["train", "eval"] = "train"
    ) -> dict[str, Any]:
        """Receive env outputs from mapped env ranks and merge if needed.

        Args:
            input_channel: Channel carrying env->rollout outputs.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.

        Returns:
            A single env output dict. When multiple env ranks are mapped to this
            rollout worker, outputs are merged on batch dimension.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_ranks[mode]
        obs_batches = []
        for src_rank, expected_size in src_ranks_and_sizes:
            obs_batch = await input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_obs"
                ),
                async_op=True,
            ).async_wait()
            actual_size = self._infer_env_batch_size(obs_batch)
            assert actual_size == expected_size, (
                f"Expected env output batch size {expected_size} from env rank {src_rank}, "
                f"got {actual_size}."
            )
            obs_batches.append(obs_batch)
        merged = self._merge_obs_batches(obs_batches)
        if self._pipeline_trace_enabled and mode == "train":
            self.logger.info(
                format_block(
                    "PIPELINE TRACE [ROLLOUT RECV ENV]",
                    [
                        f"rank={self._rank} mode={mode} version={int(self.version)} src={src_ranks_and_sizes}",
                        f"merged_obs={summarize_value(merged.get('obs'), max_items=self._pipeline_trace_cfg['max_items'])}",
                        f"merged_final_obs={summarize_value(merged.get('final_obs'), max_items=self._pipeline_trace_cfg['max_items'])}",
                    ],
                )
            )
        return merged

    def _split_actions(
        self, actions: torch.Tensor | np.ndarray, sizes: list[int]
    ) -> list[torch.Tensor | np.ndarray]:
        """Split rollout actions into size-specified shards along dim-0.

        Args:
            actions: Model-predicted action chunk batch (tensor or ndarray).
            sizes: Batch sizes for each destination env rank.

        Returns:
            A list of action shards aligned with destination rank order.
        """
        assert sum(sizes) == actions.shape[0], (
            f"Number of actions ({actions.shape[0]}) must equal split sizes sum ({sum(sizes)})."
        )
        if isinstance(actions, np.ndarray):
            split_indices = np.cumsum(sizes[:-1]).tolist()
            return list(np.split(actions, split_indices, axis=0))
        return list(torch.split(actions, sizes, dim=0))

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        obs = obs_batch["obs"] if "obs" in obs_batch else obs_batch
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from env obs.")

    @staticmethod
    def _merge_obs_batches(obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_dicts = [
            obs_batch["obs"] if "obs" in obs_batch else obs_batch
            for obs_batch in obs_batches
        ]
        final_obs_list = [obs_batch.get("final_obs", None) for obs_batch in obs_batches]

        def _merge_obs_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
            merged: dict[str, Any] = {}
            for key in dicts[0].keys():
                values = [obs_dict[key] for obs_dict in dicts]
                first_non_none = next(
                    (value for value in values if value is not None), None
                )
                if first_non_none is None:
                    merged[key] = None
                elif isinstance(first_non_none, torch.Tensor):
                    merged[key] = torch.cat(values, dim=0)
                elif isinstance(first_non_none, list):
                    merged[key] = [item for sublist in values for item in sublist]
                else:
                    merged[key] = values
            return merged

        merged_obs = _merge_obs_dicts(obs_dicts)
        merged_final_obs = None
        if any(final_obs is not None for final_obs in final_obs_list):
            final_obs_or_obs = [
                final_obs if final_obs is not None else obs_dict
                for obs_dict, final_obs in zip(obs_dicts, final_obs_list)
            ]
            merged_final_obs = _merge_obs_dicts(final_obs_or_obs)

        return {"obs": merged_obs, "final_obs": merged_final_obs}

    def send_chunk_actions(
        self,
        output_channel: Channel,
        chunk_actions: torch.Tensor | np.ndarray,
        mode: Literal["train", "eval"] = "train",
    ):
        """Send action shards to mapped env ranks.

        Args:
            output_channel: Channel carrying rollout->env action chunks.
            chunk_actions: Predicted action chunk batch (tensor or ndarray).
            mode: Rollout mode, either ``"train"`` or ``"eval"``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_ranks[mode]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        chunk_actions_split = self._split_actions(chunk_actions, split_sizes)
        for (dst_rank, _), chunk_action_i in zip(
            dst_ranks_and_sizes, chunk_actions_split
        ):
            if isinstance(chunk_action_i, torch.Tensor):
                chunk_action_i = (
                    chunk_action_i.detach().cpu().contiguous()
                )  # for evaluation
            output_channel.put(
                chunk_action_i,
                key=CommMapper.build_channel_key(
                    self._rank, dst_rank, extra=f"{mode}_actions"
                ),
                async_op=True,
            )

    def _split_rollout_result(
        self, rollout_result: RolloutResult, sizes: list[int]
    ) -> list[RolloutResult]:
        def _split_optional_tensor(
            tensor: torch.Tensor | None,
        ) -> tuple[torch.Tensor | None, ...]:
            if tensor is None:
                return tuple(None for _ in sizes)
            return tuple(torch.split(tensor, sizes, dim=0))

        split_actions = _split_optional_tensor(rollout_result.actions)
        split_prev_logprobs = _split_optional_tensor(rollout_result.prev_logprobs)
        split_prev_values = _split_optional_tensor(rollout_result.prev_values)
        split_bootstrap_values = _split_optional_tensor(rollout_result.bootstrap_values)
        split_save_flags = _split_optional_tensor(rollout_result.save_flags)
        split_versions = _split_optional_tensor(rollout_result.versions)
        split_forward_inputs = (
            [{} for _ in sizes]
            if not rollout_result.forward_inputs
            else [
                {
                    key: torch.split(value, sizes, dim=0)[idx]
                    for key, value in rollout_result.forward_inputs.items()
                }
                for idx in range(len(sizes))
            ]
        )

        return [
            RolloutResult(
                actions=split_actions[idx],
                prev_logprobs=split_prev_logprobs[idx],
                prev_values=split_prev_values[idx],
                bootstrap_values=split_bootstrap_values[idx],
                save_flags=split_save_flags[idx],
                forward_inputs=split_forward_inputs[idx],
                versions=split_versions[idx],
            )
            for idx in range(len(sizes))
        ]

    def send_rollout_result(
        self,
        output_channel: Channel,
        rollout_result: RolloutResult,
        mode: Literal["train", "eval"] = "train",
    ):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_ranks[mode]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        split_rollout_results = self._split_rollout_result(rollout_result, split_sizes)
        for (dst_rank, _), rollout_result_i in zip(
            dst_ranks_and_sizes, split_rollout_results
        ):
            output_channel.put(
                rollout_result_i,
                key=CommMapper.build_channel_key(
                    self._rank, dst_rank, extra=f"{mode}_rollout_results"
                ),
                async_op=True,
            )

    def set_global_step(self, global_step: int):
        self.version = global_step
        if self.finished_episodes is None:
            self.finished_episodes = (
                self.version * self.total_num_train_envs * self.rollout_epoch
            )
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
