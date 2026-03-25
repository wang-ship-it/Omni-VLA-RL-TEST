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

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
from rlinf.runners.async_ppo_embodied_runner import AsyncPPOEmbodiedRunner
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.async_env_worker import AsyncEnvWorker
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)
    use_async_sac = cfg.algorithm.loss_type == "embodied_sac"
    use_async_ppo = bool(cfg.rollout.get("recompute_logprobs", False))

    if use_async_sac:
        from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
            AsyncEmbodiedSACFSDPPolicy,
        )

        actor_worker_cls = AsyncEmbodiedSACFSDPPolicy
        rollout_worker_cls = AsyncMultiStepRolloutWorker
        env_worker_cls = AsyncEnvWorker
        runner_cls = AsyncEmbodiedRunner
    elif use_async_ppo:
        from rlinf.workers.actor.async_ppo_fsdp_worker import (
            AsyncPPOEmbodiedFSDPActor,
        )

        actor_worker_cls = AsyncPPOEmbodiedFSDPActor
        rollout_worker_cls = AsyncMultiStepRolloutWorker
        env_worker_cls = AsyncEnvWorker
        runner_cls = AsyncPPOEmbodiedRunner
    else:
        if cfg.algorithm.loss_type == "embodied_sac":
            from rlinf.workers.actor.fsdp_sac_policy_worker import (
                EmbodiedSACFSDPPolicy,
            )

            actor_worker_cls = EmbodiedSACFSDPPolicy
        else:
            from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

            actor_worker_cls = EmbodiedFSDPActor
        rollout_worker_cls = MultiStepRolloutWorker
        env_worker_cls = EnvWorker
        runner_cls = EmbodiedRunner

    print(
        json.dumps(
            {
                "selected_actor_worker": actor_worker_cls.__name__,
                "selected_rollout_worker": rollout_worker_cls.__name__,
                "selected_env_worker": env_worker_cls.__name__,
                "selected_runner": runner_cls.__name__,
                "use_async_sac": use_async_sac,
                "use_async_ppo": use_async_ppo,
            },
            indent=2,
        )
    )

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")
    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )
    # Create rollout worker group
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = rollout_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    env_group = env_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    runner = runner_cls(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
