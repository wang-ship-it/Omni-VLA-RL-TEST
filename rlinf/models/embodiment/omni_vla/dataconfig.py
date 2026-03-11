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

import dataclasses
import difflib
from typing import Optional

import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from openpi.training.config import (
    AssetsConfig,
    DataConfig,
    TrainConfig,
)

from rlinf.models.embodiment.openpi.dataconfig.libero_dataconfig import (
    LeRobotLiberoDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.maniskill_dataconfig import (
    LeRobotManiSkillDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.metaworld_dataconfig import (
    LeRobotMetaworldDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.calvin_dataconfig import (
    LeRobotCalvinDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.robocasa_dataconfig import (
    LeRobotRobocasaDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.robotwin_aloha_dataconfig import (
    LeRobotAlohaDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.behavior_dataconfig import (
    LeRobotBehaviorDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.gsenv_dataconfig import (
    LeRobotGSEnvDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.franka_dataconfig import (
    CustomDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.franka_co_training_dataconfig import (
    LeRobotFrankaEEDataConfig,
)

from rlinf.models.embodiment.omni_vla.omni_vla_action_model import OmniVLAConfig

# Define OmniVLA configs corresponding to OpenPi configs
_CONFIGS = [
    TrainConfig(
        name="omni_vla_libero",
        model=OmniVLAConfig(),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/omni_vla_libero/assets"),
            extra_delta_transform=True,
        ),
        pytorch_weight_path="checkpoints/torch/omni_vla_libero",
    ),
    TrainConfig(
        name="omni_vla_maniskill",
        model=OmniVLAConfig(),
        data=LeRobotManiSkillDataConfig(
            repo_id="physical-intelligence/maniskill",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/omni_vla_maniskill/assets"),
            extra_delta_transform=False,
        ),
        pytorch_weight_path="checkpoints/torch/omni_vla_maniskill",
    ),
    # Add other configs as needed, defaulting to OmniVLAConfig
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def _override_with_model_path(config: TrainConfig, model_path: str) -> TrainConfig:
    """Return a copy of the config with assets/weight paths set from model_path."""
    data_config = config.data
    if (
        dataclasses.is_dataclass(data_config)
        and hasattr(data_config, "assets")
        and dataclasses.is_dataclass(data_config.assets)
    ):
        data_config = dataclasses.replace(
            data_config,
            assets=dataclasses.replace(data_config.assets, assets_dir=model_path),
        )

    replace_kwargs = {
        "data": data_config,
        "pytorch_weight_path": model_path,
    }
    if dataclasses.is_dataclass(config) and any(
        field.name == "assets_dirs" for field in dataclasses.fields(config)
    ):
        replace_kwargs["assets_dirs"] = model_path

    return dataclasses.replace(config, **replace_kwargs)


def get_omni_vla_config(
    config_name: str, model_path: Optional[str] = None, batch_size: Optional[int] = None
) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(
            config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0
        )
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    config = _CONFIGS_DICT[config_name]
    if model_path is not None:
        config = _override_with_model_path(config, model_path)
    if batch_size is not None:
        config = dataclasses.replace(config, batch_size=batch_size)

    return config
