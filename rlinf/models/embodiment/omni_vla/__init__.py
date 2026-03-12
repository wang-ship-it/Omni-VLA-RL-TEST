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
# omni_vla model configs

import logging
import os

from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def get_model(cfg: DictConfig, torch_dtype=None):
    import glob
    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors
    import openpi.shared.normalize as _normalize

    from rlinf.models.embodiment.omni_vla.dataconfig import get_omni_vla_config
    from rlinf.models.embodiment.omni_vla.omni_vla_action_model import (
        OmniVLAConfig,
        OmniVLAForRLActionPrediction,
    )

    # config
    config_name = getattr(cfg.omni_vla, "config_name", None)
    if config_name is None:
        raise ValueError(
            "omni_vla.config_name must be specified in the model config. "
            "Available configs: omni_vla_libero, omni_vla_maniskill"
        )

    actor_train_config = get_omni_vla_config(config_name, model_path=cfg.model_path)
    actor_model_config = actor_train_config.model
    # Convert to OmniVLAConfig if it's not already (it should be if we defined it in dataconfig)
    if not isinstance(actor_model_config, OmniVLAConfig):
         actor_model_config = OmniVLAConfig(**actor_model_config.__dict__)
         
    override_config_kwargs = cfg.omni_vla
    if override_config_kwargs is not None:
        for key, val in override_config_kwargs.items():
            if hasattr(actor_model_config, key):
                 # Use object.__setattr__ because it's frozen
                 object.__setattr__(actor_model_config, key, val)

    # load model
    checkpoint_dir = download.maybe_download(str(cfg.model_path))
    weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not weight_paths:
        weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]

    model: OmniVLAForRLActionPrediction = OmniVLAForRLActionPrediction(
        actor_model_config
    )
    
    # Freeze parameters according to config
    model.set_requires_grad()

    # Load weights
    for weight_path in weight_paths:
        # strict=False because we might have extra heads or partial weights
        safetensors.torch.load_model(model, weight_path, strict=False)
    
    # Ensure correct dtype for specific parts if needed, usually handled by load_model or init
    # OmniVLA handles dtype in init mostly.

    # load data stats
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    norm_stats = None
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            logger.warning("data_config.asset_id is None, skipping norm_stats loading.")
        try:
            norm_stats_dir = os.path.join(checkpoint_dir, data_config.asset_id)
            norm_stats = _normalize.load(norm_stats_dir)
        except Exception as e:
            logger.warning(
                f"Failed to load norm_stats from checkpoint_dir={checkpoint_dir}, "
                f"asset_id={data_config.asset_id}: {e}. "
                "Model will run without input/output normalization."
            )

    # wrappers
    repack_transforms = transforms.Group()
    default_prompt = None
    
    # Prepare transforms list
    transforms_list = [
        *repack_transforms.inputs,
        transforms.InjectDefaultPrompt(default_prompt),
        *data_config.data_transforms.inputs,
    ]
    
    if norm_stats is not None:
         transforms_list.append(transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ))
            
    transforms_list.extend(data_config.model_transforms.inputs)

    output_transforms_list = [
        *data_config.model_transforms.outputs,
    ]
    
    if norm_stats is not None:
        output_transforms_list.append(transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ))
            
    output_transforms_list.extend([
        *data_config.data_transforms.outputs,
        *repack_transforms.outputs,
    ])

    model.setup_wrappers(
        transforms=transforms_list,
        output_transforms=output_transforms_list,
    )

    return model
