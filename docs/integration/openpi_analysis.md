# OpenPI 在 RLinf 中的集成分析

本文档分析了官方 `openpi` 包目前是如何集成到 `RLinf` 框架中的。此分析是将其替换为优化后的 `Omni_VLA` 的先决条件。

## 1. 安装

`openpi` 包通过 `requirements/install.sh` 脚本进行安装。

- **来源**：直接从 GitHub 仓库安装：
  ```bash
  uv pip install git+${GITHUB_PREFIX}https://github.com/RLinf/openpi
  ```
- **安装后步骤**：安装脚本还执行以下操作：
  - 使用来自 `openpi` (`openpi/models_pytorch/transformers_replace/`) 的自定义版本替换特定的 `transformers` 文件。
  - 使用 `embodied/download_assets.sh --assets openpi` 下载资源。

## 2. Python 代码使用情况

`openpi` 在 `RLinf` 代码库的几个关键区域被使用。

### 2.1 模型包装器 (`rlinf/models/embodiment/openpi/`)
这是核心集成点，其中 `openpi` 模型被包装以适应 `RLinf` 的策略接口。

- **`openpi_action_model.py`**：
  - 导入：
    ```python
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
    from openpi import transforms as _transforms
    from openpi.models import model as _model
    from openpi.models.pi0_config import Pi0Config
    ```
  - 类 `OpenPi0ForRLActionPrediction`：继承自 `PI0Pytorch` 和 `BasePolicy`。它重写了 `forward` 和 `sample_actions` 等方法，以适应 RL 训练（PPO 等）和推理。

- **`__init__.py`**：
  - 导入：
    ```python
    import openpi.shared.download as download
    import openpi.transforms as transforms
    from openpi.training import checkpoints as _checkpoints
    ```
  - 函数 `get_model`：加载模型权重，标准化统计数据，并使用 `openpi` 实用工具设置转换。

### 2.2 SFT Worker (`rlinf/workers/sft/fsdp_vla_sft_worker.py`)
用于 VLA 模型的监督微调（SFT）。

- 导入：
  ```python
  import openpi.training.data_loader as openpi_data_loader
  ```
- 用法：使用 `openpi_data_loader.create_data_loader` 创建用于训练的数据加载器。

### 2.3 评估脚本 (`toolkits/eval_scripts_openpi/`)
用于在各种基准测试中评估模型。

- **`__init__.py`**：
  - 导入：
    ```python
    import openpi.policies.policy as _policy
    import openpi.shared.download as download
    import openpi.transforms as transforms
    from openpi.models_pytorch import pi0_pytorch
    from openpi.training import checkpoints as _checkpoints
    from openpi.training import config as _config
    ```
  - 函数 `create_trained_policy`：使用 `openpi` 的基础设施从检查点重建策略。

### 2.4 配置 (`rlinf/config.py`)
- 定义了 `SupportedModel.OPENPI` 枚举成员。

### 2.5 检查点转换 (`rlinf/utils/ckpt_convertor/`)
- **`convert_openpi_jax_to_python.py`**：可能使用 `openpi` 加载 JAX 检查点并进行转换。

## 3. 关键依赖摘要

为了成功用 `Omni_VLA` 替换 `openpi`，来自 `Omni_VLA` 的以下模块/类需要兼容或进行适配：

- `models_pytorch.pi0_pytorch.PI0Pytorch`
- `models_pytorch.pi0_pytorch.make_att_2d_masks`
- `models.pi0_config.Pi0Config`
- `models.model`
- `transforms`
- `training.data_loader`
- `training.checkpoints`
- `training.config`
- `shared.download`
- `policies.policy`

## 4. 下一步计划

- 分析 `Omni_VLA` 以确认它是否提供这些模块以及 API 是否匹配。
- 修改 `install.sh` 以安装 `Omni_VLA` 而不是 `openpi`。
- 更新 `RLinf` 中的导入以指向新包（如果包名更改）或确保 `Omni_VLA` 安装为 `openpi`。
