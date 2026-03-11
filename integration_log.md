# Omni_VLA into RLinf Integration Log

This document records the process of integrating `Omni_VLA` (optimized `openpi`) into the `RLinf` framework.

## Task Overview
- **Goal**: Replace or integrate `Omni_VLA` into `RLinf` to use the optimized features.
- **Current State**: `RLinf` uses the official `openpi` package. `Omni_VLA` is a local directory with optimizations.

## Progress Log

### [2026-03-11] Initialization
- Created this log file.
- Started analysis of `RLinf` dependency on `openpi`.

### [2026-03-11] 本地安装排障记录（openpi）
- 执行 `bash requirements/install.sh embodied --model openpi --env maniskill_libero --use-mirror`，安装流程在 `download_assets.sh` 阶段报错：`mani_skill is not installed`。
- 进一步验证后确认并非 `mani_skill` 包缺失，而是导入链路中缺少 `sapien`：`ModuleNotFoundError: No module named 'sapien'`。
- 核心原因：
  - `requirements/embodied/envs/common.txt` 中 `sapien==3.0.1` 仅在 Linux 平台安装（带有 `platform_system=='Linux'` 条件）。
  - 当前为 macOS 环境，导致 `mani_skill` 安装后仍无法正常 `import`。
  - `openpi` 官方依赖中包含 `jax[cuda12]==0.5.3`，该依赖在 macOS 上无可用 wheel，完整官方安装不可达。
- 结论：
  - 在当前 macOS 本机环境下，`openpi + maniskill_libero` 官方完整安装路径不可完全打通。
  - 可行替代方案是仅做"核心代码依赖安装/静态集成验证"，不要绑定 `maniskill_libero` 环境安装链路。

---

### [2026-03-11] Bug 修复（Claude Code）

已修复以下问题：

1. **P5 (entropy 置零)**: `get_log_prob_value` 中 `chains_entropy = torch.zeros_like(...)` 改为 `torch.stack(chains_entropy, dim=1)`，恢复 entropy bonus 功能
2. **P1 (flow_noise 不支持)**: 移除 `ExploreNoiseNet` 引用，在 `__init__` 中添加断言限制 `noise_method` 仅支持 `flow_sde`，`sample_mean_var_val` 中非 flow_sde 分支改为 raise ValueError
3. **P4 (vlm_value 空实现)**: 添加 `NotImplementedError`，禁止 `value_after_vlm=True`；移除 `sample_actions` 中无效的 pass 块
4. **P3 (norm_stats 静默失败)**: 添加 `logger.warning` 替代 `except: pass`
5. **P2 (config_name None)**: 添加清晰的 `ValueError` 提示可用配置名
6. **P6 (forward 返回值解构错误)**: `reasoning_spatial_expert.forward()` 返回 `([prefix, middle, suffix], past_key_values)`。修复两处错误的 `(...)[1]` 解构为正确的 `_, past_key_values = ...` 格式
7. **P8 (eval 配置)**: 新增 `examples/embodiment/eval/libero_spatial_ppo_omni_vla_eval.yaml`
8. **清理**: 移除未使用的 `prefix_len` 变量

未修复（暂不需要）：
- P9/P10 (更多环境 dataconfig 和 policies): 按需扩展，当前 libero 和 maniskill 足够初始验证
- P12 (jax 依赖): 沿用 openpi 模式，不影响功能
- P13 (hardcoded proj_width): 验证后 1024/2048 与实际模型维度匹配，无需修改

---

### [2026-03-11] 集成审查报告（Claude Code Review）

## 一、已完成部分总结

### 1. 安装脚本 (`requirements/install.sh`)
- ✅ 新增 `install_omni_vla_model()` 函数，支持7个环境（behavior, maniskill_libero, metaworld, calvin, robocasa, robotwin, minimal）
- ✅ 使用 `-e "$SCRIPT_DIR/../Omni_VLA"` 以 editable 模式安装
- ✅ 复制 transformers 替换文件（从 `Omni_VLA/src/openpi/models_pytorch/transformers_replace/`）
- ✅ `SUPPORTED_MODELS` 数组中已添加 `"omni_vla"`

### 2. 模型注册 (`rlinf/config.py`)
- ✅ `SupportedModel` 枚举中添加了 `OMNI_VLA = ("omni_vla", "embodied")`

### 3. 模型加载入口 (`rlinf/models/__init__.py`)
- ✅ 添加了条件导入：`from rlinf.models.embodiment.omni_vla import get_model`

### 4. 核心模型文件
- ✅ `rlinf/models/embodiment/omni_vla/__init__.py` — 模型加载、权重加载、transforms 配置
- ✅ `rlinf/models/embodiment/omni_vla/omni_vla_action_model.py` — 856行，包含 `OmniVLAConfig` + `OmniVLAForRLActionPrediction`
- ✅ `rlinf/models/embodiment/omni_vla/dataconfig.py` — 2个数据配置（libero, maniskill）

### 5. 配置文件
- ✅ `examples/embodiment/config/model/omni_vla.yaml`
- ✅ `examples/embodiment/config/libero_spatial_ppo_omni_vla_quickstart.yaml`

### 6. 新增算法
- ✅ `rlinf/algorithms/losses.py` 中新增 `@register_policy_loss("gspo")`

---

## 二、发现的问题和漏洞

### 🔴 高优先级（可能导致运行时错误）

#### P1: `flow_noise` 方法引用了 `ExploreNoiseNet` 但未导入
- **文件**: `omni_vla_action_model.py:104-105`
- **问题**: `_no_split_modules` 中引用了 `ExploreNoiseNet`，但 omni_vla 文件从未 import 该模块，且 `__init__` 中也没有像 openpi 版本那样初始化 `self.noise_head`
- **对比**: openpi 版本在 `__init__` L162-170 中有 `self.noise_head = ExploreNoiseNet(...)` 的初始化
- **影响**: 当 `noise_method="flow_noise"` 时，FSDP 分片会引用不存在的模块，且无法正确计算噪声
- **修复**: 要么添加 `ExploreNoiseNet` 的导入和初始化，要么在 OmniVLA 中移除 `flow_noise` 支持并在配置中限制

#### P2: `__init__.py` 中 `config_name` 为 None 时无处理
- **文件**: `omni_vla/__init__.py:33-39`
- **问题**: 当 `config_name` 为 None 时只有 `pass`，后续 `get_omni_vla_config(None, ...)` 会直接报错 `Config 'None' not found`
- **修复**: 添加默认值或提前抛出清晰的错误信息

#### P3: `norm_stats` 加载失败被静默忽略
- **文件**: `omni_vla/__init__.py:87-91`
- **问题**: `try/except Exception: pass` 会吞掉所有错误。如果 norm_stats 加载失败，模型将不做归一化，导致推理结果错误但不报错
- **修复**: 至少打印 warning，或者在必须有 norm_stats 的场景中抛错

#### P4: `use_vlm_value=True` 时值估计为空
- **文件**: `omni_vla_action_model.py:486-494`
- **问题**: `sample_actions()` 中当 `self.use_vlm_value` 为 True 时只有 `pass`，不计算任何 value，而后续 `values = torch.stack(values, dim=1)` 可能为空或维度不对
- **修复**: 要么实现 VLM value 计算，要么在配置验证时禁止 `value_after_vlm=True`

#### P5: `get_log_prob_value` 中 entropy 被置零
- **文件**: `omni_vla_action_model.py:833`
- **问题**: `chains_entropy = torch.zeros_like(chains_log_probs)` 直接覆盖了前面计算的 `chains_entropy` 列表，导致 entropy bonus 永远为0
- **对比**: 这会使 `entropy_type: token_level` 配置失效
- **修复**: 改为 `chains_entropy = torch.stack(chains_entropy, dim=1)`

#### P6: `reasoning_spatial_expert.forward` 返回值解构可能不匹配
- **文件**: `omni_vla_action_model.py:429-435, 470-476, 661-667, 752-758`
- **问题**: 多处对 `self.reasoning_spatial_expert.forward()` 返回值进行解构，但返回结构取决于 Omni_VLA 中 `VLMWithSpatialActionExpertModel.forward()` 的实现。代码中有的用 `(_, past_key_values) = ...()[1]`，有的用 `(_, _, _), past_key_values = ...`。需要验证 `reasoning_spatial_expert.forward` 的实际返回格式
- **修复**: 需确认 `VLMWithSpatialActionExpertModel.forward` 的返回签名并统一解构方式

### 🟡 中优先级（功能缺失或不完整）

#### P7: 缺少 `flow_noise` 和 `flow_cps` 的完整实现
- **文件**: `omni_vla_action_model.py:718-722`
- **问题**: `sample_mean_var_val` 中 `noise_method != "flow_sde"` 时用了简化的 ODE 采样（`x_t_std = 0`），没有真正的 flow_noise/flow_cps 实现
- **影响**: 配置中设置 `flow_noise` 或 `flow_cps` 不会报错但行为不正确

#### P8: 缺少 eval 配置文件
- **问题**: openpi 有 `examples/embodiment/eval/libero_spatial_ppo_openpi_eval.yaml`，但 omni_vla 没有对应的 eval yaml
- **影响**: 无法直接用标准流程运行 omni_vla 的评估

#### P9: dataconfig 只有2个环境配置
- **文件**: `omni_vla/dataconfig.py`
- **问题**: 只定义了 `omni_vla_libero` 和 `omni_vla_maniskill`，而安装脚本支持7个环境
- **影响**: metaworld, calvin, robocasa, robotwin, behavior 环境无法使用

#### P10: 缺少 policies 目录
- **问题**: openpi 有 `rlinf/models/embodiment/openpi/policies/` 目录下9个环境的 policy wrapper，omni_vla 完全没有
- **影响**: 如果 RLinf 的 rollout 流程依赖这些 policy wrapper，omni_vla 无法正常执行

### 🟢 低优先级（代码质量/一致性）

#### P11: `_no_split_names` 缺少 pi05 相关名称
- **文件**: `omni_vla_action_model.py:109-116`
- **问题**: 相比 openpi 版本缺少 `"lm_head"`, `"time_mlp_in"`, `"time_mlp_out"`。如果 OmniVLA 模型中有这些层，FSDP 分片可能不正确
- **修复**: 检查 OmniVLA 模型结构，补充必要的 no_split_names

#### P12: `jax` 依赖在 omni_vla 中仍然被使用
- **文件**: `omni_vla_action_model.py:21, 185, 194, 201, 219, 234`
- **问题**: 代码中使用 `jax.tree.map` 来做 tree 操作。虽然 Omni_VLA 号称是纯 PyTorch，但 RLinf 集成层仍依赖 jax（沿用 openpi 模式）
- **影响**: 安装 omni_vla 时仍需安装 jax。这不是 bug 但增加了依赖复杂度

#### P13: `value_after_vlm` 的 proj_width 硬编码
- **文件**: `omni_vla_action_model.py:146-148`
- **问题**: `proj_width = 2048` (vlm) 或 `1024` (action) 是硬编码的。OmniVLA 的 hidden_size 可能不同于 openpi
- **修复**: 应从 `self.config` 中读取实际维度

---

## 三、修复优先级建议

1. **P5 (entropy 置零)** — 一行修复，影响训练效果
2. **P1 (flow_noise 缺失)** — 如果不用 flow_noise 可暂缓，否则需补全
3. **P3 (norm_stats 静默失败)** — 加 warning 即可
4. **P4 (vlm_value 空实现)** — 加断言禁止即可
5. **P2 (config_name None)** — 加默认值或错误提示
6. **P6 (forward 返回值)** — 需要实际运行验证
7. **P8 (eval yaml)** — 补充配置文件
8. **P9/P10 (更多环境支持)** — 按需扩展

---

## 四、集成架构概览

```
RLinf
├── requirements/install.sh          # ✅ omni_vla 安装入口
├── rlinf/config.py                  # ✅ OMNI_VLA 枚举
├── rlinf/models/__init__.py         # ✅ 条件导入
├── rlinf/models/embodiment/omni_vla/
│   ├── __init__.py                  # ✅ get_model() 加载逻辑
│   ├── omni_vla_action_model.py     # ✅ 核心模型（有上述问题）
│   ├── dataconfig.py                # ⚠️ 仅2个环境
│   ├── policies/                    # ❌ 缺失
│   └── eval configs                 # ❌ 缺失
├── rlinf/algorithms/losses.py       # ✅ gspo loss
└── examples/embodiment/config/
    ├── model/omni_vla.yaml          # ✅ 模型配置
    └── libero_spatial_ppo_omni_vla_quickstart.yaml  # ✅ 训练配置
```

依赖关系：`Omni_VLA/` (editable install) → 提供 `openpi.*` namespace → `rlinf` 通过 `from openpi.models_pytorch.omni_vla import OmniVLA` 等使用
