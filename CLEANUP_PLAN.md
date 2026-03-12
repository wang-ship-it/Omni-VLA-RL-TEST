# Omni_VLA 项目整理方案

## 现状分析

项目中存在两个几乎平行的目录：`omni_vla/` 和 `openpi/`。
`omni_vla` 是在 `openpi` 基础上扩展的，但把 openpi 的代码**整个复制**了一份进来，导致大量重复。

---

## 一、完全重复的文件（可删除 omni_vla 中的副本）

以下文件在 `omni_vla/` 和 `openpi/` 中**完全一致**，属于纯粹的冗余拷贝：

### src/openpi/shared/（全部一致）
| 文件 | 说明 |
|------|------|
| `shared/__init__.py` | |
| `shared/array_typing.py` | |
| `shared/download.py` | |
| `shared/download_test.py` | |
| `shared/image_tools.py` | |
| `shared/image_tools_test.py` | |
| `shared/nnx_utils.py` | |
| `shared/normalize.py` | |
| `shared/normalize_test.py` | |

### src/openpi/policies/（大部分一致）
| 文件 | 说明 |
|------|------|
| `policies/aloha_policy.py` | 完全一致 |
| `policies/droid_policy.py` | 完全一致 |
| `policies/libero_policy.py` | 完全一致 |

### src/openpi/training/（大部分一致）
| 文件 | 说明 |
|------|------|
| `training/data_loader.py` | 完全一致 |
| `training/data_loader_test.py` | 完全一致 |
| `training/droid_rlds_dataset.py` | 完全一致 |
| `training/optimizer.py` | 完全一致 |
| `training/sharding.py` | 完全一致 |
| `training/utils.py` | 完全一致 |
| `training/weight_loaders.py` | 完全一致 |
| `training/misc/polaris_config.py` | 完全一致 |
| `training/misc/roboarena_config.py` | 完全一致 |

### src/openpi/models/（大部分一致）
| 文件 | 说明 |
|------|------|
| `models/__init__.py` | 完全一致 |
| `models/gemma.py` | 完全一致 |
| `models/gemma_fast.py` | 完全一致 |
| `models/lora.py` | 完全一致 |
| `models/lora_test.py` | 完全一致 |
| `models/pi0.py` | 完全一致 |
| `models/pi0_fast.py` | 完全一致 |
| `models/pi0_test.py` | 完全一致 |
| `models/siglip.py` | 完全一致 |
| `models/tokenizer.py` | 完全一致 |
| `models/tokenizer_test.py` | 完全一致 |
| `models/utils/fsq_tokenizer.py` | 完全一致 |
| `models/vit.py` | 完全一致 |

### src/openpi/models_pytorch/（部分一致）
| 文件 | 说明 |
|------|------|
| `models_pytorch/preprocessing_pytorch.py` | 完全一致 |
| `models_pytorch/transformers_replace/` 全部 | 完全一致 |

### src/openpi/serving/
| 文件 | 说明 |
|------|------|
| `serving/websocket_policy_server.py` | 完全一致 |

### 其他
| 文件 | 说明 |
|------|------|
| `transforms.py`, `transforms_test.py` | 完全一致 |
| `conftest.py`, `__init__.py`, `py.typed` | 完全一致 |
| `examples/` 整个目录 | 完全一致（除了 omni_vla 多了 InternVLA） |
| `docs/` 整个目录 | 基本一致 |
| `scripts/compute_norm_stats.py` | 完全一致 |
| `scripts/train.py`, `scripts/train_test.py` | 完全一致 |
| 各种顶层文件（LICENSE, CONTRIBUTING.md 等） | 完全一致 |
| `pyproject.toml`, `uv.lock` | 完全一致 |

---

## 二、有修改的文件（需要合并差异到 openpi，再删除 omni_vla 副本）

| omni_vla 中的文件 | 改动内容 |
|---|---|
| `src/openpi/models/model.py` | 添加了 OmniVLA 模型加载逻辑 |
| `src/openpi/models/pi0_config.py` | 新增 5 个配置字段（spatial_expert_variant, freeze_* 等） |
| `src/openpi/policies/policy.py` | 使用重构后的 `Observation.from_dict()` 接口 |
| `src/openpi/policies/policy_config.py` | 引用 `model.reasoning_spatial_expert` 替代 `model.paligemma_with_expert` |
| `src/openpi/training/checkpoints.py` | 使用 TYPE_CHECKING guard 替代直接 import |
| `src/openpi/training/config.py` | **改动最大**：新增 omni_config 导入、omni_libero/omni_droid 训练配置、不同的默认路径和参数 |
| `src/openpi/models_pytorch/gemma_pytorch.py` | 修改了 image_features 获取方式 |
| `src/openpi/models_pytorch/pi0_pytorch.py` | 注释掉了 transformers_replace 校验 |
| `scripts/train_pytorch.py` | 新增 wandb 图片日志功能（30+ 行） |
| `scripts/serve_policy.py` | 新增 LIBERO checkpoint 配置 |

**处理方式：** 将 omni_vla 的修改合并回 `openpi/` 对应文件，使 openpi 成为唯一的源码。

---

## 三、omni_vla 独有的文件（需要迁移到 openpi 中）

这些是 OmniVLA 真正新增的代码，应该迁移到 `openpi/src/openpi/` 下：

### 核心模型（→ 移入 openpi/src/openpi/models_pytorch/）
| 文件 | 说明 |
|---|---|
| `models_pytorch/omni_vla.py` | OmniVLA 主模型（1100+ 行） |
| `models_pytorch/omni_config.py` | OmniVLA 配置 |
| `models_pytorch/g2vlm_pi0_pytorch.py` | G2VLM + Actor Expert 模型 |
| `models_pytorch/vlm_with_spatial.py` | VLM + Spatial Action Expert |

### VLM 数据处理（→ 移入 openpi/src/openpi/data_vlm/）
| 目录 | 说明 |
|---|---|
| `data_vlm/` 全部 21 个文件 | VLM 数据处理管线 |

### VGGT 模型（→ 移入 openpi/src/openpi/vggt/）
| 目录 | 说明 |
|---|---|
| `vggt/` 全部文件 | VGGT 模型实现 |

### VLM Expert（→ 移入 openpi/src/openpi/vlm_expert/）
| 目录 | 说明 |
|---|---|
| `vlm_expert/g2vlm/` | G2VLM expert |
| `vlm_expert/qwen2/` | Qwen2 语言模型 |
| `vlm_expert/qwen2vl/` | Qwen2-VL 图像处理 |
| `vlm_expert/dinov2_with_registers/` | DINOv2 |
| `vlm_expert/dinov3/` | DINOv3 |
| `vlm_expert/pi3/` | PI3 视觉模型 |

### 训练脚本
| 文件 | 说明 |
|---|---|
| `scripts/train_omni.py` | OmniVLA 专用训练入口 |

### 示例
| 文件 | 说明 |
|---|---|
| `examples/InternVLA_A1_3B/` | InternVLA 模型支持 |
| `examples/modeling_internvla_a1.py` | |

---

## 四、建议的最终目录结构

```
Omni_VLA/
├── openpi/                          # 唯一的模型源码（包含 openpi 原始 + omni_vla 扩展）
│   ├── src/openpi/
│   │   ├── models/                  # 原有 + pi0_config 扩展
│   │   ├── models_pytorch/          # 原有 + omni_vla.py, omni_config.py, g2vlm_pi0_pytorch.py, vlm_with_spatial.py
│   │   ├── policies/               # 原有（合并 omni_vla 修改）
│   │   ├── training/               # 原有（合并 config.py 修改）
│   │   ├── serving/                # 原有
│   │   ├── shared/                 # 原有
│   │   ├── data_vlm/              # ← 从 omni_vla 迁入
│   │   ├── vggt/                  # ← 从 omni_vla 迁入
│   │   └── vlm_expert/            # ← 从 omni_vla 迁入
│   ├── scripts/
│   │   ├── train.py
│   │   ├── train_pytorch.py       # 合并 wandb 日志修改
│   │   ├── train_omni.py          # ← 从 omni_vla 迁入
│   │   └── ...
│   └── examples/
│       ├── InternVLA_A1_3B/       # ← 从 omni_vla 迁入
│       └── ...
├── rlinf/                          # RL 基础设施（不变）
├── toolkits/                       # 工具集（不变）
├── data/                           # 数据（不变）
├── ray_utils/                      # Ray 工具（不变）
├── tests/                          # 测试（不变）
├── docker/                         # Docker（不变）
└── pyproject.toml                  # 顶层 rlinf 包配置
```

**删除 `omni_vla/` 整个目录**（所有内容已合并入 openpi 或确认为重复）。

---

## 五、执行步骤

### 步骤 1：迁移 omni_vla 独有文件到 openpi
```bash
# 迁移核心模型
cp omni_vla/src/openpi/models_pytorch/omni_vla.py openpi/src/openpi/models_pytorch/
cp omni_vla/src/openpi/models_pytorch/omni_config.py openpi/src/openpi/models_pytorch/
cp omni_vla/src/openpi/models_pytorch/g2vlm_pi0_pytorch.py openpi/src/openpi/models_pytorch/
cp omni_vla/src/openpi/models_pytorch/vlm_with_spatial.py openpi/src/openpi/models_pytorch/

# 迁移新增模块
cp -r omni_vla/src/openpi/data_vlm openpi/src/openpi/
cp -r omni_vla/src/openpi/vggt openpi/src/openpi/
cp -r omni_vla/src/openpi/vlm_expert openpi/src/openpi/

# 迁移训练脚本
cp omni_vla/scripts/train_omni.py openpi/scripts/

# 迁移示例
cp -r omni_vla/examples/InternVLA_A1_3B openpi/examples/
cp omni_vla/examples/modeling_internvla_a1.py openpi/examples/
```

### 步骤 2：合并有差异的文件
对第二节列出的 10 个文件逐个 diff，将 omni_vla 的修改合并到 openpi 对应文件中。

### 步骤 3：验证 import 路径
所有 import 仍然使用 `from openpi.xxx import ...`，路径不需要变。
但需要确认 `rlinf/models/embodiment/omni_vla/` 中的 import 指向正确。

### 步骤 4：测试
确认 `openpi/` 下的代码可正常 import 和运行。

### 步骤 5：删除 omni_vla/ 目录

---

## 六、其他清理建议

1. **`__pycache__/` 目录**：`omni_vla/` 和 `openpi/` 中都有，应加入 `.gitignore` 并删除
2. **`.DS_Store` 文件**：多处存在，应加入 `.gitignore` 并删除
3. **`omni_vla/Omni_VLA_Analysis.md`**：分析文档，迁移后可删除
4. **`omni_vla/scripts/optimization_summary.md`**：可选保留或迁入 docs/
5. **重复的配置文件**：`omni_vla/.gitignore`, `.dockerignore`, `.pre-commit-config.yaml` 等与 openpi 重复，合并后删除
