# Omni_VLA

> 面向具身智能的多专家协同视觉-语言-动作模型

Omni_VLA 是在 [OpenPI](https://github.com/Physical-Intelligence/openpi)（Physical Intelligence 开源机器人模型 π₀/π₀-FAST/π₀.₅）基础上深度扩展的下一代 Vision-Language-Action（VLA）模型框架。通过引入多专家协同注意力机制、Qwen3-VL 视觉理解主干以及 VGGT 空间感知编码器，Omni_VLA 显著提升了机器人对复杂场景的理解与操控能力，并无缝集成到 [RLinf](https://github.com/openrlhf/rlinf) 强化学习训练框架中，支持端到端的在线 RL 策略优化。

---

## 核心亮点

| 特性 | 描述 |
|------|------|
| **多专家协同注意力** | Reasoning / Spatial / Action 三专家在每一 Transformer 层共享注意力空间，实现深度特征融合 |
| **VGGT 空间编码** | 1B 参数的多视角空间聚合器，为操控任务提供精确的几何表征 |
| **Qwen3-VL 推理主干** | 高性能视觉-语言理解，支持自然语言指令泛化 |
| **Flow Matching 动作生成** | 基于连续时间流匹配的 50 步动作序列生成，支持 SDE 采样探索 |
| **RLinf 深度集成** | 与 RLinf 强化学习框架无缝对接，支持 PPO / GRPO 等在线 RL 算法 |
| **多平台支持** | Libero、Maniskill、DROID、ALOHA 等多种机器人平台 |

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                         OmniVLA 推理流程                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  RGB Images ──→ [VGGT Spatial Encoder (1B)]                      │
│       │              ↓ spatial tokens                            │
│       │         ┌────────────────────────────────────┐           │
│       └────────→│  Reasoning Expert (PaliGemma ~3B)  │           │
│                 │  Spatial  Expert  (Gemma-300M)     │           │
│  Language ─────→│  Action   Expert  (Gemma-300M)     │           │
│  Instruction    │                                    │           │
│                 │  ← Shared Multi-Head Attention →   │           │
│                 └────────────────────────────────────┘           │
│                              ↓                                    │
│                    [Flow Matching Head]                           │
│                              ↓                                    │
│                  Action Chunk (50 × action_dim)                   │
└─────────────────────────────────────────────────────────────────┘
```

### 多专家协同注意力机制

Omni_VLA 的核心创新在于**层级共享注意力**：三个专家不是串行或并行地独立处理，而是在每一个 Transformer 层将各自的 Q/K/V 拼接后进行统一的全局注意力计算：

```python
# vlm_with_spatial.py: compute_layer_complete()
# 三专家 Q/K/V 沿序列维度拼接
query_states = torch.cat([reasoning_q, spatial_q, action_q], dim=2)
key_states   = torch.cat([reasoning_k, spatial_k, action_k], dim=2)
value_states = torch.cat([reasoning_v, spatial_v, action_v], dim=2)

# 统一计算全局注意力，特征在最底层实现流通
output = attention(query_states, key_states, value_states)
```

这种设计带来的优势：
- 推理知识在每层直接流向空间/动作专家，消除信息瓶颈
- 三个专家参数量远小于单一大模型，训练效率更高
- 可独立冻结或微调各专家，灵活适配不同任务

### 子模块参数规模

| 模块 | 基础模型 | 参数量 | bf16 显存 |
|------|---------|--------|---------|
| Reasoning Expert | PaliGemma (SigLIP + Gemma-2B) | ~3B | ~6 GB |
| Spatial Expert | Gemma-300M | ~311M | ~0.6 GB |
| Action Expert | Gemma-300M | ~311M | ~0.6 GB |
| VGGT Encoder | ViT Aggregator | ~1B | ~2 GB |
| **合计** | | **~4.6B** | **~9.2 GB** |

---

## 项目结构

```
Omni_VLA/
├── omni_vla/                          # OmniVLA 核心实现（基于 OpenPI 深度扩展）
│   ├── src/openpi/
│   │   ├── models_pytorch/            # PyTorch 模型实现
│   │   │   ├── omni_vla.py            # OmniVLA 主模型（初始化、前向、推理、训练）
│   │   │   ├── omni_config.py         # OmniVLA 配置（继承 Pi0Config）
│   │   │   ├── vlm_with_spatial.py    # 多专家融合层（共享注意力核心）
│   │   │   ├── g2vlm_pi0_pytorch.py   # G2VLM + Actor 专家
│   │   │   ├── gemma_pytorch.py       # PaliGemma 专家封装
│   │   │   └── pi0_pytorch.py         # Pi0 基础模型工具
│   │   ├── models/                    # 基础模型定义（JAX）
│   │   ├── vlm_expert/                # 视觉-语言子模型
│   │   │   ├── g2vlm/                 # G2VLM（基于 Qwen2-VL）
│   │   │   ├── qwen2/                 # Qwen2 语言主干
│   │   │   ├── qwen2vl/               # Qwen2-VL 多模态模型
│   │   │   ├── dinov2_with_registers/ # DINOv2 + 记忆寄存器空间特征
│   │   │   └── dinov3/                # DINOv3 变体
│   │   ├── vggt/models/               # VGGT 空间推理（1B 参数）
│   │   │   ├── vggt.py                # VGGT 聚合器主体
│   │   │   └── aggregator.py          # 多视角聚合逻辑
│   │   ├── data_vlm/                  # VLM 数据处理
│   │   │   ├── data_utils.py          # Tensor 操作、RoPE 索引、稀疏掩码
│   │   │   ├── dataset_base.py        # 基础数据集类
│   │   │   ├── transforms.py          # 图像/动作预处理
│   │   │   └── transforms_vggt.py     # VGGT 专用变换
│   │   ├── training/                  # 训练工具与配置
│   │   │   ├── config.py              # 训练配置（归一化统计、数据加载、超参数）
│   │   │   ├── data_loader.py         # LeRobot 数据集处理
│   │   │   └── weight_loaders.py      # 检查点加载/续训
│   │   ├── policies/                  # 策略封装（libero、droid、aloha）
│   │   └── shared/                    # 公共工具（归一化、下载）
│   ├── scripts/
│   │   ├── train_omni.py              # OmniVLA 训练入口（SFT）
│   │   ├── train_pytorch.py           # PyTorch 通用训练脚本
│   │   ├── serve_policy.py            # 策略服务器（推理服务）
│   │   └── compute_norm_stats.py      # 计算动作/状态归一化统计量
│   ├── examples/                      # 各平台使用示例
│   │   ├── InternVLA_A1_3B/           # InternVLA 参考架构
│   │   ├── libero/                    # Libero 仿真示例
│   │   ├── droid/                     # DROID 机器人示例
│   │   ├── aloha_*/                   # ALOHA 双臂示例
│   │   └── simple_client/             # 最简推理客户端
│   └── docs/                          # 模型文档
│
├── openpi/                            # 官方 OpenPI 镜像（未修改的参照副本）
│
├── rlinf/                             # RLinf 强化学习框架集成层
│   ├── models/embodiment/
│   │   ├── omni_vla/                  # OmniVLA RLinf 接入
│   │   │   ├── __init__.py            # 模型加载与变换配置
│   │   │   ├── omni_vla_action_model.py  # RLinf 兼容策略封装
│   │   │   └── dataconfig.py          # 环境专用数据配置
│   │   └── openpi/                    # Pi0 原版接入（参照）
│   ├── algorithms/
│   │   └── losses.py                  # gspo 损失函数（新增）
│   └── workers/                       # 训练/rollout Workers
│
├── examples/embodiment/               # RLinf 在线 RL 训练示例
│   ├── config/
│   │   ├── model/omni_vla.yaml        # OmniVLA 模型配置
│   │   └── libero_spatial_ppo_omni_vla_quickstart.yaml  # 快速启动配置
│   ├── eval/                          # 评估配置
│   └── run_embodiment.sh              # 训练启动脚本
│
├── docs/
│   ├── integration/
│   │   ├── omni_vla_optimization_analysis.md   # 显存优化分析
│   │   └── openpi_analysis.md                  # OpenPI 架构分析
│   └── debug_log_20260312.md          # 已知问题与修复记录
│
├── integration_log.md                 # RLinf 集成进度日志（Jan–Mar 2026）
├── CLEANUP_PLAN.md                    # 代码去重计划
└── pyproject.toml                     # 项目依赖配置
```

---

## 训练流程

### 1. 数据准备

Omni_VLA 使用 [LeRobot](https://github.com/huggingface/lerobot) 格式的 HuggingFace 数据集：

```bash
# 计算归一化统计量（首次训练前必须执行）
python omni_vla/scripts/compute_norm_stats.py \
    --config-name libero_spatial
```

### 2. 监督微调（SFT）

```bash
# 使用 train_omni.py 进行行为克隆预训练
uv run python omni_vla/scripts/train_omni.py \
    --config-name omni_vla_libero_spatial \
    --exp-name my_experiment
```

### 3. 在线强化学习（RLinf PPO）

```bash
# 使用 RLinf 框架进行在线 RL 微调
bash examples/embodiment/run_embodiment.sh \
    examples/embodiment/config/libero_spatial_ppo_omni_vla_quickstart.yaml
```

### 4. 推理服务

```bash
# 启动策略服务器
uv run python omni_vla/scripts/serve_policy.py \
    --config-name omni_vla \
    --checkpoint-path /path/to/checkpoint
```

---

## 显存需求

| 场景 | 显存需求 | 推荐硬件 |
|------|---------|---------|
| 推理（Inference） | > 10 GB | RTX 3090 / A100 |
| LoRA 微调 | > 24 GB | RTX 4090 / A100 40G |
| 全量 SFT | > 70 GB | 2× A100 80G |
| OmniVLA + RLinf PPO | > 75 GB | 2–4× A100 80G |

**显存优化建议：**

```yaml
# examples/embodiment/config/libero_spatial_ppo_omni_vla_quickstart.yaml

# 1. 开启梯度检查点（节省 ~15-18 GB，训练速度降低约 25%）
gradient_checkpointing: true

# 2. 降低 micro_batch_size（默认 8，OOM 时尝试 4）
micro_batch_size: 8

# 3. 关闭 FSDP 分片（小规模训练使用 no_shard 更稳定）
fsdp:
  sharding_strategy: no_shard
  use_orig_params: true

# 4. 减少并发环境数
total_num_envs: 4
```

---

## 安装

### 环境要求

- Python 3.11
- CUDA 12.x
- [uv](https://github.com/astral-sh/uv) 包管理器

### 安装步骤

```bash
# 克隆项目
git clone <repo-url> Omni_VLA
cd Omni_VLA

# 安装依赖（跳过 Git LFS 大文件）
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e omni_vla/

# 应用 transformers 补丁（必须执行）
cp -r omni_vla/src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

### 通过 RLinf 安装

```bash
# 使用 RLinf 安装脚本（自动处理 omni_vla 可编辑安装）
bash requirements/install.sh
```

---

## 核心依赖

| 依赖包 | 版本 | 用途 |
|--------|------|------|
| `torch` | 2.7.1 | 深度学习框架 |
| `transformers` | 4.53.2 | Hugging Face 模型库 |
| `jax[cuda12]` | 0.5.3 | JAX 数值计算（Pi0 JAX 部分） |
| `lerobot` | latest | 机器人数据集与策略 |
| `einops` | ≥0.8.0 | Tensor 变换工具 |
| `wandb` | ≥0.19.1 | 实验追踪 |
| `opencv-python` | ≥4.10.0 | 图像处理 |

---

## 支持的平台与环境

| 平台 | 环境 | 状态 |
|------|------|------|
| [Libero](https://github.com/Lifelong-Robot-Learning/LIBERO) | libero_spatial、libero_goal 等 | 已支持 |
| [ManiSkill](https://github.com/haosulab/ManiSkill) | PickCube 等 | 已支持 |
| [DROID](https://droid-dataset.github.io/) | 真实机器人数据集 | 部分支持 |
| [ALOHA](https://mobile-aloha.github.io/) | 双臂操控 | 部分支持 |
| [MetaWorld](https://meta-world.github.io/) | 50 种操控任务 | 计划中 |
| [CALVIN](http://calvin.cs.uni-freiburg.de/) | 长程指令跟随 | 计划中 |

---

## 关键文件速查

| 场景 | 文件 |
|------|------|
| 主模型实现 | [omni_vla/src/openpi/models_pytorch/omni_vla.py](omni_vla/src/openpi/models_pytorch/omni_vla.py) |
| 多专家融合核心 | [omni_vla/src/openpi/models_pytorch/vlm_with_spatial.py](omni_vla/src/openpi/models_pytorch/vlm_with_spatial.py) |
| 模型配置 | [omni_vla/src/openpi/models_pytorch/omni_config.py](omni_vla/src/openpi/models_pytorch/omni_config.py) |
| SFT 训练入口 | [omni_vla/scripts/train_omni.py](omni_vla/scripts/train_omni.py) |
| RLinf PPO 配置 | [examples/embodiment/config/libero_spatial_ppo_omni_vla_quickstart.yaml](examples/embodiment/config/libero_spatial_ppo_omni_vla_quickstart.yaml) |
| RLinf 策略封装 | [rlinf/models/embodiment/omni_vla/omni_vla_action_model.py](rlinf/models/embodiment/omni_vla/omni_vla_action_model.py) |
| 调试日志 | [docs/debug_log_20260312.md](docs/debug_log_20260312.md) |
| 集成进度 | [integration_log.md](integration_log.md) |

---

## 设计理念

### 为什么使用多专家而非单一大模型？

1. **职责解耦**：语言推理、空间理解、动作生成各有侧重，专家分工比单一模型"一锅炖"更高效
2. **参数效率**：三个 300M-3B 专家的协同效果优于单个 5B 模型，同等参数下表现更好
3. **灵活冻结**：可根据任务特点选择性冻结/微调各专家，节省计算资源
4. **层级信息流**：共享注意力确保跨专家特征在最底层即融合，避免浅层特征丢失

### 为什么选择 Flow Matching？

- 相比 Diffusion Policy，Flow Matching 训练更稳定，推理步数更少
- 支持 SDE 采样以引入动作探索噪声，适配在线 RL
- 50 步动作序列生成，实现帧率稳定的机器人控制

---

## 已知问题与状态

> 详细调试记录参见 [docs/debug_log_20260312.md](docs/debug_log_20260312.md)

| 问题 | 严重程度 | 状态 |
|------|---------|------|
| FSDP `requires_grad` 不匹配 | 严重 | 已修复 |
| 训练 OOM（micro_batch_size 过大） | 严重 | 已修复 |
| FSDP storage=0 报错 | 严重 | 已修复 |
| `ExploreNoiseNet` 未实现 | 高 | 已规避 |
| 熵损失错误清零 | 高 | 已修复 |
| `reasoning_spatial_expert` 属性访问错误 | 高 | 已修复 |
| 环境配置仅支持 libero + maniskill | 低 | 进行中 |

---

## 贡献与协议

本项目基于 [OpenPI](https://github.com/Physical-Intelligence/openpi) 二次开发，遵循原项目协议。
