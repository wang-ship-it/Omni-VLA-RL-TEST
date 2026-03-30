> 状态说明
>
> - 本文档保留为 patch 级改动索引，不再作为 Omni-VLA 适配 RLinf 的主入口文档。
> - 如果你想先了解完整背景、时间线和结论，请先读：
>   [`docs/integration/omni_vla_rlinf_integration_full_journey.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/omni_vla_rlinf_integration_full_journey.md)
> - 需要追具体代码改动时，再把本文当作补充索引使用。

# Git 代码改动文档

> 仅记录 `.py` 文件的改动，按时间从新到旧排列。

---

## 目录

1. [losses.py — GSPO log-ratio 改为 MEAN](#1-lossspy--gspo-log-ratio-改为-mean)
2. [advantages.py — GSPO 优势函数重构为 Pairwise Ranking](#2-advantagespy--gspo-优势函数重构为-pairwise-ranking)
3. [registry.py — adv_type 判断修正](#3-registrypy--adv_type-判断修正)
4. [utils.py — adv_type 判断回滚](#4-utilspy--adv_type-判断回滚)
5. [advantages.py — 添加旧版 GSPO 函数](#5-advantagespy--添加旧版-gspo-函数)
6. [omni_vla_action_model.py + metric_utils.py — value_t keepdim & 指标优化](#6-omni_vla_action_modelpy--metric_utilspy--value_t-keepdim--指标优化)
7. [omni_vla_action_model.py — 添加 attn_implementation 设置](#7-omni_vla_action_modelpy--添加-attn_implementation-设置)
8. [__init__.py — 修正 lm_head key 映射 & embed_tokens 权重共享](#8-__init__py--修正-lm_head-key-映射--embed_tokens-权重共享)
9. [__init__.py — 引入 checkpoint key remapping 机制](#9-__init__py--引入-checkpoint-key-remapping-机制)
10. [__init__.py + omni_vla_action_model.py — 权重加载验证 & DEBUG 日志](#10-__init__py--omni_vla_action_modelpy--权重加载验证--debug-日志)
11. [omni_vla.py — 去除图像重缩放](#11-omni_vlay--去除图像重缩放)
12. [omni_vla.py + vlm_with_spatial.py — attention mask dtype 修复](#12-omni_vlay--vlm_with_spatialpy--attention-mask-dtype-修复)
13. [vlm_with_spatial.py + train_omni.py — rotary_emb 路径修正](#13-vlm_with_spatialpy--train_omnipy--rotary_emb-路径修正)
14. [omni_config.py — 更新模型路径配置](#14-omni_configpy--更新模型路径配置)
15. [omni_vla.py — 传入 vlm/vggt 预训练路径](#15-omni_vlay--传入-vlmvggt-预训练路径)
16. [vlm_with_spatial.py — vggt_pretrained_path 参数化](#16-vlm_with_spatialpy--vggt_pretrained_path-参数化)
17. [vlm_with_spatial.py — vlm_pretrained_path 参数化](#17-vlm_with_spatialpy--vlm_pretrained_path-参数化)
18. [image_processing_qwen2_vl.py — 兼容新版 transformers](#18-image_processing_qwen2_vlpy--兼容新版-transformers)
19. [__init__.py — 替换 norm_stats 加载方式](#19-__init__py--替换-norm_stats-加载方式)
20. [checkpoints.py — 修复循环导入](#20-checkpointspy--修复循环导入)

---

## 1. `losses.py` — GSPO log-ratio 改为 MEAN

**Commit:** `70ad389`
**文件:** `rlinf/algorithms/losses.py`

### 改动说明
GSPO actor loss 中，`seq_log_ratio` 和 `seq_adv` 从 **SUM** 改为 **MEAN**（除以有效 token 数），符合论文公式 eq.15（`1/|A| * sum`）。同时将 `token_count` 的声明提前，避免重复计算。

### 原代码

```python
# 1️⃣ 计算 sequence-level log-ratio (SUM)
seq_log_ratio = log_diff.sum(dim=reduce_dims, keepdim=True)

# 2️⃣ 计算 sequence-level advantage (SUM)
seq_adv = adv_masked.sum(dim=reduce_dims, keepdim=True)

# 有效序列 mask
token_count = loss_mask.sum(dim=reduce_dims, keepdim=True)
seq_mask = token_count > 0
```

### 改后代码

```python
# 1️⃣ 计算 sequence-level log-ratio (MEAN, per paper eq.15: 1/|A| * sum)
token_count = loss_mask.sum(dim=reduce_dims, keepdim=True).clamp(min=1)
seq_log_ratio = log_diff.sum(dim=reduce_dims, keepdim=True) / token_count

# 2️⃣ 计算 sequence-level advantage (MEAN, advantage 已是 sequence-level 常量)
seq_adv = adv_masked.sum(dim=reduce_dims, keepdim=True) / token_count

# 有效序列 mask（token_count 已在前面声明，此处删除重复声明）
seq_mask = token_count > 0
```

---

## 2. `advantages.py` — GSPO 优势函数重构为 Pairwise Ranking

**Commit:** `aa7ffce`
**文件:** `rlinf/algorithms/advantages.py`, `rlinf/algorithms/utils.py`

### 改动说明
`compute_gspo_advantages_and_returns` 函数完全重写：从基于 GAE 的方式改为 **Pairwise Ranking**（`A_i = mean(r_i - r_j)` for all `j != i`）。同时 `utils.py` 中对 `gspo` 的 values 预处理条件同步修正。

### 原代码（advantages.py）

```python
@register_advantage("gspo")
def compute_gspo_advantages_and_returns(
    rewards: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    values: Optional[torch.Tensor] = None,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages for GSPO (Group-level Sequence Policy Optimization).
    Uses GAE for advantage estimation, paired with sequence-level GSPO loss.
    """
    return compute_gae_advantages_and_returns(
        rewards=rewards,
        gamma=gamma,
        gae_lambda=gae_lambda,
        values=values,
        normalize_advantages=normalize_advantages,
        normalize_returns=normalize_returns,
        loss_mask=loss_mask,
        dones=dones,
        **kwargs,
    )
```

### 改后代码（advantages.py）

```python
@register_advantage("gspo")
def compute_gspo_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages for GSPO (Group-level Sequence Policy Optimization).
    Uses pairwise ranking: A_i = mean(r_i - r_j) for all j != i in the group.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups * group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [seq_len, num_groups * group_size]
        group_size (int): Number of sequences per group.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (advantages, None)
    """
    grouped_rewards = rewards.view(-1, group_size)  # [num_groups, group_size]

    group_sum = grouped_rewards.sum(dim=-1, keepdim=True)   # [num_groups, 1]
    # mean of r_j for j != i: (sum_all - r_i) / (G - 1)
    pairwise_baseline = (group_sum - grouped_rewards) / (group_size - 1)

    advantages = grouped_rewards - pairwise_baseline  # r_i - mean_{j!=i}(r_j)

    # Broadcast sequence-level advantage to all tokens in each sequence
    advantages = (torch.zeros_like(loss_mask) + advantages.view(1, -1)) * loss_mask

    return advantages, None
```

### 原代码（utils.py）

```python
if kwargs["adv_type"] in ("gae", "gspo"):
    flattened_values_full = values.transpose(1, 2).reshape(...)
```

### 改后代码（utils.py）

```python
if kwargs["adv_type"] == "gae":
    flattened_values_full = values.transpose(1, 2).reshape(...)
```

---

## 3. `registry.py` — adv_type 判断修正

**Commit:** `d09983b`
**文件:** `rlinf/algorithms/registry.py`

### 改动说明
`calculate_adv_and_returns` 中，对 `gspo` 类型跳过 `calculate_scores` 的判断条件修正。

### 原代码

```python
if adv_type != "gae":
    kwargs = calculate_scores(**kwargs)
```

### 改后代码

```python
if adv_type not in ("gae", "gspo"):
    kwargs = calculate_scores(**kwargs)
```

---

## 4. `utils.py` — adv_type 判断回滚

**Commit:** `1f3efc8`
**文件:** `rlinf/algorithms/utils.py`

### 改动说明
将 commit `aa7ffce` 中对 utils.py 的修改回滚，`gspo` 重新加入 values 预处理条件。

### 原代码

```python
if kwargs["adv_type"] == "gae":
    flattened_values_full = values.transpose(1, 2).reshape(...)
```

### 改后代码

```python
if kwargs["adv_type"] in ("gae", "gspo"):
    flattened_values_full = values.transpose(1, 2).reshape(...)
```

---

## 5. `advantages.py` — 添加旧版 GSPO 函数

**Commit:** `af560bc`
**文件:** `rlinf/algorithms/advantages.py`

### 改动说明
在 `grpo_dynamic` 之后新增基于 GAE 的旧版 GSPO 优势函数（后被 commit `aa7ffce` 重构替换）。

### 原代码

```python
# （此处无 gspo 注册函数）
@register_advantage("reinpp")
def compute_reinpp_advantages(...):
    ...
```

### 改后代码（新增）

```python
@register_advantage("gspo")
def compute_gspo_advantages_and_returns(
    rewards: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    values: Optional[torch.Tensor] = None,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uses GAE for advantage estimation, paired with sequence-level GSPO loss."""
    return compute_gae_advantages_and_returns(
        rewards=rewards,
        gamma=gamma,
        gae_lambda=gae_lambda,
        values=values,
        normalize_advantages=normalize_advantages,
        normalize_returns=normalize_returns,
        loss_mask=loss_mask,
        dones=dones,
        **kwargs,
    )
```

---

## 6. `omni_vla_action_model.py` + `metric_utils.py` — value_t keepdim & 指标优化

**Commit:** `b74e178`
**文件:** `rlinf/models/embodiment/omni_vla/omni_vla_action_model.py`, `rlinf/utils/metric_utils.py`

### 改动说明
- `omni_vla_action_model.py`：`value_t.mean` 的 `keepdim` 从 `False` 改为 `True`，保持维度对齐。
- `metric_utils.py`：advantages 和 returns 的统计指标计算改为先用 `loss_mask` 过滤有效 token，再统计 mean/max/min。

### 原代码（action_model）

```python
value_t = value_t.mean(dim=-1, keepdim=False)
```

### 改后代码（action_model）

```python
value_t = value_t.mean(dim=-1, keepdim=True)
```

### 原代码（metric_utils）

```python
if "advantages" in data_buffer:
    advantages = data_buffer["advantages"]
    mean_adv = torch.mean(advantages).to(torch.cuda.current_device())
    torch.distributed.all_reduce(mean_adv, op=torch.distributed.ReduceOp.AVG)
    max_adv = torch.max(advantages).detach().item()
    min_adv = torch.min(advantages).detach().item()
    ...

if data_buffer.get("returns", None) is not None:
    returns = data_buffer["returns"]
    mean_ret = torch.mean(returns).to(torch.cuda.current_device())
    torch.distributed.all_reduce(mean_ret, op=torch.distributed.ReduceOp.AVG)
    max_ret = torch.max(returns).detach().item()
    min_ret = torch.min(returns).detach().item()
```

### 改后代码（metric_utils）

```python
loss_mask = data_buffer.get("loss_mask", None)

if "advantages" in data_buffer:
    advantages = data_buffer["advantages"]
    if loss_mask is not None:
        valid_adv = advantages[loss_mask]
        mean_adv = valid_adv.mean().to(torch.cuda.current_device())
        max_adv = valid_adv.max().detach().item()
        min_adv = valid_adv.min().detach().item()
    else:
        mean_adv = torch.mean(advantages).to(torch.cuda.current_device())
        max_adv = torch.max(advantages).detach().item()
        min_adv = torch.min(advantages).detach().item()
    torch.distributed.all_reduce(mean_adv, op=torch.distributed.ReduceOp.AVG)
    ...

if data_buffer.get("returns", None) is not None:
    returns = data_buffer["returns"]
    if loss_mask is not None:
        valid_ret = returns[loss_mask]
        mean_ret = valid_ret.mean().to(torch.cuda.current_device())
        max_ret = valid_ret.max().detach().item()
        min_ret = valid_ret.min().detach().item()
    else:
        mean_ret = torch.mean(returns).to(torch.cuda.current_device())
        max_ret = torch.max(returns).detach().item()
        min_ret = torch.min(returns).detach().item()
    torch.distributed.all_reduce(mean_ret, op=torch.distributed.ReduceOp.AVG)
```

---

## 7. `omni_vla_action_model.py` — 添加 attn_implementation 设置

**Commit:** `f2b37cb`
**文件:** `rlinf/models/embodiment/omni_vla/omni_vla_action_model.py`

### 改动说明
在 RL 推理时，补充将 `spatial_expert` 和 `action_expert` 的 `_attn_implementation` 也设为 `"eager"`，与 `reasoning_expert` 保持一致。

### 原代码

```python
self.reasoning_spatial_expert.reasoning_expert.language_model.config._attn_implementation = "eager"
```

### 改后代码

```python
self.reasoning_spatial_expert.reasoning_expert.language_model.config._attn_implementation = "eager"
self.reasoning_spatial_expert.spatial_expert.config._attn_implementation = "eager"
self.reasoning_spatial_expert.action_expert.config._attn_implementation = "eager"
```

---

## 8. `__init__.py` — 修正 lm_head key 映射 & embed_tokens 权重共享

**Commit:** `18dea42`
**文件:** `rlinf/models/embodiment/omni_vla/__init__.py`

### 改动说明
- 修正 `lm_head` checkpoint key 的映射路径（PaliGemma 的 `lm_head` 在顶层，不在 `.model` 下）。
- 新增处理 PaliGemma weight-tied 情况：如果 `lm_head.weight` 存在但 `embed_tokens.weight` 缺失，自动复制。

### 原代码

```python
# checkpoint: language_model.lm_head.X -> model: model.language_model.lm_head.X
if suffix.startswith("language_model.lm_head."):
    return prefix + "model." + suffix
```

### 改后代码

```python
# checkpoint: language_model.lm_head.X -> model: lm_head.X
# (PaliGemmaForConditionalGeneration has lm_head at top level, not under .model)
if suffix.startswith("language_model.lm_head."):
    return prefix + suffix[len("language_model."):]
```

### 新增代码（embed_tokens 权重共享处理）

```python
# Handle weight-tied embed_tokens: PaliGemma ties embed_tokens with lm_head
lm_head_key = "reasoning_spatial_expert.reasoning_expert.lm_head.weight"
embed_tokens_key = "reasoning_spatial_expert.reasoning_expert.model.language_model.embed_tokens.weight"
if lm_head_key in remapped_state_dict and embed_tokens_key not in remapped_state_dict:
    remapped_state_dict[embed_tokens_key] = remapped_state_dict[lm_head_key]
```

---

## 9. `__init__.py` — 引入 checkpoint key remapping 机制

**Commit:** `dcfc0eb`
**文件:** `rlinf/models/embodiment/omni_vla/__init__.py`

### 改动说明
将原来直接调用 `safetensors.torch.load_model` 的方式替换为手动 key remapping 机制，以适配 SFT checkpoint 和模型 state_dict 的层级路径差异。

### 原代码

```python
# Load weights and verify
model_keys = set(model.state_dict().keys())
loaded_keys = set()
for weight_path in weight_paths:
    import safetensors as _sf
    ckpt_keys = set(_sf.safe_open(weight_path, framework="pt").keys())
    loaded_keys.update(ckpt_keys)
    safetensors.torch.load_model(model, weight_path, strict=False)

missing_in_ckpt = model_keys - loaded_keys
unexpected_in_ckpt = loaded_keys - model_keys
if missing_in_ckpt:
    logger.warning(f"[OmniVLA] {len(missing_in_ckpt)} model keys NOT in checkpoint: ...")
if unexpected_in_ckpt:
    logger.warning(f"[OmniVLA] {len(unexpected_in_ckpt)} checkpoint keys NOT in model: ...")
```

### 改后代码

```python
# Load weights with key remapping
# The SFT checkpoint uses a different key hierarchy for reasoning_expert:
#   checkpoint: reasoning_expert.language_model.model.layers.X...
#   model:      reasoning_expert.model.language_model.layers.X...
model_keys = set(model.state_dict().keys())

def _remap_ckpt_key(key: str) -> str:
    """Remap checkpoint keys to match model's state_dict hierarchy."""
    prefix = "reasoning_spatial_expert.reasoning_expert."
    if key.startswith(prefix):
        suffix = key[len(prefix):]
        if suffix.startswith("language_model.model."):
            return prefix + "model.language_model." + suffix[len("language_model.model."):]
        if suffix.startswith("language_model.lm_head."):
            return prefix + "model." + suffix
        if suffix.startswith("vision_tower."):
            return prefix + "model." + suffix
        if suffix.startswith("multi_modal_projector."):
            return prefix + "model." + suffix
    return key

for weight_path in weight_paths:
    import safetensors as _sf
    f = _sf.safe_open(weight_path, framework="pt")
    ckpt_keys = list(f.keys())

    remapped_state_dict = {}
    for ckpt_key in ckpt_keys:
        model_key = _remap_ckpt_key(ckpt_key)
        remapped_state_dict[model_key] = f.get_tensor(ckpt_key)

    missing, unexpected = model.load_state_dict(remapped_state_dict, strict=False)
    if missing:
        logger.warning(f"[OmniVLA] {len(missing)} keys missing after remapped load: ...")
    if unexpected:
        logger.warning(f"[OmniVLA] {len(unexpected)} unexpected keys after remapped load: ...")
```

---

## 10. `__init__.py` + `omni_vla_action_model.py` — 权重加载验证 & DEBUG 日志

**Commit:** `45e8ff2`
**文件:** `rlinf/models/embodiment/omni_vla/__init__.py`, `omni_vla_action_model.py`, `huggingface_worker.py`

### 改动说明
- `__init__.py`：原来直接 `load_model`，改为先统计 model_keys 和 loaded_keys 并打印 warning。
- `omni_vla_action_model.py`：rollout 时对 `raw_actions` 和 `final_actions` 添加前 3 步的 DEBUG 统计日志。
- `huggingface_worker.py`：将 `OMNI_VLA` 加入不需要 action chunk 处理的模型类型列表。

### 原代码（action_model rollout）

```python
actions = self.output_transform(
    {"actions": outputs["actions"], "state": observation.state}
)["actions"].numpy()
```

### 改后代码（action_model rollout）

```python
raw_actions = outputs["actions"]
if self.global_step < 3:
    self.logger.info(
        f"[OmniVLA DEBUG] raw_actions stats: "
        f"mean={raw_actions.mean().item():.4f}, std={raw_actions.std().item():.4f}, "
        f"min={raw_actions.min().item():.4f}, max={raw_actions.max().item():.4f}, "
        f"shape={tuple(raw_actions.shape)}"
    )
actions = self.output_transform(
    {"actions": raw_actions, "state": observation.state}
)["actions"].numpy()
if self.global_step < 3:
    self.logger.info(
        f"[OmniVLA DEBUG] final_actions stats: "
        f"mean={actions.mean():.4f}, std={actions.std():.4f}, "
        f"min={actions.min():.4f}, max={actions.max():.4f}, "
        f"shape={actions.shape}"
    )
```

### 原代码（huggingface_worker）

```python
if SupportedModel(self.cfg.actor.model.model_type) in [
    SupportedModel.OPENPI,
    SupportedModel.MLP_POLICY,
    ...
```

### 改后代码（huggingface_worker）

```python
if SupportedModel(self.cfg.actor.model.model_type) in [
    SupportedModel.OPENPI,
    SupportedModel.OMNI_VLA,
    SupportedModel.MLP_POLICY,
    ...
```

---

## 11. `omni_vla.py` — 去除图像重缩放

**Commit:** `001ca1e`
**文件:** `omni_vla/src/openpi/models_pytorch/omni_vla.py`

### 改动说明
发现图像已在 `Observation.from_dict` 阶段归一化到 `[-1, 1]`，此处的 `* 2 - 1` 为重复操作，予以删除。

### 原代码

```python
images_flat = images_flat * 2 - 1  # [0,1] -> [-1,1]
```

### 改后代码

```python
# images are already in [-1, 1] from Observation.from_dict, no need to rescale
```

---

## 12. `omni_vla.py` + `vlm_with_spatial.py` — attention mask dtype 修复

**Commit:** `9e36371`
**文件:** `omni_vla/src/openpi/models_pytorch/omni_vla.py`, `vlm_with_spatial.py`

### 改动说明
- `omni_vla.py`：`_prepare_attention_masks_4d` 新增 `dtype` 参数，支持返回时转换数据类型。
- `vlm_with_spatial.py`：`spatial_expert` 和 `action_expert` 调用时，先获取各自参数的 dtype，再将 attention_mask 转换为对应 dtype 后传入，避免类型不匹配。

### 原代码（omni_vla.py）

```python
def _prepare_attention_masks_4d(self, att_2d_masks):
    """Helper method to prepare 4D attention masks for transformer."""
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)
```

### 改后代码（omni_vla.py）

```python
def _prepare_attention_masks_4d(self, att_2d_masks, dtype=None):
    """Helper method to prepare 4D attention masks for transformer."""
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    mask = torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)
    if dtype is not None:
        mask = mask.to(dtype=dtype)
    return mask
```

### 原代码（vlm_with_spatial.py）

```python
# spatial_expert 调用
middle_output = self.spatial_expert.forward(
    inputs_embeds=inputs_embeds[1],
    attention_mask=attention_mask,
    ...
)

# action_expert 调用
suffix_output = self.action_expert.forward(
    inputs_embeds=inputs_embeds[2],
    attention_mask=attention_mask,
    ...
)
```

### 改后代码（vlm_with_spatial.py）

```python
# spatial_expert 调用
spatial_dtype = next(self.spatial_expert.parameters()).dtype
spatial_attention_mask = attention_mask.to(dtype=spatial_dtype) if attention_mask is not None else None
middle_output = self.spatial_expert.forward(
    inputs_embeds=inputs_embeds[1],
    attention_mask=spatial_attention_mask,
    ...
)

# action_expert 调用
action_dtype = next(self.action_expert.parameters()).dtype
action_attention_mask = attention_mask.to(dtype=action_dtype) if attention_mask is not None else None
suffix_output = self.action_expert.forward(
    inputs_embeds=inputs_embeds[2],
    attention_mask=action_attention_mask,
    ...
)
```

---

## 13. `vlm_with_spatial.py` + `train_omni.py` — rotary_emb 路径修正

**Commit:** `bf5e1ea`
**文件:** `omni_vla/src/openpi/models_pytorch/vlm_with_spatial.py`, `omni_vla/scripts/train_omni.py`, `rlinf/models/embodiment/omni_vla/omni_vla_action_model.py`

### 改动说明
- `vlm_with_spatial.py`：`reasoning_expert.language_model` 对应 `GemmaModel`（非 `GemmaForCausalLM`），移除多余的 `.model.` 层级访问。同时删除调试用的 key 打印代码，修正 `_remap_paligemma_key` 中的路径映射。
- `train_omni.py`：注释掉训练开始时打印 trainable parameters 的代码。
- `omni_vla_action_model.py`：同步修正 `language_model.model.layers` → `language_model.layers`。

### 原代码（vlm_with_spatial.py — rotary_emb）

```python
cos, sin = reasoning_expert.language_model.model.rotary_emb(query_states, position_ids)
scaling = reasoning_expert.language_model.model.layers[layer_idx].self_attn.scaling
att_output, _ = modeling_gemma.eager_attention_forward(
    reasoning_expert.language_model.model.layers[layer_idx].self_attn, ...
)
head_dim = reasoning_expert.language_model.model.layers[layer_idx].self_attn.head_dim
num_attention_heads = reasoning_expert.language_model.model.layers[layer_idx].self_attn.config.num_attention_heads
```

### 改后代码（vlm_with_spatial.py — rotary_emb）

```python
cos, sin = reasoning_expert.language_model.rotary_emb(query_states, position_ids)
scaling = reasoning_expert.language_model.layers[layer_idx].self_attn.scaling
att_output, _ = modeling_gemma.eager_attention_forward(
    reasoning_expert.language_model.layers[layer_idx].self_attn, ...
)
head_dim = reasoning_expert.language_model.layers[layer_idx].self_attn.head_dim
num_attention_heads = reasoning_expert.language_model.layers[layer_idx].self_attn.config.num_attention_heads
```

### 原代码（vlm_with_spatial.py — paligemma key remap）

```python
if key.startswith("model.language_model."):
    # model.language_model.layers.X.xxx -> language_model.model.layers.X.xxx
    return "language_model.model." + key[len("model.language_model."):]
```

### 改后代码（vlm_with_spatial.py — paligemma key remap）

```python
if key.startswith("model.language_model."):
    # model.language_model.layers.X.xxx -> language_model.layers.X.xxx
    # (AutoModel returns GemmaModel directly, no extra .model level)
    return "language_model." + key[len("model.language_model."):]
```

### 原代码（omni_vla_action_model.py）

```python
if (
    self.reasoning_spatial_expert.reasoning_expert.language_model.model.layers[0].self_attn.q_proj.weight.dtype
    == torch.bfloat16
```

### 改后代码（omni_vla_action_model.py）

```python
if (
    self.reasoning_spatial_expert.reasoning_expert.language_model.layers[0].self_attn.q_proj.weight.dtype
    == torch.bfloat16
```

---

## 14. `omni_config.py` — 更新模型路径配置

**Commit:** `6a23e6d`
**文件:** `omni_vla/src/openpi/models_pytorch/omni_config.py`

### 改动说明
更新默认模型路径为新的服务器路径，清空 G2VLM 相关路径。

### 原代码

```python
pretrained_g2vlm_path: str = '/data/openpi_temp/checkpoints/pi0_libero_low_mem_finetune/omni_9/30000'
g2vlm_config_path: str = "/home/user/robot/model/G2VLM-2B-MoT"

vlm_pretrained_path: Optional[str] = "/root/autodl-tmp/huggingface/lerobot/pi0_torch_libero/pi0_torch_libero/model.safetensors"
vggt_pretrained_path: Optional[str] = "/root/autodl-tmp/huggingface/lerobot/VGGT-1B/model.safetensors"
```

### 改后代码

```python
pretrained_g2vlm_path: str = ''
g2vlm_config_path: str = ""

vlm_pretrained_path: Optional[str] = "/root/data/models/lerobot/model.safetensors"
vggt_pretrained_path: Optional[str] = "/root/data/models/VGGT-1B/model.safetensors"
```

---

## 15. `omni_vla.py` — 传入 vlm/vggt 预训练路径

**Commit:** `f133543`
**文件:** `omni_vla/src/openpi/models_pytorch/omni_vla.py`

### 改动说明
`OmniVLA` 初始化时将 `config.vlm_pretrained_path` 和 `config.vggt_pretrained_path` 显式传入 `VLMWithSpatialActionExpertModel`。

### 原代码

```python
self.reasoning_spatial_expert = VLMWithSpatialActionExpertModel(
    paligemma_config,
    spatial_config,
    action_expert_config,
    precision=config.dtype,
)
```

### 改后代码

```python
self.reasoning_spatial_expert = VLMWithSpatialActionExpertModel(
    paligemma_config,
    spatial_config,
    action_expert_config,
    vlm_pretrained_path=config.vlm_pretrained_path,
    vggt_pretrained_path=config.vggt_pretrained_path,
    precision=config.dtype,
)
```

---

## 16. `vlm_with_spatial.py` — vggt_pretrained_path 参数化

**Commit:** `ca62644`
**文件:** `omni_vla/src/openpi/models_pytorch/vlm_with_spatial.py`

### 改动说明
将 VGGT 权重加载路径从硬编码常量 `VGGT_PRETRAINED_PATH` 改为函数参数 `vggt_pretrained_path`。

### 原代码

```python
state_dict_vggt = load_file(VGGT_PRETRAINED_PATH, device="cpu")
```

### 改后代码

```python
state_dict_vggt = load_file(vggt_pretrained_path, device="cpu")
```

---

## 17. `vlm_with_spatial.py` — vlm_pretrained_path 参数化

**Commit:** `ca62644`（同上 commit 早期版本，实为独立改动）
**文件:** `omni_vla/src/openpi/models_pytorch/vlm_with_spatial.py`

### 改动说明
将 VLM 权重加载路径从硬编码常量 `MODEL_PATH` 改为函数参数 `vlm_pretrained_path`。

### 原代码

```python
state_dict = load_file(MODEL_PATH, device="cpu")
```

### 改后代码

```python
state_dict = load_file(vlm_pretrained_path, device="cpu")
```

---

## 18. `image_processing_qwen2_vl.py` — 兼容新版 transformers

**Commit:** `c8a409c`
**文件:** `omni_vla/src/openpi/vlm_expert/qwen2vl/image_processing_qwen2_vl.py`

### 改动说明
新版 `transformers` 移除了 `VideoInput` 和 `make_batched_videos`，通过本地定义兼容 shim 解决导入错误。

### 原代码

```python
from transformers.image_utils import (
    ...
    VideoInput,
    ...
    make_batched_videos,
    ...
)
```

### 改后代码

```python
from transformers.image_utils import (
    ...
    # VideoInput 和 make_batched_videos 已从新版 transformers 移除
)

# VideoInput was removed in newer transformers versions; define as type alias
VideoInput = list

def make_batched_videos(videos):
    """Compat shim: ensure videos is a list of video (each video is a list of frames)."""
    if videos is None:
        return videos
    if isinstance(videos, (list, tuple)) and len(videos) > 0 and isinstance(videos[0], (list, tuple)):
        return list(videos)
    return [videos]
```

---

## 19. `__init__.py` — 替换 norm_stats 加载方式

**Commit:** `866ffb9`
**文件:** `rlinf/models/embodiment/omni_vla/__init__.py`

### 改动说明
将 `_checkpoints.load_norm_stats` 替换为直接使用 `_normalize.load`，并手动拼接路径，去掉对 `checkpoints` 模块的依赖。

### 原代码

```python
from openpi.training import checkpoints as _checkpoints
...
norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
```

### 改后代码

```python
import openpi.shared.normalize as _normalize
...
norm_stats_dir = os.path.join(checkpoint_dir, data_config.asset_id)
norm_stats = _normalize.load(norm_stats_dir)
```

---

## 20. `checkpoints.py` — 修复循环导入

**Commit:** `4c6fcd0`
**文件:** `omni_vla/src/openpi/training/checkpoints.py`

### 改动说明
将 `data_loader` 模块从直接导入改为 `TYPE_CHECKING` 条件导入，避免循环依赖问题。

### 原代码

```python
from typing import Protocol
...
import openpi.training.data_loader as _data_loader
```

### 改后代码

```python
from typing import TYPE_CHECKING, Protocol
...
if TYPE_CHECKING:
    import openpi.training.data_loader as _data_loader
```

---

*文档生成时间: 2026-03-16*
