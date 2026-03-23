# OmniVLA 激活显存改动记录（2026-03-23）

## 目标

- 解决训练阶段 prefix / middle 仍持有 activation 图的问题。
- 清理 KV cache 相关的隐藏显存放大点（复制与图关联）。
- 让 gradient checkpointing 与 `use_cache` 行为一致，避免“训练里用推理优化”。

---

## 改动 1：移除训练路径中的 KV cache copy

### 文件

- `rlinf/models/embodiment/omni_vla/omni_vla_action_model.py`

### 改前

```python
if past_key_values is not None:
    cached_seq_len = past_key_values.get_seq_length()
    from transformers.cache_utils import DynamicCache
    past_key_values_copy = DynamicCache()
    past_key_values_copy.key_cache = list(past_key_values.key_cache)
    past_key_values_copy.value_cache = list(past_key_values.value_cache)
    if hasattr(past_key_values, '_seen_tokens'):
        past_key_values_copy._seen_tokens = past_key_values._seen_tokens
else:
    cached_seq_len = 0
    past_key_values_copy = None

outputs_embeds, _ = self.reasoning_spatial_expert.forward(
    ...,
    past_key_values=past_key_values_copy,
    ...,
)
```

### 改后

```python
if past_key_values is not None:
    cached_seq_len = past_key_values.get_seq_length()
else:
    cached_seq_len = 0

outputs_embeds, _ = self.reasoning_spatial_expert.forward(
    ...,
    past_key_values=past_key_values,
    ...,
)
```

### 为什么改

- `DynamicCache` 的 key/value 列表复制会额外占用显存，且容易引入隐藏引用链。
- 在当前 suffix 前向 `use_cache=False` 的路径下，复制 cache 没有必要。

### 作用

- 降低 KV cache 带来的瞬时显存峰值。
- 减少无意义的 cache 对象分配和引用复杂度。

---

## 改动 2：训练时 prefix / middle 强制 no_grad

### 文件

- `rlinf/models/embodiment/omni_vla/omni_vla_action_model.py`

### 改前

```python
prefix_middle_no_grad = self.training and self.config.train_expert_only

prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(...)
...
_, past_key_values = self.reasoning_spatial_expert.forward(..., use_cache=True)

middle_embs, middle_pad_masks, middle_att_masks = self.embed_spatial(...)
...
(_, _, _), past_key_values = self.reasoning_spatial_expert.forward(..., use_cache=True)
```

### 改后

```python
prefix_middle_no_grad = self.training

if prefix_middle_no_grad:
    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(...)
else:
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(...)

if prefix_middle_no_grad:
    with torch.no_grad():
        _, past_key_values = self.reasoning_spatial_expert.forward(..., use_cache=True)
else:
    _, past_key_values = self.reasoning_spatial_expert.forward(..., use_cache=True)

if prefix_middle_no_grad:
    with torch.no_grad():
        middle_embs, middle_pad_masks, middle_att_masks = self.embed_spatial(...)
else:
    middle_embs, middle_pad_masks, middle_att_masks = self.embed_spatial(...)

if prefix_middle_no_grad:
    with torch.no_grad():
        (_, _, _), past_key_values = self.reasoning_spatial_expert.forward(..., use_cache=True)
    if past_key_values is not None:
        past_key_values.key_cache = [k.detach() for k in past_key_values.key_cache]
        past_key_values.value_cache = [v.detach() for v in past_key_values.value_cache]
else:
    (_, _, _), past_key_values = self.reasoning_spatial_expert.forward(..., use_cache=True)
```

### 为什么改

- prefix / middle 不应参与当前训练目标的梯度回传，却仍会构图并保留 activation。
- 仅在 `train_expert_only` 下 no_grad 不够稳妥，训练模式下应统一切断这两段图。

### 作用

- 显著减少 prefix / middle 激活图占用。
- 降低 OOM 风险，提升长序列和多步 diffusion 场景稳定性。
- `detach` cache 进一步避免图通过 KV 引用“挂回去”。

---

## 改动 3：gradient checkpointing 与 use_cache 联动

### 文件

- `omni_vla/src/openpi/models_pytorch/omni_vla.py`

### 改前

```python
def gradient_checkpointing_enable(self):
    self.gradient_checkpointing_enabled = True
    self.reasoning_spatial_expert.reasoning_expert.language_model.gradient_checkpointing = True
    self.reasoning_spatial_expert.reasoning_expert.vision_tower.gradient_checkpointing = True
    self.reasoning_spatial_expert.spatial_expert.model.gradient_checkpointing = True
    self.reasoning_spatial_expert.action_expert.model.gradient_checkpointing = True
```

### 改后

```python
def gradient_checkpointing_enable(self):
    self.gradient_checkpointing_enabled = True
    self.reasoning_spatial_expert.reasoning_expert.language_model.gradient_checkpointing = True
    self.reasoning_spatial_expert.reasoning_expert.vision_tower.gradient_checkpointing = True
    self.reasoning_spatial_expert.spatial_expert.model.gradient_checkpointing = True
    self.reasoning_spatial_expert.action_expert.model.gradient_checkpointing = True
    if hasattr(self.reasoning_spatial_expert.reasoning_expert.language_model, "config"):
        self.reasoning_spatial_expert.reasoning_expert.language_model.config.use_cache = False
    if hasattr(self.reasoning_spatial_expert.spatial_expert.model, "config"):
        self.reasoning_spatial_expert.spatial_expert.model.config.use_cache = False
    if hasattr(self.reasoning_spatial_expert.action_expert.model, "config"):
        self.reasoning_spatial_expert.action_expert.model.config.use_cache = False
```

并在 `gradient_checkpointing_disable` 中对称恢复 `use_cache = True`。

### 为什么改

- checkpointing 与 cache 混用容易导致训练期显存策略不一致。
- 训练打开 checkpointing 时应默认关闭 cache，避免推理态缓存策略残留。

### 作用

- 减少训练阶段 cache 行为导致的额外内存占用。
- 使“训练模式”与“推理优化”策略分离更清晰。

---

## 验证结果

- 语言诊断：通过（无新增错误）。
- 语法编译：通过。
  - `python -m py_compile rlinf/models/embodiment/omni_vla/omni_vla_action_model.py`
  - `python -m py_compile omni_vla/src/openpi/models_pytorch/omni_vla.py`

---

## 影响范围与注意事项

- 本次改动聚焦训练路径显存，不改变推理路径输出格式。
- 训练速度可能略受影响（no_grad 与 cache策略变化会改变部分计算路径），但显存稳定性会更好。
- 若后续仍出现激活峰值，下一步建议将训练路径改为“完全无 KV cache”版本（代价是进一步降速）。
