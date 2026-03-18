# GSPO Importance Ratio 归一化修复

## 问题背景

GSPO 训练持续崩溃：`success_once` 从 0.8 降到 0.2，`ratio` 系统性偏低（0.89-0.93）。
经论文公式逐一对比代码，发现 GSPO loss 中论文 Eq.14 的 `1/|A|` 归一化被架空。

---

## 改动总结

共修改 **2 个文件**，只影响 **GSPO 算法**，不影响 PPO/GRPO。

### 改动 1: `rlinf/algorithms/losses.py` — GSPO loss 中补回 `1/|A|` 归一化

**修改前：**
```python
# 1️⃣ 计算 sequence-level log-ratio
log_diff = torch.where(loss_mask, logprobs - old_logprobs, torch.zeros_like(logprobs))

token_count = loss_mask.sum(dim=reduce_dims, keepdim=True).clamp(min=1)
seq_log_ratio = log_diff.sum(dim=reduce_dims, keepdim=True) / token_count
```

**修改后：**
```python
# 1️⃣ 计算 sequence-level log-ratio (MEAN, per paper Eq.14: 1/|A| * sum)
log_diff = torch.where(loss_mask, logprobs - old_logprobs, torch.zeros_like(logprobs))

token_count = loss_mask.sum(dim=reduce_dims, keepdim=True).clamp(min=1)

# When chunk_level preprocessing already summed logprobs to [bsz] scalar,
# reduce_dims is empty and token_count=1, bypassing the 1/|A| normalization.
# Apply Eq.14 normalization using single_action_dim info from kwargs.
if logprobs.ndim == 1 and "single_action_dim" in kwargs:
    single_action_dim = kwargs["single_action_dim"]
    num_action_chunks = kwargs.get("num_action_chunks", 1)
    action_seq_len = num_action_chunks * single_action_dim
    token_count = torch.tensor(action_seq_len, dtype=log_diff.dtype, device=log_diff.device)

seq_log_ratio = log_diff.sum(dim=reduce_dims, keepdim=True) / token_count
```

### 改动 2: `rlinf/workers/actor/fsdp_actor_worker.py` — 传入 `num_action_chunks`

**修改前：**
```python
kwargs = {
    "loss_type": self.cfg.algorithm.loss_type,
    "logprob_type": self.cfg.algorithm.logprob_type,
    "reward_type": self.cfg.algorithm.reward_type,
    "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
    "logprobs": output_dict["logprobs"],
    ...
}
```

**修改后：**
```python
kwargs = {
    "loss_type": self.cfg.algorithm.loss_type,
    "logprob_type": self.cfg.algorithm.logprob_type,
    "reward_type": self.cfg.algorithm.reward_type,
    "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
    "num_action_chunks": self.cfg.actor.model.get("num_action_chunks", 1),  # 新增
    "logprobs": output_dict["logprobs"],
    ...
}
```

### 未修改: `rlinf/algorithms/utils.py` — 预处理保持原样

```python
# 保持不变，chunk_level 仍然用 sum（对 PPO/GRPO 是正确的概率比）
elif logprob_type == "chunk_level":
    logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2])
    old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2])
```

---

## 问题根因分析

### 数据流追踪

```
logprobs 原始 shape: [bsz, num_action_chunks, action_dim] = [bsz, 5, 7]
                                    │
                    preprocess_loss_inputs (utils.py)
                    chunk_level: .sum(dim=[1,2])
                                    │
                    logprobs shape: [bsz]  (35个值被sum成标量)
                                    │
                    compute_gspo_actor_loss_fn (losses.py)
                    reduce_dims = tuple(range(1, 1)) = ()  ← 空！
                    token_count = 1                         ← 归一化失效！
                                    │
                    seq_log_ratio = sum_of_35_values / 1    ← 应该 / 35
                    ratio = exp(seq_log_ratio)              ← 放大了35倍
```

### 论文 vs 代码对比

**论文 Eq.14:**
```
s_{i,t}(θ) = exp( 1/|A_{i,t}| × Σ_{τ=0}^{K-1} log [p_θ / p_{θ_old}] )
```
其中 `|A_{i,t}|` = action_chunk × action_dim = 5 × 7 = 35

**修改前代码（等效）：**
```
ratio = exp( Σ log_ratio )           # 缺少 1/|A| 归一化
```

**修改后代码（等效）：**
```
ratio = exp( 1/35 × Σ log_ratio )    # 正确的 1/|A| 归一化
```

### 数值影响

假设每个 token 的 log ratio = -0.01（策略微小变化）：

| 指标 | 修改前 (÷1) | 修改后 (÷35) |
|------|------------|-------------|
| seq_log_ratio | -0.01 × 35 = **-0.35** | -0.01 × 35 / 35 = **-0.01** |
| ratio = exp(seq_log_ratio) | exp(-0.35) = **0.70** | exp(-0.01) = **0.99** |
| clip 触发? (ε=0.2) | 0.70 < 0.8 → **是** | 0.99 ∈ [0.8,1.2] → **否** |

**修改前**：ratio 被系统性压低 → clip 过早触发 → 策略持续降低动作概率 → 崩溃

**修改后**：ratio 正确反映 per-element 平均变化 → clip 合理工作 → 策略稳定

---

## 影响范围

### 为什么只改 GSPO loss，不改预处理？

| 算法 | importance ratio 定义 | 预处理 sum 是否正确 |
|------|---------------------|-------------------|
| **PPO** | `r = π_new/π_old = exp(Σ log ratio)` — 标准概率比 | ✅ sum 正确 |
| **GRPO** | 同 PPO | ✅ sum 正确 |
| **GSPO** | `s = exp(1/\|A\| × Σ log ratio)` — 论文 Eq.14 归一化 | ❌ 需要额外 ÷\|A\| |

PPO/GRPO 的 importance ratio 就是标准的概率之比 `π_new/π_old`，用 `exp(sum)` 是数学上正确的。
只有 GSPO 论文特别设计了 `1/|A|` 归一化来防止高维动作空间中 ratio 偏离过大。

### 受影响的代码路径

```
只有 loss_type="gspo" 时才进入 compute_gspo_actor_loss_fn
    → 只有这个函数中的 if logprobs.ndim == 1 分支被触发
    → PPO (loss_type="actor_critic") 和 GRPO (loss_type="grpo") 完全不受影响
```

---

## 验证方法

跑 GSPO 训练 10 步，对比修改前后：

| 指标 | 修改前（预期） | 修改后（预期） |
|------|-------------|-------------|
| `train/actor/ratio` | 0.89 - 0.93 | **0.95 - 1.05** |
| `train/actor/seq_log_ratio_mean` | -0.18 ~ -0.24 | **-0.05 ~ 0.05** |
| `env/success_once` | 从 0.8 降到 0.2（崩溃） | **稳定在 0.7+** |

---

## 超参数建议

修复后 GSPO 的 ratio scale 恢复正常，之前为了补偿 bug 调整的超参可以恢复：

| 参数 | 之前为补偿 bug 的值 | 修复后建议值 |
|------|------------------|------------|
| `update_epoch` | 1 | **2-4**（可以恢复，ratio 稳定后多轮更新安全） |
| `clip_ratio_high/low` | 0.1 | **0.2**（恢复默认） |
| `lr` | 5e-6 | **1e-5**（恢复默认） |
| `kl_beta` | 0.01 | **0.01**（保持，KL scale 也变小了） |
