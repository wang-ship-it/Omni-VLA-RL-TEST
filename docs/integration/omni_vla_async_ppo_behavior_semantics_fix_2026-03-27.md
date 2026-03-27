# Omni-VLA Async PPO 行为语义错位定位与修复（2026-03-27）

## 结论摘要

本次问题的核心不是：

- backlog / replay 积压
- rollout 权重滞后
- 少数高 `behav_weight` 尾部样本
- actor 更新步长过大

而是：

**Omni-VLA 在 async PPO 中，rollout / old-policy logprob 使用的行为语义，与 actor 当前训练前向使用的训练语义不一致。**

这会导致：

- `train/actor/behav_approx_kl` 长期虚高
- `train/actor/prox_old_logprob_gap_abs_mean` 长期虚高
- `current logprobs` 与 `old/proximal logprobs` 不在同一口径

本次修复后，关键指标已经明显收敛：

- `train/actor/behav_approx_kl`：从长期 `30-50+` 降到约 `0.000-0.003`
- `train/actor/prox_old_logprob_gap_abs_mean`：从 `40+` 降到约 `0.036-0.040`
- `train/actor/debug_train_rollout_logprob_gap_abs_mean`：降到 `0`
- `train/actor/proximal_ratio`：回到 `1.0` 附近

---

## 现象

问题最初表现为：

- `behav_approx_kl` 长期偏高
- `proximal_*` 指标却比较正常
- rollout 版本同步指标没有异常
- 环境指标未必立刻崩，但策略更新显著受扰

典型现象是：

- `behav_approx_kl` 长期在 `35-55`
- `proximal_approx_kl` 却接近 `0`
- `proximal_ratio` 接近 `1`
- `current_version_minus_batch_version_max` 没有显示严重 backlog

这说明问题更像“语义错位”，而不是 PPO 更新过猛。

---

## 排查路径

### 1. 排除版本滞后与 backlog

从 rollout / train 指标中确认：

- `rollout/applied_minus_runner_version = 0`
- `rollout/weight_sync_apply_total` 正常
- `train/actor/dropped_rollout_batches = 0`
- `train/actor/current_version_minus_batch_version_max` 不支持严重积压解释

结论：

- 不是 rollout 权重同步问题
- 不是 replay backlog 主导

### 2. 排除高 `behav_weight` 尾部样本主导

观察：

- `behav_weight_mean / p95 / gt_1_fraction / gt_2_fraction`

结论：

- 有局部尖峰，但不足以解释持续高位的 `behav_approx_kl`
- 更像系统性 logprob 口径错位

### 3. 引入整链路文本日志

为了避免只看聚合指标，增加了三段链路日志：

1. rollout 当场生成 `old_logprobs`
2. actor 侧 proximal 重算 `proximal_logprobs`
3. actor 训练真正消费的 `current logprobs`

并通过 `trace_id` 串联同一样本。

相关实现：

- `rlinf/utils/chain_trace.py`
- `rlinf/workers/rollout/hf/huggingface_worker.py`
- `rlinf/workers/actor/async_ppo_fsdp_worker.py`

### 4. rollout 当场 local recompute 验证

关键验证结果：

- `rollout_old_vs_local_gap_abs_mean ≈ 0.012-0.014`

结论：

- rollout 侧生成的 `old_logprobs` 是自洽的
- 不是 rollout 保存错了，也不是传输 / reshape / sample pairing 错了

### 5. proximal 改为 eval 语义验证

将：

- `proximal_recompute_model_mode: eval`

后，观察到：

- `prox_prev_gap ≈ 0.004-0.005`
- `behav_approx_kl ≈ 0.004`
- `prox_old_gap_abs_mean ≈ 0.04`

结论：

- `old_logprobs` 与 rollout/eval 语义对齐
- 问题不在 old-policy 数据链路
- 问题在 current train forward 语义

---

## 根因

根因是 Omni-VLA 将多种语义同时绑在 `self.training` 上：

- 行为 policy 语义
- dropout / 自定义前向分支语义
- gradient checkpointing / cache / 显存优化语义

在 async PPO 中，这会导致：

- rollout 用一套更接近行为 policy 的前向
- actor 当前训练更新时用另一套训练优化语义前向

结果是：

- `old_logprobs` 和 `proximal(eval)` 一致
- `current logprobs(train)` 与它们系统性偏离

### 关键底层嫌疑

定位过程中发现，底层模型在 `self.training` 条件下会影响：

- `gradient_checkpointing`
- `use_cache`
- cache 的实际传递方式
- 某些 attention / mask / expert 路径

其中较关键的一条是：

- 训练时 checkpointing 与 cache 组合会把上层“想走 rollout 语义”的前向改写成另一条路径

---

## 修复策略

### 原则

不要再粗暴地把“是否训练”直接等同于“是否使用行为语义”。

应该拆开两件事：

1. **行为语义**
   - 用于 rollout / old-policy / behavior logprob 对齐
2. **显存优化语义**
   - 用于训练期 checkpointing、是否保留大 cache、是否走更省显存路径

### 修复实现

在 Omni-VLA actor 路径中引入了更细粒度的语义控制：

- `prefix_middle_no_grad_override`
- `clone_past_key_values_override`
- `gradient_checkpointing_override`
- `behavior_eval_override`
- `behavior_eval_include_vision`

并加入轻量行为对齐模式：

- `actor_train_model_mode: behavior_align`

其逻辑是：

- 对语言 / spatial / action 模块临时使用行为语义
- 不强制整条 vision 路径切到 full eval
- 尽量保留训练可承受的显存形态

对应实现位置：

- `rlinf/models/embodiment/omni_vla/omni_vla_action_model.py`
- `rlinf/workers/actor/async_ppo_fsdp_worker.py`

### 为什么不是直接用 full eval

曾尝试：

- `actor_train_model_mode: eval`

它虽然也能对齐语义，但会导致训练主路径显存大幅上涨，实测在 vision / SigLIP 路径上容易 OOM。

因此最终采用：

- `behavior_align`（历史实验名 `eval_light`）

它在显存和语义一致性之间取得了更好的平衡。

---

## 当前推荐配置

当前稳定基线建议为：

```yaml
algorithm:
  proximal_recompute_model_mode: eval
  actor_train_model_mode: behavior_align
  debug_train_rollout_semantics_gap: False
  debug_chain_trace: False
```

说明：

- `proximal_recompute_model_mode: eval`
  - 保证 proximal / old-policy logprob 口径与 rollout 对齐
- `actor_train_model_mode: behavior_align`
  - 保证 current train forward 尽量贴近行为 policy 语义
- 两个 debug 开关默认关闭
  - 只在继续诊断时打开，避免额外开销

---

## 关键证据链

### 修复前

典型结果：

- `behav_approx_kl ≈ 35-55`
- `prox_old_logprob_gap_abs_mean ≈ 40+`
- `pre_loss_logprob_old_gap ≈ 1.0-1.4`
- `pre_loss_logprob_prox_gap ≈ 0`

解释：

- `current` 与 `proximal(train)` 口径一致
- 但它们一起偏离 `old`
- 说明 old/rollout 语义与 current train 语义不同

### 中间验证

切到：

- `proximal_recompute_model_mode: eval`

后：

- `prox_prev_gap ≈ 0.005`
- `behav_approx_kl ≈ 0.004`
- `prox_old_gap_abs_mean ≈ 0.04`
- 但 `current logprobs` 仍和 `old/proximal` 差很大

解释：

- old-policy 与 rollout/eval 语义完全匹配
- 真正的问题只剩 current train forward

### 修复后

采用：

- `actor_train_model_mode: behavior_align`

后：

- `behav_approx_kl` 近 0
- `prox_old_logprob_gap_abs_mean` 低位
- `debug_train_rollout_logprob_gap_abs_mean = 0`
- `proximal_ratio ≈ 1`
- `behav_weight_mean ≈ 1`

解释：

- current / proximal / old 三者已基本对齐
- 原始高 KL 问题已被实质性解决

---

## 诊断日志与调试开关

为了定位本问题，新增了整链路 trace 能力：

- rollout 侧：
  - `STEP SUMMARY [ROLLOUT]`
  - `ROLLOUT TRACE`
- proximal 侧：
  - `STEP SUMMARY [PROXIMAL]`
  - `PROXIMAL RECOMPUTE TRACE`
- train 侧：
  - `STEP SUMMARY [TRAIN]`
  - `TRAIN CONSUME TRACE`

相关配置：

```yaml
algorithm:
  debug_chain_trace: True
  debug_chain_trace_mode: anomaly
  debug_chain_trace_topk: 3
  debug_chain_trace_rank0_only: True
  debug_chain_trace_thresholds:
    behav_approx_kl: 20.0
    prox_old_gap_abs_mean: 20.0
    train_rollout_gap_abs_mean: 0.5
```

建议：

- 日常稳定训练默认关闭
- 只有在重新怀疑语义错位、logprob 配对错误或数据链路错位时再打开

---

## 额外工程修复

定位过程中还顺手补了两个工程问题：

### 1. metric 聚合的 CUDA tensor -> numpy 崩溃

问题：

- debug metric 中混入 CUDA tensor
- 在 `np.mean` 时触发 `TypeError`

修复：

- 在 actor worker 聚合 metric 前统一转成 CPU float

### 2. robosuite 析构期 `MjRenderContextOffscreen.con` 缺失

问题：

- env 子进程退出时，robosuite 的 `__del__` 在半初始化对象上访问 `self.con`

修复：

- 增加防御性 monkey patch，避免 teardown 阶段误报

---

## 后续建议

### 1. 继续拉长训练验证

重点继续看：

- `train/actor/behav_approx_kl`
- `train/actor/prox_old_logprob_gap_abs_mean`
- `env/return`
- `env/success_once`
- 显存是否稳定

### 2. 后续代码整理方向

当前 `behavior_align` 已可作为正式语义模式使用，但后面仍建议进一步整理命名：

- `behavior_align`
- `behavior_align_full`

可以继续演化为更明确的“行为语义 / 训练显存语义”两套配置，而不是继续借用 `train/eval` 这些容易误解的名字。

### 3. 若未来再次出现高 `behav_approx_kl`

优先排查顺序建议为：

1. rollout / proximal / current 三者是否仍在同一语义
2. `debug_train_rollout_logprob_gap_abs_mean` 是否抬高
3. `old_vs_local_recompute_gap_abs_mean` 是否抬高
4. 再考虑 backlog、版本滞后和 importance weight 尾部

---

## 相关文件

- `rlinf/models/embodiment/omni_vla/omni_vla_action_model.py`
- `rlinf/workers/actor/async_ppo_fsdp_worker.py`
- `rlinf/workers/rollout/hf/huggingface_worker.py`
- `rlinf/utils/chain_trace.py`
- `examples/embodiment/config/libero_spatial_ppo_omni_vla_quickstart.yaml`

