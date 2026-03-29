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

---

## 2026-03-30 补充诊断：中等版训练与 checkpoint/save

### 1. 中等版训练配置的阶段性结论

本轮在保持语义修复配置不变的前提下，采用了更积极但仍偏保守的训练配置：

```yaml
algorithm:
  update_epoch: 2

actor:
  optim:
    lr: 1.875e-7
    value_lr: 2.6e-5
```

结合后续多轮 `env / rollout / train` 指标，阶段性结论为：

- 训练主链路稳定
- 语义对齐仍然成立
- actor 更新强度已进入“有效但不过猛”的区间
- critic 比此前更稳，`explained_variance` 整体改善
- 当前问题主矛盾已不再是语义错位，而是 checkpoint/save 的工程稳定性

### 2. 训练本体状态

从多轮曲线看：

- `train/actor/behav_approx_kl` 持续低位
- `train/actor/proximal_ratio` 与 `train/actor/clipped_proximal_ratio` 长期在 `1.0` 附近
- `train/actor/prox_old_logprob_gap_abs_mean` 维持在修复后的低位范围
- `train/actor/clip_fraction` 大致在 `0.08-0.11`
- `train/actor/behav_weight_mean ≈ 1`
- `train/actor/behav_weight_gt_2_fraction = 0`
- `train/actor/dropped_rollout_batches = 0`
- `train/actor/current_version_minus_batch_version_max = 1`

`env / rollout` 侧也没有看到：

- backlog
- rollout 权重滞后
- returns 明显塌陷
- episode 边界错乱

因此可以认为：

- 当前配置已可作为“可长跑”的训练配置
- 训练本体与 rollout 主链路没有明显异常

### 3. 视频复核

对最新训练视频抽帧后，观察到：

- 机械臂动作整体连贯
- 没有明显高频抖动、原地抽搐或长时间卡死
- 成功样本不是纯随机碰撞，更像是学到了一条基本正确的动作模式
- 成功后画面继续渲染且出现 `termination: True`，与 eval/视频侧继续渲染设置相符，不构成异常

阶段性判断：

- 视频没有暴露新的策略异常
- 当前策略已具备基本正确的任务行为，但仍有继续提纯空间

### 4. checkpoint/save 现象

后续排查发现，训练在触发 `save_interval` 时容易被误判为“卡死”，但新的日志显示：

- `actor.save_checkpoint()` 实际可以成功返回
- 在一次 `global_step = 2` 的测试中，runner 记录到：
  - save dispatched
  - save finished
  - 总耗时约 `102.67s`

这说明至少有一部分“卡死感”来自：

- checkpoint/save 本身耗时较长
- 保存期间主循环没有继续推进
- 短时间内缺少额外日志，用户容易误判为 hang

### 5. checkpoint/save 的真实工程风险

虽然存在“保存很慢但能成功”的情况，但此前在 `save_interval = 30/40` 等场景下仍多次出现：

- `Unsupported object type: <random int>`
- metadata size 被读成超大值
- 随后触发异常内存申请或 Gloo peer closed

这类现象更像：

- async env / rollout / actor 主通信仍在进行
- checkpoint 流程插入了额外的分布式/collective 交互
- 某些 recv 读到了被打乱的消息头或 metadata

因此目前对 save 问题的判断应拆成两层：

1. **保存能力本身**
   - 已证明并非必然失败
   - 至少在某些情况下可以成功完成，只是很慢
2. **边训练边保存的时序安全性**
   - 仍然存在工程风险
   - 尚不能认定当前 async 流程下的任意保存点都稳定安全

### 6. 新增诊断日志

为进一步定位 save 具体卡点，新增了 checkpoint 调试日志，覆盖：

- runner 进入 `_save_checkpoint()` 前后
- runner dispatch `actor.save_checkpoint()` 后等待阶段
- actor 侧进入 `FSDPModelManager.save_checkpoint()`
- FSDP strategy 的：
  - pre-save barrier
  - 构建 training state
  - `dcp.save(...)`
  - post-save barrier
  - full model state dict 导出

同时还新增了：

- `_save_checkpoint()` 返回后，runner 恢复主循环的日志
- async PPO 每轮 step 开始时的日志

相关文件：

- `rlinf/runners/embodied_runner.py`
- `rlinf/runners/async_ppo_embodied_runner.py`
- `rlinf/hybrid_engines/fsdp/fsdp_model_manager.py`
- `rlinf/hybrid_engines/fsdp/strategy/base.py`

### 7. 调试过程中的一次额外问题

在新增 checkpoint 调试日志时，曾一度误用不存在的 `self.rank` 字段，导致：

- actor 在进入 `save_checkpoint()` 早期即抛出 `AttributeError`
- Ray 杀死 actor
- env / rollout 侧随后出现通信断裂与 `Unsupported object type`

该问题是日志代码本身引入的二次故障，不是 checkpoint 主问题本身，随后已修正为读取已有 rank 字段。

### 8. 当前建议

当前建议将训练与 checkpoint 问题分开处理：

- 训练配置可继续保持当前“中等版”
- 若目标是稳定长跑，避免使用过小的 `save_interval`
- 若目标是验证 checkpoint 是否能成功，优先使用极短 smoke test 并配合新的 debug 日志
- 后续若要彻底解决 save 风险，优先方向应为：
  - 在 save 前让 async env / rollout 通信 quiesce
  - checkpoint 使用更独立的通信路径或同步屏障
  - 对 metadata/object header 增加更严格的 sanity check

当前阶段的综合结论是：

- **训练已经基本稳定**
- **视频没有暴露新的策略异常**
- **save 能力不是完全坏的，但“边训练边保存”的时序安全性仍需进一步工程化修复**

