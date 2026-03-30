> 状态说明
>
> - 这篇文档仍然保留，作为 `Omni-VLA async PPO` 的指标观察与训练判读手册。
> - 如果你想先快速了解 Omni-VLA 适配 RLinf 的全流程，请先读：
>   [`docs/integration/omni_vla_rlinf_integration_full_journey.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/omni_vla_rlinf_integration_full_journey.md)
> - 如果你已经确认问题集中在 async PPO 训练判读，而不是接入或 checkpoint 工程问题，再读本文最合适。

# Omni-VLA Async PPO 指标观察手册

如果当前 run 的核心异常表现为：

- `behav_approx_kl` 长期高位
- `proximal_*` 却接近正常

建议先看这份专项记录：

- [`docs/integration/omni_vla_async_ppo_behavior_semantics_fix_2026-03-27.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/omni_vla_async_ppo_behavior_semantics_fix_2026-03-27.md)

该问题在 2026-03-27 已定位为“rollout 行为语义与 actor train forward 语义分叉”，不是简单的 async lag 或超参过大。

这份手册面向当前这条训练链：

- `AsyncPPOEmbodiedRunner`
- `AsyncPPOEmbodiedFSDPActor`
- `AsyncMultiStepRolloutWorker`
- `AsyncEnvWorker`
- `loss_type = decoupled_actor_critic`
- `rollout.recompute_logprobs = True`

目标不是解释所有 PPO 理论，而是帮助你在训练时快速回答四个问题：

1. 训练现在是不是跑在正确链路上。
2. 策略有没有真的在更新。
3. 更新是健康推进，还是被 clip、lag、错配卡住了。
4. 当前问题更像出在 `env`、`rollout`、`actor` 还是 `critic`。

## 1. 先看什么

每次新起一个 run，建议按这个顺序看：

1. `env/*`
2. `rollout/*`
3. `train/actor/proximal_*`
4. `train/actor/behav_*`
5. `train/actor/policy_loss`、`clip_fraction`
6. `train/critic/*`

原因很简单：

- `env` 决定任务本身有没有在成功。
- `rollout` 决定采样出来的数据好不好。
- `proximal_*` 决定 decoupled PPO 有没有接对。
- `behav_*` 决定 async policy lag 大不大。
- `critic/*` 决定 value 估计有没有在跟上。

## 2. 指标分层

可以把整套指标分成四层。

### A. 环境层

核心看：

- `env/success_once`
- `env/reward`
- `env/return`
- `env/episode_len`

怎么理解：

- `success_once` 最直观，越高越好。
- `reward` 反映单步奖励密度。
- `return` 反映整条轨迹累计质量。
- `episode_len` 反映 episode 是提早结束，还是一直打满 horizon。

经验判断：

- `success_once` 上升，通常比单独看 reward 更可靠。
- `return` 上升但 `success_once` 不动，说明可能只是 shaping reward 变好了。
- `episode_len` 永远打满，不一定错，但说明策略还没稳定学会“更早完成任务”。

### B. Rollout 层

核心看：

- `rollout/rewards`
- `rollout/reward_nonzero_fraction`
- `rollout/returns_mean`
- `rollout/returns_min`
- `rollout/returns_max`
- `rollout/advantages_positive_fraction`
- `rollout/advantages_mean`

怎么理解：

- `returns_mean` 是 rollout 数据质量的核心指标。
- `returns_min` 看最差轨迹是不是也在改善。
- `reward_nonzero_fraction` 看稀疏奖励有没有越来越容易被打到。
- `advantages_mean` 正常情况下应接近 0。
- `advantages_positive_fraction` 通常在 `0.4-0.6` 附近比较正常。

经验判断：

- `returns_mean` 持续上升，是训练有效的强信号。
- `returns_min` 从很负慢慢往 0 靠，说明最差行为在变少。
- `advantages_mean` 明显偏离 0，通常表示标准化、value 或 mask 有问题。
- `advantages_positive_fraction` 长期接近 0 或 1，通常不正常。

### C. Actor 层

这是 async PPO 最关键的一层。

核心看三组：

#### 1) Proximal 组

- `train/actor/proximal_ratio`
- `train/actor/proximal_approx_kl`
- `train/actor/clipped_proximal_ratio`

它回答的问题是：

“训练时当前策略”和“proximal 重算策略”是不是对齐。

经验判断：

- `proximal_ratio` 应该在 `1` 附近。
- `proximal_approx_kl` 应该接近 `0`。
- `clipped_proximal_ratio` 应该也在 `1` 附近，而不是长期贴边界。

异常信号：

- `proximal_ratio` 远离 `1`，比如只有 `0.1-0.3`。
- `proximal_approx_kl` 很大，比如几十。
- `clipped_proximal_ratio` 长期钉在 `0.95x` 或 `1.05x`。

这通常意味着：

- `proximal_logprobs` 和训练时 `current logprobs` 不是同一条前向路径。
- 常见根因是 `eval/train` 模式不一致、mask 错位、chunk 聚合口径不一致。

#### 2) Behavior 组

- `train/actor/behav_approx_kl`
- `train/actor/behav_clip_fraction`
- `train/actor/current_version`
- `train/actor/average_version`

它回答的问题是：

“当前训练策略”和“rollout 采样时的行为策略”差了多少。

经验判断：

- `behav_approx_kl` 小，说明 async lag 小。
- `behav_approx_kl` 大，说明 rollout 数据已经明显旧了。
- `current_version - average_version` 越大，通常代表 lag 越重。

异常信号：

- `behav_approx_kl` 长期很大，但 `proximal_*` 正常。

这通常意味着：

- 不是 logprob 计算坏了。
- 而是 async PPO 的 policy lag 在变大。

处理优先级：

1. 先确认 `proximal_*` 正常。
2. 再考虑减小 lag，比如减小更新幅度、缩短版本差、减少异步积压。

#### 3) Update 强度组

- `train/actor/policy_loss`
- `train/actor/total_loss`
- `train/actor/clip_fraction`
- `train/actor/grad_norm`
- `train/actor/entropy_loss`

怎么理解：

- `policy_loss` 看 actor 优化目标有没有在动。
- `clip_fraction` 看 PPO 有多少样本进入 clipping。
- `grad_norm` 看梯度是不是过大或过小。
- `entropy_loss` 看探索是不是在收缩。

经验判断：

- `clip_fraction` 在中低水平通常更健康。
- 长期特别高，说明更新经常被 clip。
- 长期特别低，也可能说明更新太弱。
- `grad_norm` 偶尔尖峰可以接受，长期爆高要警惕。

### D. Critic 层

核心看：

- `train/critic/value_loss`
- `train/critic/explained_variance`
- `train/critic/value_clip_ratio`
- `rollout/return_value_gap_mean`
- `rollout/return_value_gap_abs_mean`

怎么理解：

- `value_loss` 看回归误差。
- `explained_variance` 看 value 对 return 的解释能力。
- `return_value_gap_*` 看 rollout value 和实际 return 的差距。

经验判断：

- `value_loss` 下降通常是好事。
- `explained_variance` 从负值往 0、再往正值走，是健康信号。
- `return_value_gap_abs_mean` 下降，说明 critic 在贴近真实回报。

异常信号：

- `value_loss` 降不下去。
- `explained_variance` 长期大负值。
- `return_value_gap_abs_mean` 长期很大不动。

这通常意味着：

- critic 学不动。
- reward/value 标尺不一致。
- 或者 actor 分布变化太快，critic 跟不上。

## 3. 一套最实用的观察流程

每次看图时，可以按下面的顺序走。

### Step 1. 先确认是不是“真训练”

检查：

- `env/success_once`
- `rollout/returns_mean`
- `train/actor/policy_loss`
- `train/critic/value_loss`

如果这些都几乎完全不动，先怀疑：

- 数据没进来。
- 梯度没更新。
- 路径没走对。

### Step 2. 确认 decoupled PPO 有没有接对

检查：

- `train/actor/proximal_ratio`
- `train/actor/proximal_approx_kl`
- `train/actor/clipped_proximal_ratio`

健康状态：

- `proximal_ratio` 在 `1` 附近。
- `proximal_approx_kl` 接近 `0`。
- `clipped_proximal_ratio` 在 `1` 附近。

如果不是这样，先不要急着调 learning rate。
优先排查前向路径是否一致。

### Step 3. 再看 async lag 大不大

检查：

- `train/actor/behav_approx_kl`
- `train/actor/current_version`
- `train/actor/average_version`

如果：

- `proximal_*` 正常
- `behav_approx_kl` 很大

那说明不是 proximal 算错，而是 behavior policy 太旧。

### Step 4. 再看 actor 是不是被 clip 卡住

检查：

- `train/actor/clip_fraction`
- `train/actor/policy_loss`
- `train/actor/grad_norm`

如果 `clip_fraction` 很高，且 `policy_loss` 不动，说明 actor 更新在被压扁。

### Step 5. 最后看 critic 跟不跟得上

检查：

- `train/critic/explained_variance`
- `rollout/return_value_gap_abs_mean`

如果 actor 看起来正常，但 return 提不上去，而 critic 也很差，就要先补 critic。

## 4. 异常模式速查表

### 模式 A. `proximal_ratio` 远离 1，`proximal_approx_kl` 很大

现象：

- `proximal_ratio` 不是 `1` 附近
- `proximal_approx_kl` 很大
- `clipped_proximal_ratio` 长期贴边

含义：

- proximal 重算和训练前向不一致。

优先排查：

1. `model.eval()` 和 `model.train()` 是否混用。
2. `self.training` 分支是否改变了 logprob 路径。
3. `chunk_level`、`token_level` 聚合口径是否一致。
4. mask、shift、denoise step 对齐是否一致。

### 模式 B. `proximal_*` 正常，但 `behav_approx_kl` 很大

现象：

- `proximal_ratio` 正常
- `proximal_approx_kl` 接近 0
- `behav_approx_kl` 很大

含义：

- decoupled PPO 接线没问题。
- 主要问题是 async policy lag。

优先排查：

1. rollout 数据是不是太旧。
2. actor 更新是不是太快。
3. 版本差是不是越来越大。

### 模式 C. `returns_mean` 上升，但 `success_once` 不上升

含义：

- 更像在学 shaping reward，还没真正学会任务。

建议：

- 优先看 success，而不是只看 return。

### 模式 D. `value_loss` 降了，但 `explained_variance` 还是很差

含义：

- critic 可能学到了均值，但没学到结构。

建议：

- 联合看 `return_value_gap_abs_mean`，不要只看 `value_loss`。

### 模式 E. `clip_fraction` 很低，policy_loss 也很小，return 不动

含义：

- actor 更新可能太弱。

建议：

- 再确认是不是学习率、clip、update_epoch 太保守。

## 5. 当前这条 Omni-VLA 训练里，最值得长期盯的指标

如果时间不多，我建议固定盯这 10 个：

1. `env/success_once`
2. `env/return`
3. `rollout/returns_mean`
4. `rollout/return_value_gap_abs_mean`
5. `train/actor/proximal_ratio`
6. `train/actor/proximal_approx_kl`
7. `train/actor/behav_approx_kl`
8. `train/actor/clip_fraction`
9. `train/critic/explained_variance`
10. `train/critic/value_loss`

这 10 个基本已经够回答：

- 任务有没有学会。
- actor 前向是否一致。
- async lag 是否过大。
- critic 是否跟上。

## 6. 一份简化版口令

你可以把训练观察简化成下面这几句自问：

1. `success` 和 `return` 有没有涨。
2. `returns_mean` 有没有涨，最差轨迹有没有变好。
3. `proximal_ratio` 是否在 1 附近，`proximal_approx_kl` 是否接近 0。
4. `behav_approx_kl` 是不是过大。
5. `clip_fraction` 是不是过高。
6. `explained_variance` 和 `return_value_gap_abs_mean` 有没有改善。

如果第 3 条不对，先修链路。
如果第 3 条对、第 4 条很差，先看 async lag。
如果 actor 都正常但第 6 条差，先补 critic。

## 7. 结合这次排障经验的记忆点

这次我们实际遇到过两类问题：

### 问题 1. 跑错 runner

现象：

- 配置写了 `recompute_logprobs=True`
- 但实际入口走的是同步 `EmbodiedRunner`

教训：

- 先确认“实际跑的是哪条链路”，不要只信 yaml。

### 问题 2. `proximal` 和 `current` 前向模式不一致

现象：

- `proximal_ratio` 远离 1
- `proximal_approx_kl` 巨大
- `behav_approx_kl` 反而很小

教训：

- 这种组合通常不是 behavior lag，而是前向路径不一致。
- 在 Omni-VLA 里，`self.training` 本身就会改变 logprob 计算链。

## 8. 最后一句

不要只看单个指标，要看“组合”。

最有用的观察方式不是问“这个数高不高”，而是问：

- 它和同层其他指标一致吗。
- 它和上一轮相比在变好还是变坏。
- 它更像是链路问题、更新强度问题，还是任务本身问题。

一旦你习惯用“指标组合”来判断，训练图会比单看某一条线清楚很多。
