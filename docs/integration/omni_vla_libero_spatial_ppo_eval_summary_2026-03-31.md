# Omni-VLA Libero-Spatial PPO 实验结论整理

## 1. 文档定位

这篇文档用于沉淀一轮具体实验的结果：`Omni-VLA` 在 `libero_spatial` 上以基础模型为起点，经过 `RLinf async PPO` 训练后，各 checkpoint 在统一评测口径下的真实表现。

它回答四个问题：

1. 基础模型在固定测试集上的 baseline 是多少。
2. PPO 在这轮实验中是否真的带来了收益。
3. 最佳 checkpoint 出现在什么时候，而不是最终 step。
4. 下一轮超参该往什么方向调。

如果你想先了解整个适配背景和工程问题，请先读：

- [`docs/integration/omni_vla_rlinf_integration_full_journey.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/omni_vla_rlinf_integration_full_journey.md)

如果你想了解 async PPO 指标该怎么判读，再配合阅读：

- [`docs/omni_vla_async_ppo_metric_handbook.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/omni_vla_async_ppo_metric_handbook.md)

## 2. 实验设置

### 2.1 训练配置

训练配置使用：

- [`examples/embodiment/config/libero_spatial_ppo_omni_vla_quickstart.yaml`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/examples/embodiment/config/libero_spatial_ppo_omni_vla_quickstart.yaml)

这轮实验从 `global_step_60` checkpoint 恢复继续训练，持续观察到 `global_step_400`。

### 2.2 标准化评测配置

为了保证不同 checkpoint 之间可比，所有 checkpoint 都使用同一套固定 eval 口径：

- `runner.only_eval = True`
- `algorithm.eval_rollout_epoch = 1`
- `env.eval.total_num_envs = 10`
- `env.eval.auto_reset = True`
- `env.eval.max_episode_steps = 240`
- `env.eval.max_steps_per_rollout_epoch = 2400`
- `env.eval.use_fixed_reset_state_ids = True`
- `env.eval.use_ordered_reset_state_ids = True`
- `env.seed = 0`
- `eval/num_trajectories = 100`

评测命令统一为：

```bash
bash examples/embodiment/eval_embodiment.sh libero_spatial_ppo_omni_vla_eval
```

### 2.3 指标解释

在这组 `LIBERO` 任务里，几个核心指标的含义如下：

- `eval/success_once`: 一条轨迹中是否曾经成功过一次。
- `eval/success_at_end`: 轨迹结束时是否仍然成功，是更可靠的主指标。
- `eval/return`: 整条轨迹累计回报。在当前任务里与 `success_at_end` 高度一致。
- `eval/reward`: 近似等于 `return / episode_len`，因为 `episode_len=240` 固定，所以数值天然很小。

## 3. 评测结果

### 3.1 基础模型 baseline

不加载任何 RL checkpoint，仅使用基础模型：

- `model_path = /root/data/kaelynwang/Omni_VLA/omni_vla_checkpoint_10000`
- `resume_dir = null`
- `ckpt_path = null`

得到的 `100 trajectories` 固定 eval 结果为：

| 模型 | success_once | success_at_end | return | episode_len |
|---|---:|---:|---:|---:|
| 基础模型 | 0.86 | 0.78 | 0.78 | 240 |

这说明基础 `Omni-VLA` 并不是很弱的起点，未做 RL 时就已经有 `0.78` 的 `success_at_end`。

### 3.2 PPO checkpoint 对比

在统一的 `100 trajectories` 固定 eval 口径下，主要 checkpoint 的结果如下：

| Checkpoint | success_once | success_at_end | return | 结论 |
|---|---:|---:|---:|---|
| Base | 0.86 | 0.78 | 0.78 | 基础模型 baseline |
| 80 | 0.84 | 0.78 | 0.78 | 与 baseline 持平 |
| 160 | 0.92 | 0.85 | 0.85 | 当前最佳之一 |
| 200 | 0.91 | 0.85 | 0.85 | 当前最佳之一 |
| 340 | 0.89 | 0.78 | 0.78 | 回落到接近 baseline |
| 400 | 0.87 | 0.77 | 0.77 | 略低于 baseline |

### 3.3 直接结论

这组结果可以明确说明：

1. `PPO` 在这轮实验中是有效的。
2. 有效提升并不出现在最后，而是在 `160-200` 步区间。
3. `160/200` checkpoint 将 `success_at_end` 从 `0.78` 拉升到了 `0.85`。
4. 继续训练到 `340/400` 后，提升没有被保持住，反而回落到接近基础模型。

## 4. 训练曲线与离线评测的对应关系

### 4.1 为什么训练时有时看到 `env/success_once = 1.0`

在线训练图里的 `env/success_once` 与离线 `100 trajectories` eval 不是同一件事：

- 在线 `env/*` 是当前训练 step 上那一批 rollout 的统计，样本量更小，波动更大。
- 离线 eval 是固定测试集上的独立评测，样本量更大，更适合选 checkpoint。

因此，训练图上某一步出现 `env/success_once = 1.0`，并不意味着该 checkpoint 在固定测试集上也一定是 `1.0`。

### 4.2 为什么 `340` 训练图看起来不错，但离线 eval 只有 `0.78`

这说明在线训练统计比真实固定评测更乐观，属于正常现象。当前这条链路里，判断 checkpoint 好坏时应优先相信离线 eval，而不是单个 step 的在线 `env` 指标。

### 4.3 为什么 `reward` 总是很低

在当前配置中：

- `episode_len = 240`
- `eval/reward ≈ eval/return / 240`

所以即便 `return = 1.0`，`reward` 也只有约 `0.00417`。这个值小是定义导致的，不是训练出错。

## 5. 这轮实验真正说明了什么

### 5.1 PPO 不是没学到

如果只看最优 checkpoint：

- Base: `success_at_end = 0.78`
- PPO best (`160/200`): `success_at_end = 0.85`

说明 PPO 真实带来了约 `+7` 个点的提升。

### 5.2 但 PPO 当前配置学得偏保守

结合 `train/*`、`rollout/*` 和离线 eval，可以把这轮 run 概括为：

- async 机制是健康的，没有明显 stale batch 或语义错位复发。
- actor 和 critic 都在工作，不是假训练。
- 但 actor update 偏保守，更新很稳定，增益出来得慢。
- 最优点出现后没有被后续训练稳定守住。

### 5.3 最后 checkpoint 不能代表最好结果

这是这轮实验最重要的工程结论之一：

- 如果只看 `global_step_400`，会误以为 PPO 基本没有收益。
- 但按标准化 eval 选 best，`global_step_160` 和 `global_step_200` 明显优于基础模型。

因此，后续实验必须：

1. 固定 eval 配置。
2. 按 `eval/success_at_end` 选 best checkpoint。
3. 不再默认用最后 checkpoint 代表整条 run。

## 6. 当前最合理的结论话术

可以直接用下面这段总结本轮实验：

> 在统一的 100 条轨迹固定 eval 配置下，基础 Omni-VLA 的 baseline 为 `success_at_end=0.78`。当前 PPO 配置可在 `global_step_160-200` 将结果提升到 `0.85`，说明 RL 是有效的；但继续训练到 `340/400` 后结果回落到接近基础模型，表明当前 PPO 配置提升不稳定，best checkpoint 必须依赖标准化 eval 选择，而不能直接取最后 step。

## 7. 下一轮实验建议

基于当前结果，下一轮最值得优先尝试的方向不是重做 async 机制，而是让 PPO 更新略微更积极一些。

推荐优先尝试：

```yaml
algorithm:
  clip_ratio_high: 0.1
  clip_ratio_low: 0.1

actor:
  optim:
    lr: 3.75e-7
```

这组调整的目标是：

- 保留当前已经验证健康的 async PPO 语义和 freshness 机制
- 避免“更新过小、提升来得慢、峰值守不住”的问题

同时建议：

1. 继续保留标准化 `100 trajectories` eval 配置不变。
2. 重点评测 `80 / 120 / 160 / 200 / 240 / 300` 这类 checkpoint。
3. 以 `eval/success_at_end` 作为 best checkpoint 主指标。

## 8. 最终建议

对于当前这条 run，最该保留和复用的是：

- `global_step_160`
- `global_step_200`

对于后续实验流程，最该固化的做法是：

1. 用固定 eval 配置做 checkpoint 间可比评测。
2. 把基础模型 baseline 一起报告，不再只汇报 RL 后结果。
3. 用 best checkpoint，而不是最终 checkpoint，代表该 run 的真实上限。
