# Omni-VLA 适配 RLinf 全过程总览

## 1. 背景与目标

本文档用于沉淀 `Omni-VLA` 适配 `RLinf` 强化学习框架的完整工程过程，面向项目内部协作与后续维护者。

它回答四个核心问题：

1. 为什么这次适配不能简单复用 `openpi` 的集成方式。
2. Omni-VLA 接入 RLinf 的过程中到底改了什么。
3. 关键问题是如何被逐步定位和修复的。
4. 当前已经稳定了什么，仍有哪些工程风险。

建议把本文当作主入口阅读；更深的 async PPO 专题分析与指标判读，分别放在：

- [`docs/integration/omni_vla_async_ppo_behavior_semantics_fix_2026-03-27.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/omni_vla_async_ppo_behavior_semantics_fix_2026-03-27.md)
- [`docs/omni_vla_async_ppo_metric_handbook.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/omni_vla_async_ppo_metric_handbook.md)

## 2. 起点：openpi 集成方式与 Omni-VLA 架构差异

### 现象

项目最初并不是从零开始接入 embodied VLA，而是已有一套 `openpi -> RLinf` 的集成路径。直觉上最自然的想法，是把 `Omni-VLA` 看成 `openpi` 的增强版，沿用同样的安装、注册和 policy wrapper 逻辑。

### 根因

后续分析表明，这个假设只在最表层成立：

- 包结构上，`Omni-VLA` 兼容 `openpi` 的大体组织方式。
- 但模型内部结构、依赖项、训练前向语义、显存策略和 checkpoint 形态都已经明显偏离原始 `openpi`。

尤其需要注意的差异包括：

- `Omni-VLA` 采用了更重、更定制化的多专家架构，而不是原始 `openpi` 的基础 VLA 路径。
- `Omni-VLA` 在训练与推理态上复用了更多共享逻辑，很多行为不只是“有没有 dropout”，而是会改变前向语义。
- 模型对 `gradient checkpointing`、`use_cache`、attention mask、vision/backbone 初始化路径等细节更敏感。
- SFT checkpoint 与 RLinf 内部模型对象的 `state_dict` 层级并不天然一致。

### 修复

适配策略从“直接替换 openpi”调整为“以 openpi 集成为参考，但承认 Omni-VLA 需要专项工程化适配”：

- 保留 `openpi` 作为理解 RLinf embodied 链路的对照样本。
- 单独增加 `omni_vla` 模型类型、安装流程、配置读取与 policy wrapper。
- 把后续问题拆成几个独立专题：启动兼容性、显存与 activation、checkpoint 权重加载、async PPO 语义、checkpoint/save 稳定性。

### 验证

后续所有重大问题都证明了这个判断是对的：如果继续把 Omni-VLA 当成 `openpi` 的无脑替换，训练确实能“接上”，但无法可靠地“跑通”。

### 最终结论

`Omni-VLA` 与 `openpi` 的关系更像“共享部分接口约定的近亲”，而不是“可直接互换的同一模型族”。适配工作必须覆盖模型注册、前向语义、显存策略、checkpoint 语义和 async RL 交互这几层。

## 3. 第一次接入：模型注册、安装、wrapper、配置对齐

### 现象

第一阶段的目标是先让 `RLinf` 识别并构造 `Omni-VLA`，至少跑到训练入口，而不是立刻优化训练质量。

### 根因

`RLinf` embodied 链路要求模型满足一组明确接口：

- 能被 `SupportedModel` 识别
- 有统一的模型构造入口
- 具备训练 forward 与 rollout 预测接口
- 能根据数据配置完成 transform、norm stats、action/logprob/value 相关逻辑

### 修复

首次接入阶段完成了以下关键适配：

- 安装脚本补充 `Omni-VLA` 安装能力。
- 在 `SupportedModel` 中加入 `omni_vla`。
- 新增 `rlinf/models/embodiment/omni_vla/` 下的模型包装与配置读取逻辑。
- 将 RL 所需接口适配到 `OmniVLAForRLActionPrediction` 这类 wrapper 上，使其能对接 `RLinf` 的 actor / rollout / worker。
- 数据配置层引入 `omni_vla_libero`、`omni_vla_maniskill` 等配置映射，尽可能复用已有 embodied 数据配置思路。

### 验证

这一阶段的成功标准不是“训练指标好看”，而是：

- 依赖能装上
- 模型能被识别
- 训练脚本能跑到模型初始化与 worker 构造之后

这些能力在后续调试中已被反复验证，否则后面的启动期 bug 和训练期 bug 根本无法出现。

### 最终结论

首次接入阶段解决的是“系统是否认识 Omni-VLA”这个问题，为后续所有更深层问题打开了排查入口。

## 4. 训练启动期问题与修复

### 现象

在第一次能真正拉起训练后，系统暴露出一系列基础兼容问题，典型包括：

- 预训练路径变量未定义或调用方未传参
- 某些运行时属性访问路径和实际模型结构不一致
- attention mask / dtype 不匹配
- 某些模块存在硬编码冻结逻辑或默认行为与 RL 训练预期冲突
- 训练日志中出现与实现细节相关的显式报错或 silent mismatch

### 根因

这些问题的共同特点是：它们不是 RL 算法问题，而是“Omni-VLA 原始工程假设”与 “RLinf embodied 运行时假设”不一致。例如：

- 原始训练脚本默认依赖的模型路径、预训练路径、资产路径，在 RLinf 里不一定由同一调用栈传入。
- 原始代码里的属性层级和 wrapper 后的对象层级未必一致。
- 某些 attention / dtype 处理在原仓库环境中能工作，但在 FSDP + RLinf + 多 worker 的组合下会暴露。
- 部分默认冻结逻辑适合原先的训练方案，但不适合 RL 策略微调。

### 修复

这一阶段完成了大量“先让它稳定启动”的工程修补，主要包括：

- 修正 `vlm_pretrained_path`、`vggt_pretrained_path` 等关键初始化路径的传递。
- 修复 attention mask dtype 及相关前向兼容问题。
- 校正模型内部部分属性访问路径和结构适配。
- 去掉或调整不符合 RL 训练预期的硬编码冻结行为。
- 在必要位置增加 debug 日志和基本加载验证，避免 silent failure。

### 验证

这一阶段的验证标准是：

- 训练能稳定启动，不再在模型构造或首轮前向时报基础错误。
- 模型初始化、权重装载、worker 建立和 rollout 入口都能连续跑通。

### 最终结论

启动期问题解决后，系统从“不能跑”进入“能跑但不一定对”的状态，为后续显存、权重加载和 async PPO 问题的真实暴露创造了条件。

## 5. 显存与 activation / gradient checkpointing 调整

### 现象

训练一旦真正开始，显存迅速成为主要瓶颈之一。早期现象包括：

- activation 显存占用过高
- 某些组合下 `gradient checkpointing` 与 `use_cache` 行为不一致
- 训练路径中混入推理态缓存或与期望不一致的内存策略

### 根因

Omni-VLA 的模型体量和内部结构决定了：

- 不可能简单沿用原始轻量 VLA 的显存策略
- `gradient checkpointing` 不只是性能开关，还会影响训练路径选择
- `use_cache` 在训练态残留会放大显存压力，并可能改变前向执行方式

### 修复

围绕 activation memory 做了几类关键调整：

- 梳理并收紧 `gradient checkpointing` 的开启/关闭路径。
- 明确 `gradient checkpointing` 与 `use_cache` 的联动关系：训练开 checkpointing 时关闭 cache，关闭 checkpointing 时再对称恢复。
- 修正 Omni-VLA 模型内部相关模块的 checkpointing 切换行为，避免部分专家走错路径。
- 在 RLinf wrapper 层补充适配，让训练前向与推理前向在显存策略上不再互相污染。

### 验证

验证信号主要包括：

- 训练能够在目标 batch / FSDP 配置下稳定运行
- 显存占用回到可接受区间
- 不再出现明显由 cache / checkpointing 组合引起的异常警告或错误路径

### 最终结论

这一步解决的是“能不能长期训下去”的基础条件。没有显存策略对齐，后面的 PPO 语义诊断很容易被资源问题干扰。

## 6. SFT checkpoint / 权重加载 / norm stats / key remap 问题

### 现象

即使训练能跑，也一度出现“模型像没学到东西”或行为明显异常的情况。排查后发现，问题并不全来自 RL 本身，还有一层关键风险：SFT checkpoint 可能没有被正确加载。

相关现象包括：

- `strict=False` 下大量 key 被静默跳过
- `reasoning_expert` 权重没有真正对齐到模型对象
- `norm_stats` 路径与 checkpoint 实际资产布局不一致
- `lm_head`、`embed_tokens` 等权重共享或映射路径与原假设不同

### 根因

主要根因有两类：

1. checkpoint key 层级与 RLinf 中 `state_dict` 层级不一致
2. `norm_stats` 和相关资产路径沿用了原训练工程假设，在 RLinf 环境中不完全成立

这意味着“加载成功”不能只看有没有报错，必须显式验证实际加载覆盖率。

### 修复

围绕权重加载做了系统修复：

- 用手动 key remapping 替代简单的 `strict=False` 静默加载。
- 修正 `reasoning_expert.language_model`、`lm_head`、`embed_tokens` 等关键层级映射。
- 补充 missing / unexpected key 的显式日志，避免 silent mismatch。
- 修正 `norm_stats` 的加载方式与路径拼接逻辑，使其与实际资产布局对齐。
- 对 rollout / model 初始化阶段增加必要的 DEBUG 验证，确认载入后的模型不是“看似加载成功、实际还在用预训练原始权重”。

### 验证

这一阶段的验证标准包括：

- checkpoint missing / unexpected key 数量显著下降，并可解释
- 关键模块的权重确实来自 SFT checkpoint
- `norm_stats` 不再因为路径假设错误而持续失效

### 最终结论

这一轮修复解决的是“训练对象是不是你以为的那个模型”这个问题。它是训练质量问题里最容易被忽视、但后果最重的一层。

## 7. Async PPO 语义错位问题

### 现象

在 async PPO 训练真正跑起来之后，最关键的一轮问题不是单纯 reward 不涨，而是指标长期出现语义冲突：

- `behav_approx_kl` 长期高位
- `proximal_*` 指标却相对正常
- rollout 版本同步看起来也没有明显 backlog

这类现象说明 actor 当前训练口径与 rollout / old-policy 口径不在同一条语义链上。

### 根因

最终定位到的根因是：

**Omni-VLA 将多种语义同时绑定在 `self.training` 上。**

也就是说，`self.training` 在这个模型里承载的并不只是 dropout / BN 这类传统训练态语义，还同时影响：

- 训练前向 vs rollout 前向
- logprob 计算口径
- gradient checkpointing / use_cache / 显存优化路径

在 async PPO 中，这会让 rollout / old-policy 的行为语义，与 actor 当前训练前向的语义发生系统性错位。

### 修复

这一阶段的关键不是改 PPO，而是把模型语义显式拆开：

- 在 Omni-VLA actor 路径中引入更细粒度的语义控制，而不是继续依赖粗粒度 `self.training`。
- 区分 rollout 行为语义、训练语义和显存优化语义。
- 对 logprob 路径与训练前向路径做口径对齐。
- 保证训练时的行为比较是“同一条语义链上的 current / old / proximal logprob 对照”。

### 验证

修复后的关键指标变化非常明确：

- `train/actor/behav_approx_kl` 从长期高位降到接近 `0`
- `train/actor/prox_old_logprob_gap_abs_mean` 降到低位
- `train/actor/debug_train_rollout_logprob_gap_abs_mean` 回到 `0`
- `train/actor/proximal_ratio` 回到 `1.0` 附近

### 最终结论

这是整个适配过程中最重要的一次“从表象指标走到模型语义”的修复。它证明 async PPO 的主矛盾一度并不在 backlog 或超参，而在模型前向语义本身。

## 8. Async PPO 指标与训练判读方法

### 现象

语义修复之后，另一个问题变成：如何判断训练是真的健康推进，而不是“某个指标看着正常”。

### 根因

async PPO 链路长、参与方多，只盯单一指标很容易误判。尤其在 Omni-VLA 这类前向语义敏感模型里：

- `env/*`
- `rollout/*`
- `train/actor/proximal_*`
- `train/actor/behav_*`
- `train/critic/*`

这几层必须联动解读。

### 修复

为此整理了一套指标观察手册，明确了：

- 新 run 应该按什么顺序看指标
- 哪些指标判断“有没有更新”
- 哪些指标判断“是不是语义错位”
- 哪些指标判断“更像 lag、clip、critic 问题还是 env 问题”

### 验证

这套判读方式已经被用于：

- 排除 rollout version lag 与 backlog
- 确认语义错位问题不是简单的 policy lag
- 在修复后判断训练曲线是否回到合理区间

### 最终结论

指标手册不是附属材料，而是 Omni-VLA async PPO 维护的一部分。后续只要换配置、换模型、换 runner，都应该按同样分层方式复核。

## 9. checkpoint/save 稳定性问题与最终工程修复

### 现象

语义问题修复后，新的主矛盾转移到了 checkpoint/save：

- 表面上像“保存卡死”
- 某些时候 `actor.save_checkpoint()` 能成功返回，但耗时很长
- 另一些时候会在 save 前后出现通信错位、随机 object type、shape 错误或恢复后主循环不再推进

### 根因

这一块经历了多轮误判，最终逐步收敛出两个层次：

1. 保存能力本身不是完全坏的
2. 边训练边保存的时序安全性存在工程风险

后续进一步定位发现，真正的稳定修复不只是“等 save 完”，还包括：

- checkpoint 前让旧一轮 env/rollout 自然跑到安全边界
- 重启后不能复用旧 channel
- rollout 内部 runtime 状态不能直接沿用 checkpoint 前的累积值

### 修复

围绕 save 稳定性做了成体系的工程修复：

- 在 runner 侧为 async PPO checkpoint 增加 quiesce 逻辑，先停 env/rollout 异步流水线，再进行 save。
- 将硬 `cancel()` 改为 graceful stop，让 env / rollout 在当前 round 结束后自然退出，避免半截 batch。
- checkpoint 恢复后使用新的 channel generation，避免重新连回旧 channel。
- 为 rollout worker 增加 runtime reset，复位 `finished_episodes`、pending weight sync 等状态，避免 checkpoint 后被 stale 判定卡住。
- 增加 checkpoint/save 与 step 恢复阶段的分层 debug 日志，帮助区分“保存慢”“恢复卡住”“通信错位”这几类不同问题。

### 验证

近期验证表明：

- checkpoint 文件可以稳定产出
- 保存结束后训练可以继续推进
- 旧有的“保存后随机 object type / 半截 batch / 重启后卡死”问题已经被逐步修正

### 最终结论

这一步解决的是“训练是否具备工程可运行性”问题。相比模型语义问题，它更偏系统工程，但对长跑训练同样关键。

## 10. 当前结果与最终建议

### 当前结果

截至目前，这条 Omni-VLA + RLinf embodied async PPO 链路已经完成了从“能接上”到“能训练、能判读、能保存”的多轮适配。

可以认为当前已经相对稳定的能力包括：

- Omni-VLA 在 RLinf 中的模型注册、构造、基础 wrapper 与数据配置链路
- 训练启动期的主要兼容性问题
- 显存 / activation / checkpointing 关键开关的基本联动
- SFT checkpoint 权重加载与 key remapping 的关键路径
- async PPO 行为语义对齐
- checkpoint/save 的主要工程稳定性修复

### 仍需保持警惕的风险

以下内容虽然已显著改善，但仍建议保持工程警惕：

- 新配置或新模型版本引入后，`self.training` 相关语义可能再次分叉
- checkpoint/save 仍是多组件同步的高风险区域，后续改 runner 或 worker 时应优先回归
- `norm_stats`、checkpoint key 映射、资产路径这类“看似不是算法问题”的基础设施问题，仍然可能在换 checkpoint 或换数据集时复发

### 最终建议

1. 新同学接手时先读本文，再读 async PPO 专题和指标手册。
2. 每次换模型版本或大配置前，优先回归：
   - 权重加载覆盖率
   - `self.training` 相关语义
   - checkpoint/save 恢复链路
3. 后续若重做 checkpoint 机制，优先把“异步通信 quiesce + runtime reset + channel generation”视作一个整体，而不是分散 patch。

## 11. 关键改动索引

以下改动类别是整个适配过程中最关键的工程抓手：

- `omni_vla` 模型注册、安装与 embodied wrapper 接入
- 预训练路径、attention mask、dtype、属性访问等启动期兼容修复
- `gradient checkpointing` / `use_cache` / activation memory 联动
- SFT checkpoint key remap、`lm_head` / `embed_tokens` 权重映射、norm stats 加载修复
- async PPO 行为语义控制与 logprob 口径对齐
- async checkpoint/save 的 quiesce、graceful stop、channel generation 与 runtime reset

如需更细的 patch 级线索，可参考：

- [`docs/git_changes.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/git_changes.md)
- [`docs/debug_log_20260312.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/debug_log_20260312.md)
- [`docs/debug_log_2026-03-20.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/debug_log_2026-03-20.md)

## 12. 附录导航

### 建议长期保留的专题附录

- [`docs/integration/omni_vla_async_ppo_behavior_semantics_fix_2026-03-27.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/omni_vla_async_ppo_behavior_semantics_fix_2026-03-27.md)
- [`docs/omni_vla_async_ppo_metric_handbook.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/omni_vla_async_ppo_metric_handbook.md)
- [`docs/integration/omni_vla_libero_spatial_ppo_eval_summary_2026-03-31.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/omni_vla_libero_spatial_ppo_eval_summary_2026-03-31.md)

### 已被本文吸收的历史材料

- [`docs/integration/openpi_analysis.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/openpi_analysis.md)
- [`docs/integration/omni_vla_optimization_analysis.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/omni_vla_optimization_analysis.md)
- [`docs/integration/omni_vla_integration_step3.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/omni_vla_integration_step3.md)
- [`docs/debug_log_20260312.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/debug_log_20260312.md)
- [`docs/debug_log_2026-03-20.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/debug_log_2026-03-20.md)
- [`docs/integration/omni_vla_activation_memory_change_log_2026-03-23.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/omni_vla_activation_memory_change_log_2026-03-23.md)
- [`docs/git_changes.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/git_changes.md)
