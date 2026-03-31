# Omni-VLA / RLinf 文档导航

这组文档记录了 `Omni-VLA` 适配 `RLinf` 强化学习框架的完整过程。

如果你是第一次接手这条链路，建议按下面顺序阅读：

1. [`docs/integration/omni_vla_rlinf_integration_full_journey.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/omni_vla_rlinf_integration_full_journey.md)
2. [`docs/integration/omni_vla_async_ppo_behavior_semantics_fix_2026-03-27.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/omni_vla_async_ppo_behavior_semantics_fix_2026-03-27.md)
3. [`docs/omni_vla_async_ppo_metric_handbook.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/omni_vla_async_ppo_metric_handbook.md)
4. [`docs/integration/omni_vla_libero_spatial_ppo_eval_summary_2026-03-31.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/omni_vla_libero_spatial_ppo_eval_summary_2026-03-31.md)

## 文档分层

### 主入口

- [`docs/integration/omni_vla_rlinf_integration_full_journey.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/omni_vla_rlinf_integration_full_journey.md)
  - 面向项目内部协作与后续维护者的总览文档。
  - 覆盖从初始接入、训练启动、显存优化、checkpoint 加载、async PPO 语义问题到 checkpoint/save 工程修复的完整时间线。

### 专题附录

- [`docs/integration/omni_vla_async_ppo_behavior_semantics_fix_2026-03-27.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/omni_vla_async_ppo_behavior_semantics_fix_2026-03-27.md)
  - Async PPO 行为语义错位问题的专题定位与修复记录。
- [`docs/omni_vla_async_ppo_metric_handbook.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/omni_vla_async_ppo_metric_handbook.md)
  - Async PPO 指标观察与训练判读手册。
- [`docs/integration/omni_vla_libero_spatial_ppo_eval_summary_2026-03-31.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/omni_vla_libero_spatial_ppo_eval_summary_2026-03-31.md)
  - `libero_spatial` 上基础模型与 PPO checkpoints 的标准化 eval 结果整理，以及当前最佳 checkpoint 区间和下一轮调参建议。

### 原始排查 / 历史记录

以下文档已被主文档吸收，但仍保留原始细节，适合在需要追 patch 线索或复盘某次误判时查阅：

- [`docs/debug_log_20260312.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/debug_log_20260312.md)
- [`docs/debug_log_2026-03-20.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/debug_log_2026-03-20.md)
- [`docs/integration/omni_vla_activation_memory_change_log_2026-03-23.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/omni_vla_activation_memory_change_log_2026-03-23.md)
- [`docs/git_changes.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/git_changes.md)
- [`docs/integration/openpi_analysis.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/openpi_analysis.md)
- [`docs/integration/omni_vla_optimization_analysis.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/omni_vla_optimization_analysis.md)
- [`docs/integration/omni_vla_integration_step3.md`](/Users/kaelynwang/Desktop/kaelynwang/Omni_VLA/docs/integration/omni_vla_integration_step3.md)
