from __future__ import annotations

from typing import Any

import numpy as np
import torch


def get_chain_trace_config(cfg) -> dict[str, Any]:
    algorithm_cfg = cfg.get("algorithm", cfg)
    thresholds_cfg = algorithm_cfg.get("debug_chain_trace_thresholds", {})
    return {
        "enabled": bool(algorithm_cfg.get("debug_chain_trace", False)),
        "mode": str(algorithm_cfg.get("debug_chain_trace_mode", "anomaly")).lower(),
        "topk": int(algorithm_cfg.get("debug_chain_trace_topk", 3)),
        "rank0_only": bool(algorithm_cfg.get("debug_chain_trace_rank0_only", True)),
        "thresholds": {
            "behav_approx_kl": float(
                thresholds_cfg.get("behav_approx_kl", 20.0)
            ),
            "prox_old_gap_abs_mean": float(
                thresholds_cfg.get("prox_old_gap_abs_mean", 20.0)
            ),
            "train_rollout_gap_abs_mean": float(
                thresholds_cfg.get("train_rollout_gap_abs_mean", 0.5)
            ),
        },
    }


def get_pipeline_trace_config(cfg) -> dict[str, Any]:
    algorithm_cfg = cfg.get("algorithm", cfg)
    return {
        "enabled": bool(algorithm_cfg.get("debug_pipeline_trace", False)),
        "rank0_only": bool(algorithm_cfg.get("debug_pipeline_trace_rank0_only", True)),
        "log_every": max(int(algorithm_cfg.get("debug_pipeline_trace_log_every", 1)), 1),
        "max_items": max(int(algorithm_cfg.get("debug_pipeline_trace_max_items", 3)), 1),
    }


def should_enable_chain_trace(cfg, rank: int) -> bool:
    trace_cfg = get_chain_trace_config(cfg)
    if not trace_cfg["enabled"]:
        return False
    if trace_cfg["rank0_only"] and int(rank) != 0:
        return False
    return True


def should_enable_pipeline_trace(cfg, rank: int) -> bool:
    trace_cfg = get_pipeline_trace_config(cfg)
    if not trace_cfg["enabled"]:
        return False
    if trace_cfg["rank0_only"] and int(rank) != 0:
        return False
    return True


def scalarize(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().item())
    return float(value)


def tensor_stats_str(
    tensor: torch.Tensor | None,
    *,
    max_values: int = 6,
) -> str:
    if tensor is None:
        return "None"
    detached = tensor.detach().float().cpu()
    flat = detached.reshape(-1)
    head = ", ".join(f"{value:.6f}" for value in flat[:max_values].tolist())
    return (
        f"shape={tuple(detached.shape)}, "
        f"mean={detached.mean().item():.6f}, "
        f"std={detached.std(unbiased=False).item():.6f}, "
        f"min={detached.min().item():.6f}, "
        f"max={detached.max().item():.6f}, "
        f"head=[{head}]"
    )


def rowwise_abs_mean(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    detached = tensor.detach().float()
    if detached.ndim <= 1:
        return detached.abs()
    return detached.abs().reshape(detached.shape[0], -1).mean(dim=1)


def select_topk_indices(scores: torch.Tensor, topk: int) -> list[int]:
    if scores.numel() == 0 or topk <= 0:
        return []
    k = min(int(topk), int(scores.numel()))
    return torch.topk(scores, k=k).indices.detach().cpu().tolist()


def trace_id_to_str(trace_id: Any) -> str:
    if trace_id is None:
        return "none"
    if torch.is_tensor(trace_id):
        if trace_id.numel() == 0:
            return "none"
        return str(int(trace_id.detach().reshape(-1)[0].item()))
    return str(int(trace_id))


def format_block(title: str, lines: list[str]) -> str:
    body = "\n".join(lines)
    return f"[CHAIN TRACE] {title}\n{body}"


def is_anomalous_value(value: float | None, threshold: float) -> bool:
    return value is not None and value >= float(threshold)


def summarize_value(value: Any, *, max_items: int = 3) -> str:
    if value is None:
        return "None"
    if torch.is_tensor(value):
        return tensor_stats_str(value, max_values=max_items)
    if isinstance(value, dict):
        keys = list(value.keys())
        parts = [f"keys={keys[:max_items]}"]
        for key in keys[:max_items]:
            parts.append(f"{key}={summarize_value(value[key], max_items=max_items)}")
        return "; ".join(parts)
    if isinstance(value, (list, tuple)):
        preview = ", ".join(summarize_value(v, max_items=max_items) for v in list(value)[:max_items])
        return f"len={len(value)}, head=[{preview}]"
    if isinstance(value, np.ndarray):
        return (
            f"shape={value.shape}, mean={float(value.mean()):.6f}, "
            f"std={float(value.std()):.6f}, min={float(value.min()):.6f}, max={float(value.max()):.6f}"
        )
    if isinstance(value, (int, float, bool, str)):
        return str(value)
    return type(value).__name__
