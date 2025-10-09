"""Utilities for logging system stats during training."""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers import TrainerCallback, TrainerState, TrainerControl

try:  # optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


_logger = logging.getLogger(__name__)


def _bytes_to_mebibytes(value: float) -> float:
    return value / (1024.0 ** 2)


def _collect_cpu_rss() -> Optional[float]:
    if psutil is None:
        return None
    try:
        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss)
    except Exception:  # pragma: no cover
        return None


@dataclass
class _OffloadSnapshot:
    total_params: int
    total_bytes: int
    device_bytes: Dict[str, int]


class StepSystemStatsCallback(TrainerCallback):
    """Logs per-step timing, CPU/GPU memory, and parameter offload."""

    def __init__(self, log_interval: int = 1) -> None:
        self.log_interval = max(1, log_interval)
        self._step_start_ts: Optional[float] = None
        self._offload_snapshot: Optional[_OffloadSnapshot] = None
        self._last_wandb_step: Optional[int] = None

    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._step_start_ts = time.perf_counter()
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(idx)
        return control

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        step = state.global_step
        if step % self.log_interval != 0:
            return control

        step_time = None
        if self._step_start_ts is not None:
            step_time = time.perf_counter() - self._step_start_ts

        model = kwargs.get("model")
        if model is not None and self._offload_snapshot is None:
            self._offload_snapshot = self._capture_offload(model)

        stats = {
            "step/global": step,
            "step/duration_sec": step_time,
        }

        stats["monitor/global_step"] = step

        cpu_rss = _collect_cpu_rss()
        if cpu_rss is not None:
            stats["cpu/rss_mb"] = _bytes_to_mebibytes(cpu_rss)

        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                prefix = f"gpu/{idx}"
                stats[f"{prefix}/allocated_mb"] = _bytes_to_mebibytes(torch.cuda.memory_allocated(idx))
                stats[f"{prefix}/reserved_mb"] = _bytes_to_mebibytes(torch.cuda.memory_reserved(idx))
                stats[f"{prefix}/peak_mb"] = _bytes_to_mebibytes(torch.cuda.max_memory_allocated(idx))

        if self._offload_snapshot is not None:
            total_bytes = self._offload_snapshot.total_bytes or 1
            offload_bytes = sum(
                bytes_used
                for device, bytes_used in self._offload_snapshot.device_bytes.items()
                if not device.startswith("cuda")
            )
            stats["model/params"] = self._offload_snapshot.total_params
            stats["model/bytes_mb"] = _bytes_to_mebibytes(self._offload_snapshot.total_bytes)
            stats["model/offloaded_mb"] = _bytes_to_mebibytes(offload_bytes)
            stats["model/offload_ratio"] = offload_bytes / total_bytes

        _logger.info(
            "step %s | time=%.3fs | cpu=%.1fMB | gpu=%s | offload=%.2f%%",
            step,
            stats.get("step/duration_sec") or 0.0,
            stats.get("cpu/rss_mb") or 0.0,
            ", ".join(
                f"d{idx}:{stats.get(f'gpu/{idx}/allocated_mb', 0.0):.1f}MB"
                for idx in range(torch.cuda.device_count())
            )
            if torch.cuda.is_available()
            else "n/a",
            stats.get("model/offload_ratio", 0.0) * 100.0,
        )

        try:
            import wandb  # type: ignore

            if wandb.run is not None:
                run_step = getattr(wandb.run, "step", None)
                log_step = step
                if self._last_wandb_step is not None:
                    log_step = max(log_step, self._last_wandb_step + 1)
                if run_step is not None:
                    log_step = max(log_step, int(run_step))
                wandb.log(stats, step=log_step)
                self._last_wandb_step = log_step
        except Exception:  # pragma: no cover
            pass

        return control

    def _capture_offload(self, model) -> _OffloadSnapshot:
        device_bytes: Dict[str, int] = defaultdict(int)
        total_params = 0
        total_bytes = 0

        for param in model.parameters():
            if param is None:
                continue
            numel = param.numel()
            total_params += numel
            param_bytes = numel * param.element_size()
            total_bytes += param_bytes
            device_name = str(param.device)
            device_bytes[device_name] += param_bytes

        for buf in model.buffers():
            if buf is None:
                continue
            numel = buf.numel()
            param_bytes = numel * buf.element_size()
            total_bytes += param_bytes
            device_name = str(buf.device)
            device_bytes[device_name] += param_bytes

        return _OffloadSnapshot(
            total_params=total_params,
            total_bytes=total_bytes,
            device_bytes=dict(device_bytes),
        )