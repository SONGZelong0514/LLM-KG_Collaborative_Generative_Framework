"""Runtime and GPU-memory monitoring shown in the chat interface."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time

from .config import SETTINGS


class RuntimeGpuMonitor:
    def __init__(self, label, gpu_count=None, poll_interval=None):
        self.label = label
        self.gpu_count = gpu_count if gpu_count is not None else SETTINGS.gpu_count
        self.poll_interval = poll_interval if poll_interval is not None else SETTINGS.gpu_poll_interval
        self.start_time = None
        self.peak_memory = [None] * self.gpu_count
        self.peak_total_memory = None
        self.error = None
        self._stop_event = threading.Event()
        self._thread = None

    def _query_gpu_memory(self):
        if shutil.which("nvidia-smi") is None:
            raise RuntimeError("nvidia-smi not found")
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        memory = [None] * self.gpu_count
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                index, used_mb = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if 0 <= index < self.gpu_count:
                memory[index] = used_mb
        return memory

    def _record_sample(self):
        memory = self._query_gpu_memory()
        for index, used_mb in enumerate(memory):
            if used_mb is not None and (
                self.peak_memory[index] is None or used_mb > self.peak_memory[index]
            ):
                self.peak_memory[index] = used_mb
        if all(used_mb is not None for used_mb in memory):
            total = sum(memory)
            if self.peak_total_memory is None or total > self.peak_total_memory:
                self.peak_total_memory = total

    def _sample_loop(self):
        while not self._stop_event.wait(self.poll_interval):
            try:
                self._record_sample()
            except Exception as exc:
                self.error = str(exc)
                return

    def start(self):
        self.start_time = time.perf_counter()
        try:
            self._record_sample()
        except Exception as exc:
            self.error = str(exc)
            return self
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        elapsed = time.perf_counter() - self.start_time if self.start_time is not None else 0.0
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self.error is None:
            try:
                self._record_sample()
            except Exception as exc:
                self.error = str(exc)
        return elapsed

    def summary_text(self):
        elapsed = self.stop()
        lines = ["", f"⏱️ **{self.label} runtime:** {elapsed:.2f} s"]
        if self.error:
            lines.append(f"🖥️ **GPU memory peak:** unavailable ({self.error})")
            return "\n".join(lines)

        lines.append("🖥️ **GPU memory peak:**")
        if self.peak_total_memory is None:
            lines.append(f"- Total (GPU 0-{self.gpu_count - 1}): unavailable")
        else:
            lines.append(
                f"- Total (GPU 0-{self.gpu_count - 1}): peak {self.peak_total_memory / 1024:.2f} GiB"
            )
        for index, peak in enumerate(self.peak_memory):
            value = "unavailable" if peak is None else f"peak {peak / 1024:.2f} GiB"
            lines.append(f"- GPU {index}: {value}")
        return "\n".join(lines)


def append_metrics(history, monitor):
    metrics = monitor.summary_text()
    print(metrics)
    history[-1]["content"] += f"\n\n{metrics}"
    return history

