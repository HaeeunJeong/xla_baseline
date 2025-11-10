#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compile_xla.py – Run benchmarks on **PyTorch XLA back-end** (TPU / XLA-CPU /
PJRT-GPU). 10-step warm-up + 10-step measurement, avg/min/max latency,
(가능할 경우) peak GPU 메모리 기록, CSV 저장.
"""

from __future__ import annotations

import argparse, csv, os, sys, time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple, Union


# ──────────────────────────────────────────────────────────────────────
# repo paths
# ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT_DIR), str(ROOT_DIR / "scripts")])

from scripts.pytorch_baseline import load_model, _discover_models

RESULTS_DIR = ROOT_DIR / "results" / "xla"
DUMP_DIR = RESULTS_DIR / "dump_hlo"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DUMP_DIR.mkdir(parents=True, exist_ok=True)




# ──────────────────────────────────────────────────────────────────────
# PJRT 환경변수 – torch_xla import **이전** 설정
# ──────────────────────────────────────────────────────────────────────
# os.environ["PJRT_DEVICE"] = "CUDA"   # GPU(PJRT) 사용

# 필요시 메모리 풀 축소:
os.environ["XLA_PJRT_GPU_ALLOCATOR"] = "platform"  # 또는 "default"

os.environ["XLA_PJRT_GPU_PINNED_MEMORY_POOL_SIZE"] = "256M" 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = (
                            # f"--xla_gpu_enable_triton_gemm=true"
                            f"--xla_dump_to={DUMP_DIR}/proto "
                            f"--xla_dump_hlo_pass_re=.* "
                            # f"--xla_dump_hlo_as_text "
                             "--xla_dump_hlo_as_proto "
                             # "--xla_dump_hlo_as_dot=true"
                           # --xla_hlo_profile -> unsupported on GPU
                            )


import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────
def _memory_used_mb(device) -> Union[float, None]:
    """
    Return *used* memory on XLA device in MB or None if unsupported.

    TPU          → dict{'kb_free','kb_total'}
    PJRT-GPU     → tuple(kb_free, kb_total)   (torch-xla 2.3)
    Unavailable  → RuntimeError or dict missing keys
    """
    try:
        info = xm.get_memory_info(device)
        # print(info)
    except Exception as e: 
        print(f"[MemInfo Error] {type(e).__name__}: {e}", file=sys.stderr)
        return None

    # PJRT-GPU
    if isinstance(info, dict) and "bytes_used" in info:
        return info["bytes_used"] / 1024.0 / 1024.0  # bytes to MB

    # TPU
    if isinstance(info, dict) and "kb_total" in info:
        return ((info["kb_total"] - info["kb_free"]) / 1024.0 )  # bytes to MB

    # TPU SPMD
    if isinstance(info, (tuple, list)) and len(info) == 2:
        kb_free, kb_total = info
        return ((kb_total - kb_free) / 1024.0 )  # bytes to MB

    
    print(f"[MemInfo Error] Unexpected format: {type(info)}", file=sys.stderr)
    return None

    return used_kb / 1024.0


def _fmt_line(name: str,
              device: str,
              avg: float,
              mn: float,
              mx: float,
              mem: Union[float, None],
              pad: int) -> str:
    mem_str = f"{mem:8.2f} MB" if mem is not None else "   n/a"
    return (f"{name:<{pad}s} | {device:<7s} | "
            f"{avg:>10.3f} / {mn:>10.3f} / {mx:>10.3f} ms | {mem_str}")


def _make_inputs(dummy: Any, device):
    if isinstance(dummy, tuple):
        if all(isinstance(t, torch.Tensor) for t in dummy):
            return tuple(t.to(device) for t in dummy)
        return (torch.randn(*dummy, device=device),)
    if isinstance(dummy, dict):
        return {k: v.to(device) for k, v in dummy.items()}
    return (dummy.to(device),)


# ──────────────────────────────────────────────────────────────────────
# measurement
# ──────────────────────────────────────────────────────────────────────
def _measure(model: torch.nn.Module,
             dummy: Any,
             *,
             device,
             warmup: int,
             repeats: int) -> Tuple[float, float, float, Union[float, None]]:
    model.to(device).eval()
    inputs = _make_inputs(dummy, device)

    # print("Device memory info:")
    # print(xm.get_memory_info())
    

    def _call():
        with torch.no_grad():
            if isinstance(inputs, dict):
                out = model(**inputs)
            elif isinstance(inputs, tuple):
                out = model(*inputs)
            else:
                out = model(inputs)
        return out

    # ─ warm-up (graph compile) ─────────────────────────────────────────
    for _ in range(warmup):
        _ = _call(); torch_xla.sync()
    xm.wait_device_ops()

    # peak_mem = _memory_used_mb(device)  # None if unsupported
    latencies: List[float] = []

    # ─ measurement ─────────────────────────────────────────────────────
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = _call(); torch_xla.sync()
        xm.wait_device_ops()
        latencies.append((time.perf_counter() - t0) * 1000.0)

        # cur = _memory_used_mb(device)
        # if peak_mem is not None and cur is not None:
            # peak_mem = max(peak_mem, cur)
        peak_mem = 0

    # ─ copy result to CPU (exclude comm cost) ──────────────────────────
    if isinstance(out, torch.Tensor):
        out.cpu()
    elif isinstance(out, (tuple, list)):
        [o.cpu() for o in out if isinstance(o, torch.Tensor)]

    avg_ms = sum(latencies) / len(latencies)
    return avg_ms, min(latencies), max(latencies), peak_mem


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="*", help="model keys; omit for ALL")
    ap.add_argument("--device", default="xla:1",
                    help="XLA device spec, e.g. 'xla', 'xla:0', 'xla:1'")
    ap.add_argument("--warmup", type=int, default=10, help="# warm-up steps")
    ap.add_argument("--repeats", type=int, default=10, help="# measured steps")
    ap.add_argument("--csv_path", help="Custom CSV path")
    args = ap.parse_args()

    # device = xm.xla_device() if args.device.startswith("xla") else torch.device(args.device)
    device = torch_xla.device() if args.device.startswith("xla") else torch.device(args.device)
    models = args.model if args.model else _discover_models()
    pad = max(len(m) for m in models) + 2

    header = ("model".ljust(pad) +
              " | device  |    avg /     min /     max (ms) | peak MB")
    print("─" * len(header))
    print(header)
    print("─" * len(header))

    ts = datetime.now().isoformat(timespec="seconds")
    csv_header = ["timestamp", "model_name", "device",
                  "warmup", "repeats",
                  "latency_avg_ms", "latency_min_ms", "latency_max_ms",
                  "peak_memory_mb"]
    rows: List[List[str]] = []

    for name in models:
        try:
            mdl, dummy = load_model(name)
            avg, mn, mx, mem = _measure(
                mdl, dummy, device=device,
                warmup=args.warmup, repeats=args.repeats)
            print(_fmt_line(name, str(device), avg, mn, mx, mem, pad))
            rows.append([ts, name, str(device),
                         args.warmup, args.repeats,
                         f"{avg:.3f}", f"{mn:.3f}", f"{mx:.3f}",
                         f"{mem:.2f}" if mem is not None else "n/a"])
        except Exception as e:
            reason = str(e).splitlines()[0]
            print(f"[ERROR] {name}: {reason}")
            rows.append([ts, name, str(device),
                         args.warmup, args.repeats,
                         "ERROR", "ERROR", "ERROR", reason])

    # ─ CSV 저장 ─────────────────────────────────────────────────────────
    tag = "all" if not args.model else "_".join(args.model)
    default_csv = RESULTS_DIR / f"xla_metrics_{tag}_{device}.csv"
    csv_path = Path(args.csv_path) if args.csv_path else default_csv
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(rows)

    print("─" * len(header))
    print(f"[✓] CSV saved → {csv_path}")


if __name__ == "__main__":
    main()

