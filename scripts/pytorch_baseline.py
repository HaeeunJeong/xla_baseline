#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytorch_baseline.py – Universal PyTorch latency & memory probe using model *block* modules.
"""

from __future__ import annotations

import argparse, csv, importlib, time, sys
from datetime import datetime
from pathlib import Path
from typing import Any, List

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
ROOT_DIR   = Path(__file__).resolve().parent.parent
MODEL_DIR  = ROOT_DIR / "models"
UTIL_DIR   = ROOT_DIR / "utils"
RESULT_DIR = ROOT_DIR / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _init_dir() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _discover_models() -> List[str]:
    return sorted(p.stem for p in MODEL_DIR.glob("*_block.py") if p.is_file())


def _format_line(name: str,
                 device: str,
                 avg: float,
                 mn: float,
                 mx: float,
                 mem_mb: float,
                 name_w: int) -> str:
    """
    Nicely aligned single-line summary.
    """
    return (f"{name:<{name_w}s} | {device:<4s} | "
            f"{avg:>10.6f} / {mn:>10.6f} / {mx:>10.6f} ms | "
            f"{mem_mb:>8.2f} MB")


def _resolve_device(requested: str) -> str:
    """
    Determine actual device to use.
    - If requested is 'cpu' -> 'cpu'
    - If requested is 'cuda' but CUDA unavailable -> fall back to 'cpu'
    - Else -> 'cuda'
    """
    if requested == "cpu":
        return "cpu"
    # requested == "cuda"
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        # Informative note; keep stdout minimal.
        print("[INFO] CUDA is unavailable, fallback to CPU.")
        return "cpu"
    return "cuda"


# ---------------------------------------------------------------------------
# Model Loader – 모든 로직을 각 block 내부로 위임
# ---------------------------------------------------------------------------
def load_model(name: str):
    """
    Import `models/<name>_block.py` and return (model, dummy_input).

    각 block **must** implement:
        get_model()        -> torch.nn.Module (eval mode는 block 쪽 책임)
        get_dummy_input()  -> Tensor | tuple[Tensor,...] | shape-tuple[int,...]
    """
    mod_name = f"{name.lower()}_block" if not name.lower().endswith("_block") else name.lower()
    mod = importlib.import_module(f"models.{mod_name}")
    return mod.get_model(), mod.get_dummy_input()


# ---------------------------------------------------------------------------
# Forward-pass timer & memory probe
# ---------------------------------------------------------------------------
def _measure(model: torch.nn.Module,
             call,
             device_obj: torch.device,
             repeats: int = 10) -> tuple[float, float, float, float]:
    """
    Return (avg_latency_ms, min_latency_ms, max_latency_ms, peak_mem_MB).

    * Latency: pure compute – H2D / D2H traffic excluded.
    * Memory : peak GPU memory during the measured repetitions (MB, 0 on CPU).
    """
    # peak-mem bookkeeping (GPU only)
    if device_obj.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_obj)

    latencies: list[float] = []

    # --------------------------- measurement -------------------------------
    torch.cuda.nvtx.range_push(f"inference_loop: {model.__class__.__name__}") if device_obj.type == "cuda" else None
    with torch.no_grad():
        for _ in range(repeats):
            start = time.perf_counter()
            out = call()
            if device_obj.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)
        if device_obj.type == "cuda":
            torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop() if device_obj.type == "cuda" else None

    # D2H copy *after* timing so it is excluded
    if isinstance(out, torch.Tensor):
        _ = out.cpu()
    elif isinstance(out, (tuple, list)):
        _ = [o.cpu() if isinstance(o, torch.Tensor) else o for o in out]

    peak_mem = (torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2)
                if device_obj.type == "cuda" else 0.0)

    avg_ms = sum(latencies) / len(latencies)
    return avg_ms, min(latencies), max(latencies), peak_mem


def time_forward(model: torch.nn.Module,
                 dummy_input: Any,
                 device: str = "cuda",
                 warmup: int = 10,
                 repeats: int = 10) -> tuple[float, float, float, float]:
    """
    Return (avg, min, max latency ms, peak_mem_MB).

    * warm-up: `warmup` forward passes (no timing, excluded from stats)
    * measured repetitions: `repeats`, averaged
    """
    device_obj = torch.device(device)
    model = model.to(device_obj).eval()

    # ------------------------- 입력을 device로 이동 --------------------------
    if isinstance(dummy_input, torch.Tensor):
        inp = dummy_input.to(device_obj)
        call = lambda: model(inp)

    elif isinstance(dummy_input, tuple):
        # ➊ tuple of Tensors: Vision/GNN/LLM positional 입력
        if all(isinstance(t, torch.Tensor) for t in dummy_input):
            tensors = [t.to(device_obj) for t in dummy_input]
            call = lambda: model(*tensors)
        # ➋ tuple of ints: shape 지정 → 랜덤 Tensor 생성
        else:
            x = torch.randn(*dummy_input, device=device_obj)
            call = lambda: model(x)

    else:
        raise TypeError("dummy_input must be Tensor or Tuple")

    # --------------------------- warm-up ------------------------------------
    torch.cuda.nvtx.range_push("warmup") if device_obj.type == "cuda" else None
    with torch.no_grad():
        for _ in range(warmup):
            call()
        if device_obj.type == "cuda":
            torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop() if device_obj.type == "cuda" else None

    # ----------------------- timed measurement ------------------------------
    return _measure(model, call, device_obj, repeats=repeats)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="*", help="model keys; omit for ALL")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--csv_path", default=None,
                        help="Custom CSV path. If omitted, auto-generated.")
    parser.add_argument("--repeats", type=int, default=10,
                        help="How many timed repetitions (avg/min/max).")
    parser.add_argument("--warmup", type=int, default=10,
                        help="How many warm-up iterations (no timing).")
    args = parser.parse_args()

    _init_dir()

    # ---- device resolve (CUDA 미가용 시 자동 CPU 폴백) ----------------------
    actual_device = _resolve_device(args.device)

    targets = args.model if args.model else _discover_models()
    name_width = max(len(n) for n in targets) + 2  # dynamic padding
    hdr = ("model".ljust(name_width) +
           " | dev  |   avg /    min /    max (ms) |  peak MB")
    print("─" * (len(hdr)))
    print(hdr)
    print("─" * (len(hdr)))

    timestamp = datetime.now().isoformat(timespec="seconds")
    csv_header = ["timestamp",
                  "model_name",
                  "device",
                  "warmup",
                  "repeats",
                  "latency_avg_ms",
                  "latency_min_ms",
                  "latency_max_ms",
                  "peak_memory_mb"]
    csv_rows: list[list[str]] = []

    for name in targets:
        try:
            mdl, dummy = load_model(name)
            avg, mn, mx, mem_mb = time_forward(
                mdl, dummy, actual_device, warmup=args.warmup, repeats=args.repeats
            )
            print(_format_line(name, actual_device, avg, mn, mx, mem_mb, name_width))
            csv_rows.append([timestamp, name, actual_device,
                             str(args.warmup), str(args.repeats),
                             f"{avg:.6f}", f"{mn:.6f}", f"{mx:.6f}", f"{mem_mb:.2f}"])
        except Exception as exc:
            print(f"[ERROR] {name}: {exc}")
            csv_rows.append([timestamp, name, actual_device,
                             str(args.warmup), str(args.repeats),
                             "ERROR", "ERROR", "ERROR", str(exc)])

    # --------------------------- CSV dump -----------------------------------
    tag = "all" if not args.model else "_".join(args.model)
    default_path = RESULT_DIR / f"metrics_{tag}_{actual_device}.csv"
    path = Path(args.csv_path) if args.csv_path else default_path
    path.parent.mkdir(parents=True, exist_ok=True)

    # always create fresh file → header included as 1행
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)
    print("─" * (len(hdr)))
    print(f"[✓] Metrics saved to {path}")


if __name__ == "__main__":
    main()
