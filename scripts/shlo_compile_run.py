#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compile_xla.py – Run benchmarks on **PyTorch-XLA StableHLO bundles**  
(TPU / XLA-CPU / PJRT-GPU). 10-step warm-up + 10-step measurement,  
avg/min/max latency, CSV 저장.

(메모리 사용량 측정 기능 제거됨)
"""

from __future__ import annotations

import argparse, csv, os, re, sys, time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

# ──────────────────────── 0) 조기 인자파싱: --dump 여부만 판단 ────────────
early_parser = argparse.ArgumentParser(add_help=False)
early_parser.add_argument("--dump", action="store_true")
EARLY_ARGS, _ = early_parser.parse_known_args()

# ───────────────────────────────────────────── repo & 결과 디렉터리 설정 ──
ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results" / "xla"
STABLE_DIR = RESULTS_DIR / "StableHLO"
DUMP_DIR = RESULTS_DIR / "dump_shlo/proto"
for d in (RESULTS_DIR, STABLE_DIR, DUMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

sys.path.extend([str(ROOT_DIR), str(ROOT_DIR / "scripts")])
from scripts.pytorch_eager import load_model, _discover_models  # noqa: E402

# ───────────────────────────────────────────── PJRT & XLA 환경변수 설정 ──
os.environ.setdefault("PJRT_DEVICE", "CPU")
os.environ.setdefault("XLA_PJRT_GPU_ALLOCATOR", "platform")
os.environ.setdefault("XLA_PJRT_GPU_PINNED_MEMORY_POOL_SIZE", "256M")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")

if EARLY_ARGS.dump:
    os.environ["XLA_FLAGS"] = (
        f"--xla_dump_to={DUMP_DIR} "
        f"--xla_dump_hlo_pass_re=.* "
        "--xla_dump_hlo_as_proto "
    )

import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.stablehlo import StableHLOGraphModule

import torch_xla.debug.metrics as met
import numpy as np

# ───────────────────────────────────────────── Helper Functions ───────────

_NUMPY_MAGIC = b"\x93NUMPY"

_SHLO_DTYPE_MAP = {
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "bf16": np.uint16,  # bfloat16 → 2-byte raw
}

# Pattern: __constant_{dims}x{dtype}  optionally followed by _{digit_suffix}
# Examples: __constant_16x10xf32, __constant_xf32 (scalar), __constant_1x1x32x32xi8_0
_CONST_RE = re.compile(
    r"^__constant_"
    r"(?P<shape>(?:\d+x)*)"  # shape dims: '16x10x' or '' (scalar)
    r"(?:x(?=\D))?"  # consume a bare 'x' before dtype (scalar case)
    r"(?P<dtype>[a-z]+\d+)"  # dtype: 'f32', 'i64', etc.
    r"(?:_\d+)?$"  # optional disambiguating suffix: '_0'
)


def _normalize_data_dir(data_dir: Path) -> None:
    """Convert raw-binary __constant_* files to numpy .npy in-place."""
    if not data_dir.is_dir():
        return
    for fpath in sorted(data_dir.iterdir()):
        if not fpath.is_file() or not fpath.name.startswith("__constant"):
            continue
        # Skip files that already have the numpy magic header
        with fpath.open("rb") as f:
            header = f.read(6)
        if header == _NUMPY_MAGIC:
            continue

        m = _CONST_RE.match(fpath.name)
        if m is None:
            print(f"  [WARN] cannot parse constant filename: {fpath.name}")
            continue

        shape_str = m.group("shape")  # e.g. '16x10x' or '' (scalar)
        dtype_str = m.group("dtype")  # e.g. 'f32'

        np_dtype = _SHLO_DTYPE_MAP.get(dtype_str)
        if np_dtype is None:
            print(f"  [WARN] unknown dtype '{dtype_str}' in {fpath.name}")
            continue

        shape: tuple
        if shape_str:
            shape = tuple(int(d) for d in shape_str.rstrip("x").split("x"))
        else:
            shape = ()  # scalar

        raw = fpath.read_bytes()
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()
        # np.save(str) appends .npy; use open file handle to overwrite in-place
        with fpath.open("wb") as f:
            np.save(f, arr)
        # Remove stale .npy duplicate if a previous run created one
        dup = fpath.parent / (fpath.name + ".npy")
        if dup.exists():
            dup.unlink()


def _fmt_line(name, device, avg, mn, mx, pad):
    return (
        f"{name:<{pad}s} | {device:<7s} | "
        f"{avg:>10.3f} / {mn:>10.3f} / {mx:>10.3f} ms"
    )


def _make_inputs(dummy: Any, device):
    if isinstance(dummy, tuple):
        if all(isinstance(t, torch.Tensor) for t in dummy):
            return tuple(t.to(device) for t in dummy)
        return (torch.randn(*dummy, device=device),)
    if isinstance(dummy, dict):
        return {k: v.to(device) for k, v in dummy.items()}
    return (dummy.to(device),)


def _measure(
    runner, dummy, *, device, warmup: int, repeats: int
) -> Tuple[float, float, float]:

    if hasattr(runner, "to"):
        runner = runner.to(device)
    if hasattr(runner, "eval"):
        runner.eval()

    inputs = _make_inputs(dummy, device)

    call_fn = lambda: runner(*inputs)

    # warm-up = compile
    for _ in range(warmup):
        call_fn()
        torch_xla.sync()
    xm.wait_device_ops()

    met.clear_counters()
    torch_xla._XLAC._clear_pending_irs(str(device))

    t0 = time.perf_counter()
    for _ in range(repeats):
        call_fn()
        torch_xla.sync()
    xm.wait_device_ops()
    total_ms = (time.perf_counter() - t0) * 1000.0

    avg_ms = total_ms / repeats
    return avg_ms, avg_ms, avg_ms  # min/max = avg (동일 측정구조)


# ──────────────────────────────────────────────── main ────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="*", help="model keys; omit for ALL")
    parser.add_argument("--device", default="xla:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--csv_path")
    parser.add_argument("--dump", action="store_true")
    args = parser.parse_args()

    device = (
        torch_xla.device()
        if args.device.startswith("xla")
        else torch.device(args.device)
    )

    orig_models = args.model or _discover_models()
    models = [m[:-6] if m.endswith("_block") else m for m in orig_models]
    pad = max(len(m) for m in models) + 2

    hdr = "model".ljust(pad) + " | device  |    avg /     min /     max (ms)"
    print("─" * len(hdr))
    print(hdr)
    print("─" * len(hdr))

    ts = datetime.now().isoformat(timespec="seconds")
    csv_header = [
        "timestamp",
        "model_name",
        "device",
        "input_path",
        "warmup",
        "repeats",
        "latency_avg_ms",
        "latency_min_ms",
        "latency_max_ms",
    ]
    rows: List[List[str]] = []

    for name in models:
        shlo_dir = STABLE_DIR / f"{name}_stablehlo"
        if not shlo_dir.is_dir():
            print(f"[SKIP] {name}: StableHLO dir not found")
            continue

        try:
            _normalize_data_dir(shlo_dir / "data")
            _normalize_data_dir(shlo_dir / "constants")
            _, dummy = load_model(name)
            runner = StableHLOGraphModule.load(str(shlo_dir))
            avg, mn, mx = _measure(
                runner, dummy, device=device, warmup=args.warmup, repeats=args.repeats
            )

            print(_fmt_line(name, str(device), avg, mn, mx, pad))
            rows.append(
                [
                    ts,
                    name,
                    str(device),
                    str(shlo_dir.resolve()),
                    args.warmup,
                    args.repeats,
                    f"{avg:.3f}",
                    f"{mn:.3f}",
                    f"{mx:.3f}",
                ]
            )
        except Exception as exc:
            reason = str(exc).splitlines()[0]
            print(f"[ERROR] {name}: {reason}")
            rows.append(
                [
                    ts,
                    name,
                    str(device),
                    str(shlo_dir.resolve()),
                    args.warmup,
                    args.repeats,
                    "ERROR",
                    "ERROR",
                    "ERROR",
                ]
            )

    tag = "all" if not args.model else "_".join(models)
    defaultcsv = RESULTS_DIR / f"xla_metrics_{tag}_{device}.csv"
    path = Path(args.csv_path) if args.csv_path else defaultcsv
    with path.open("w", newline="") as f:
        csv.writer(f).writerow(csv_header)
        csv.writer(f).writerows(rows)

    print("─" * len(hdr))
    print(f"[✓] CSV saved → {path}")


if __name__ == "__main__":
    main()
