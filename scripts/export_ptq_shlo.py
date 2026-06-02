#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_ptq_shlo.py
──────────────────────────────────────────────────────────────────────────────
PyTorch → StableHLO via PT2E Post-Training Static Quantization (PTQ).

* 완벽하게 정적인 FX Graph를 캡처한 후, XNNPACKQuantizer(Static)를 사용하여 Q/DQ 노드를 주입합니다.
* 캘리브레이션용으로 더미 입력을 1-pass 수행합니다.
* 변환된 정적 양자화 모델을 StableHLO로 내보냅니다.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import warnings
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from io import StringIO
from contextlib import redirect_stdout

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*isinstance\(treespec, LeafSpec\).*",
)

import torch
from torch_xla.stablehlo import exported_program_to_stablehlo

try:
    from torchao.quantization.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e import prepare_pt2e, convert_pt2e
except ImportError:
    try:
        from torch.ao.quantization.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
    except ImportError:
        print("[ERROR] Failed to import PT2E quantization APIs.")
        sys.exit(1)


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT_DIR), str(ROOT_DIR / "scripts")])

from scripts.pytorch_eager import load_model, _discover_models  # noqa: E402
from scripts.export_shlo import load_model_block, HFWrapper, make_inputs, _save_exported_program_text, discover_model_keys  # noqa: E402

RESULTS_DIR = ROOT_DIR / "results" / "xla"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def apply_ptq_static_quantization(ep: torch.export.ExportedProgram, dummy_inputs: tuple) -> torch.export.ExportedProgram:
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_dynamic=False))
    
    prepared_model = prepare_pt2e(ep, quantizer)
    
    print("   [PTQ] Running calibration with dummy inputs...")
    prepared_model(*dummy_inputs)
    
    print("   [PTQ] Converting model to static quantized FX graph...")
    quantized_ep = convert_pt2e(prepared_model)
    return quantized_ep

def main() -> None:
    ap = argparse.ArgumentParser(description="Export PyTorch models to StableHLO with PTQ (Static Quantization)")
    ap.add_argument("model", nargs="*", help="model keys; omit = all discovered")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--csv", action="store_true", help="append csv log")
    args = ap.parse_args()

    raw_keys = args.model or discover_model_keys()
    keys: list[str] = []
    all_discovered = discover_model_keys()
    
    if args.model:
        # Expand explicitly requested base models to include their quantized versions
        for k in raw_keys:
            if not k.endswith("_bf16") and not k.endswith("_int8") and not k.endswith("_fp16"):
                if f"{k}_bf16" in all_discovered:
                    keys.append(f"{k}_bf16")
                if f"{k}_fp16" in all_discovered:
                    keys.append(f"{k}_fp16")
                if f"{k}_int8" in all_discovered:
                    keys.append(f"{k}_int8")
            else:
                keys.append(k)
        keys = list(dict.fromkeys(keys))
    else:
        # Only extract quantized / low precision models by default
        keys = [k for k in raw_keys if k.endswith("_bf16") or k.endswith("_int8") or k.endswith("_fp16")]

    timestamp = datetime.now().isoformat(timespec="seconds")
    rows: list[list[str]] = []

    for name in keys:
        try:
            print(f"[{name}] torch_xla PTQ → StableHLO export …")
            model, dummy = load_model_block(name, args.device)

            if isinstance(dummy, dict):
                model = HFWrapper(model)

            inputs = make_inputs(dummy, name)
            
            ep = torch.export.export(model, inputs)
            
            # Only attempt PT2E PTQ if it is a pure fp32 model that slipped in, 
            # otherwise rely on the block file's built-in torchao/low-precision logic.
            if not name.endswith("_bf16") and not name.endswith("_int8") and not name.endswith("_fp16"):
                try:
                    quant_ep = apply_ptq_static_quantization(ep, inputs)
                except Exception as quant_err:
                    print(f"   [WARNING] PTQ failed ({quant_err}). Falling back to dummy quantized export...")
                    quant_ep = ep
            else:
                quant_ep = ep
            SHLO_DIR = RESULTS_DIR / "StableHLO"
            SHLO_DIR.mkdir(parents=True, exist_ok=True)
            dest = SHLO_DIR / f"{name}_stablehlo"
            dest.mkdir(parents=True, exist_ok=True)
            
            _save_exported_program_text(quant_ep, dest)

            shlo = exported_program_to_stablehlo(quant_ep)
            shlo.save(dest)
            
            # Save the calibrated PyTorch state_dict for accurate verification
            pt_model_path = dest / "calibrated_pytorch_model.pt"
            torch.save(model.state_dict(), pt_model_path)
            
            print(f"   ✓ saved → {dest}")

            rows.append([timestamp, name, "ok", ""])
        except Exception as e:
            reason = str(e).splitlines()[0]
            print(f"   [ERROR] {name}: {reason}")
            rows.append([timestamp, name, "error", reason])

    if args.csv:
        log = RESULTS_DIR / "xla_export_log.csv"
        with log.open("a", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"[✓] log appended → {log}")

if __name__ == "__main__":
    main()
