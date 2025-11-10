#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_torch_xla_stablehlo.py
──────────────────────────────────────────────────────────────────────────────
PyTorch  →  StableHLO  via torch-xla (exported_program_to_stablehlo).

* 모델을 주지 않으면 models/ 디렉터리의 모든 *_block.py 를 자동 탐색합니다.
* 성공 :  results/xla/<name>_stablehlo/   디렉터리(MLIR+weights) 저장
* 실패 :  STDOUT + CSV(results/xla_export_log.csv) 기록

추가 기능
────────
1) copyreg FutureWarning 억제:
   "isinstance(treespec, LeafSpec) is deprecated ..." 경고를 코드 레벨에서 무시
2) ExportedProgram 텍스트 저장:
   - exported_program.py (FX GraphModule의 .code)
   - exported_graph.txt (FX Graph의 tabular 출력)
3) StableHLO → Linalg MLIR 변환:
   - functions/*.mlir 존재 시 stablehlo-opt 실행하여 {model}_linalg.mlir 생성
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import sys
import warnings
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from io import StringIO                 # [변경] 추가
from contextlib import redirect_stdout  # [변경] 추가

# ─────────────────────────────────────────────────────────────────────────────
# 0) 경고 억제: LeafSpec deprecation (copyreg 경로에서 발생하는 FutureWarning)
#    메시지 패턴에 정확히 매칭되도록 필터링
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*isinstance\(treespec, LeafSpec\).*",
)

import torch
from torch_xla.stablehlo import exported_program_to_stablehlo

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT_DIR), str(ROOT_DIR / "scripts")])

from scripts.pytorch_baseline import load_model, _discover_models  # noqa: E402

RESULTS_DIR = ROOT_DIR / "results" / "xla"
DUMP_DIR = RESULTS_DIR / "dump"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DUMP_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 모델 블록 로딩
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"

def discover_model_keys() -> list[str]:
    """models/ 폴더의 *_block.py → key 추출"""
    return sorted(
        f.stem[:-6]  # strip "_block"
        for f in MODELS_DIR.glob("*_block.py")
        if f.stem.endswith("_block")
    )

def load_model_block(name: str, device: str = "cpu") -> tuple[torch.nn.Module, Any]:
    """
    import models/<name>_block.py  →
        get_model(), get_dummy_input()
    """
    mod = importlib.import_module(f"models.{name}_block")
    model = mod.get_model().to(device).eval()
    dummy = mod.get_dummy_input()
    return model, dummy

# ─────────────────────────────────────────────────────────────────────────────
# 입력 구성 & HF 래퍼
# ─────────────────────────────────────────────────────────────────────────────
class HFWrapper(torch.nn.Module):
    """kwargs(HF) → forward(ids, mask)"""
    def __init__(self, m: torch.nn.Module):
        super().__init__(); self.m = m
    def forward(self, ids, mask):  # type: ignore
        return self.m(input_ids=ids, attention_mask=mask).last_hidden_state

def make_inputs(dummy: Any) -> tuple:
    """dummy 사양을 torch.export 가 받을 *args 로 변환"""
    if isinstance(dummy, tuple):
        # (Tensor, Tensor) = GNN  vs  shape-tuple = Vision
        if len(dummy) == 2 and all(isinstance(t, torch.Tensor) for t in dummy):
            return dummy
        return (torch.randn(*dummy),)
    if isinstance(dummy, dict):          # LLM dict
        return (dummy["input_ids"], dummy["attention_mask"])
    raise RuntimeError(f"Unsupported dummy type: {type(dummy)}")

# ─────────────────────────────────────────────────────────────────────────────
# 보조: ExportedProgram 텍스트 저장
# ─────────────────────────────────────────────────────────────────────────────
def _save_exported_program_text(ep: "torch.export.ExportedProgram", dest_dir: Path) -> None:
    """
    - exported_program.py : FX GraphModule의 코드
    - exported_graph.txt  : FX Graph의 표 형태 출력
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    py_path = dest_dir / "exported_program.py"
    fx_path = dest_dir / "exported_graph.txt"

    # GraphModule 코드 저장
    try:
        gm = ep.graph_module
        print("gm:", type(gm))
        code_txt = getattr(gm, "code", None)
        if code_txt is None:
            # fallback: repr(GraphModule)
            code_txt = repr(gm)
            print(str(gm))
        with py_path.open("w", encoding="utf-8") as f:
            f.write(str(code_txt))
    except Exception as e:
        with py_path.open("w", encoding="utf-8") as f:
            f.write(f"# failed to dump GraphModule code: {type(e).__name__}: {e}\n")

    # FX Graph tabular 저장
    try:
        g = ep.graph
        txt: str
        # 신버전: graph.tabular() 가 문자열을 반환
        if hasattr(g, "tabular") and callable(getattr(g, "tabular")):
            txt = g.tabular()  # type: ignore[call-arg]
        else:
            # 구버전: print_tabular()는 stdout에만 출력. 캡처해서 파일로 저장.
            buf = StringIO()
            with redirect_stdout(buf):
                # 일부 버전은 인자를 받지 않습니다.
                g.print_tabular()  # type: ignore[attr-defined]
            txt = buf.getvalue()
        with fx_path.open("w", encoding="utf-8") as f:
            f.write(txt if txt.strip() else "# empty FX graph tabular output\n")
    except Exception as e:
        # 요구사항: 그래프가 없어도 되므로, 실패 시 메시지만 남기고 계속 진행
        with fx_path.open("w", encoding="utf-8") as f:
            f.write(f"# failed to dump FX graph: {type(e).__name__}: {e}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 보조: StableHLO → Linalg 변환 실행
# ─────────────────────────────────────────────────────────────────────────────
def _lower_stablehlo_to_linalg(stablehlo_dir: Path, model_name: str) -> None:
    """
    stablehlo bundle 디렉터리 내 functions/*.mlir 파일을 찾아
    다음 명령을 실행:
      stablehlo-opt <mlir> --stablehlo-legalize-to-linalg -o {model}_linalg.mlir

    - stablehlo-opt 가 PATH에 없거나, .mlir 파일이 없으면 조용히 건너뜀.
    """
    tool = shutil.which("stablehlo-opt")
    if tool is None:
        print(f"   [skip] stablehlo-opt not found in PATH")
        return

    func_dir = stablehlo_dir / "functions"
    if not func_dir.is_dir():
        print(f"   [skip] functions dir not found: {func_dir}")
        return

    # 우선순위: forward.mlir → 그 외 임의의 첫 번째 .mlir
    mlir_candidates = list(func_dir.glob("*.mlir"))
    if not mlir_candidates:
        print(f"   [skip] no .mlir found under {func_dir}")
        return

    forward_mlir = func_dir / "forward.mlir"
    src_mlir = forward_mlir if forward_mlir.exists() else mlir_candidates[0]

    out_path = stablehlo_dir / f"{model_name}_linalg.mlir"
    cmd = [tool, str(src_mlir), "--stablehlo-legalize-to-linalg", "-o", str(out_path)]

    try:
        print(f"   stablehlo-opt → {out_path.name}")
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if proc.stdout:
            pass
    except subprocess.CalledProcessError as e:
        print(f"   [stablehlo-opt ERROR] {e.returncode}: {e.stderr.strip() or e.stdout.strip()}")

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="*", help="model keys; omit = all discovered")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--csv", action="store_true", help="append csv log")
    args = ap.parse_args()

    keys: Iterable[str] = args.model or discover_model_keys()
    timestamp = datetime.now().isoformat(timespec="seconds")
    rows: list[list[str]] = []

    for name in keys:
        try:
            print(f"[{name}] torch_xla → StableHLO export …")
            model, dummy = load_model_block(name, args.device)

            # HF 모델이면 kwargs → positional 래퍼
            if isinstance(dummy, dict):
                model = HFWrapper(model)

            ep = torch.export.export(model, make_inputs(dummy))

            # 1) ExportedProgram 텍스트 저장
            SHLO_DIR = RESULTS_DIR / "StableHLO"
            SHLO_DIR.mkdir(parents=True, exist_ok=True)
            dest = SHLO_DIR / f"{name}_stablehlo"
            dest.mkdir(parents=True, exist_ok=True)
            _save_exported_program_text(ep, dest)

            # 2) StableHLO 번들 저장
            shlo = exported_program_to_stablehlo(ep)
            shlo.save(dest)
            print(f"   ✓ saved → {dest}")

            # 3) StableHLO → Linalg 변환 시도
            _lower_stablehlo_to_linalg(dest, name)

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
