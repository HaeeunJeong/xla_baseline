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
4) FX 그래프 자동 태깅:
   - --xla_debuginfo 옵션을 켜면, ExportedProgram의 graph_module에
     torch.ops.xla.write_mlir_debuginfo(tensor, tag)를 각 텐서-producing 노드 뒤에 삽입
   - tag에 FX node 고유 ID(순번) + node.name/op/target(+가능하면 stack_trace) 포함
   - 그 후, 태깅된 GraphModule을 다시 torch.export.export()하여 StableHLO export 수행
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import warnings

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from io import StringIO
from contextlib import redirect_stdout

# ─────────────────────────────────────────────────────────────────────────────
# 0) 경고 억제
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*isinstance\(treespec, LeafSpec\).*",
)

warnings.filterwarnings("ignore", message=".*write_mlir_debuginfo.*")

import torch
import torch.fx as fx
from torch_xla.stablehlo import exported_program_to_stablehlo

# write_mlir_debuginfo op 등록/활성화용 (필요)
import torch_xla.experimental.xla_mlir_debuginfo  # noqa: F401

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT_DIR), str(ROOT_DIR / "scripts")])

from scripts.pytorch_baseline import load_model, _discover_models  # noqa: E402

RESULTS_DIR = ROOT_DIR / "results" / "xla"
DUMP_DIR = RESULTS_DIR / "dump"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DUMP_DIR.mkdir(parents=True, exist_ok=True)

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
        super().__init__()
        self.m = m

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
# FX GraphModule 자동 태깅: write_mlir_debuginfo 삽입
# ─────────────────────────────────────────────────────────────────────────────
def _is_tensor_like_meta(node: fx.Node) -> bool:
    """
    torch.export.export로 만든 graph_module의 node.meta['val']에 FakeTensor가 들어가는 경우가 많습니다.
    tensor-producing 노드만 태깅하기 위한 필터입니다.
    """
    v = node.meta.get("val", None)
    return isinstance(v, torch.Tensor)


def _safe_str(x: object, limit: int = 120) -> str:
    s = str(x)
    s = s.replace("\n", "\\n")
    if len(s) > limit:
        s = s[:limit] + "..."
    return s


def _make_fx_tag(tag_prefix: str, seq: int, n: fx.Node, aten_op_counter: dict[str, int], max_len: int = 240) -> str:
    """
    최소 태그: fx_id + aten_occurrence만 포함
    
    예: "fx000003|conv2d_0"
    
    aten_op_counter: ATen operation별 occurrence 카운터 (mutable dict)
    """
    fx_id = f"fx{seq:06d}"  # 고유 순번 ID
    
    # ATen operation 이름 추출 및 occurrence 카운팅
    aten_op_name = ""
    aten_occurrence = ""
    if n.op == "call_function" and hasattr(n.target, "__name__"):
        aten_op_name = n.target.__name__
        # 이 ATen op의 occurrence 번호 증가
        aten_op_counter[aten_op_name] = aten_op_counter.get(aten_op_name, 0) + 1
        aten_occurrence = f"{aten_op_name}_{aten_op_counter[aten_op_name] - 1}"  # 0-indexed

    # 최소 태그: fx_id|aten_occurrence
    parts = [fx_id]
    if aten_occurrence:
        parts.append(aten_occurrence)
    
    tag = "|".join(parts)
    return tag




def _instrument_graphmodule_with_xla_debuginfo(gm: fx.GraphModule, tag_prefix: str) -> fx.GraphModule:
    """
    각 tensor-producing node 뒤에:
      torch.ops.xla.write_mlir_debuginfo(tensor, tag)
    를 삽입합니다.

    tag에는 FX 노드 고유 ID(fx000123) 및 ATen op occurrence 번호 포함.
    """
    g = gm.graph
    seq = 0
    aten_op_counter: dict[str, int] = {}  # ATen operation별 occurrence 카운터

    for n in list(g.nodes):
        if n.op not in ("call_function", "call_method", "call_module"):
            continue

        # 이미 debuginfo인 노드는 스킵
        if n.op == "call_function" and n.target == torch.ops.xla.write_mlir_debuginfo:
            continue

        # 텐서 출력만 태깅
        if not _is_tensor_like_meta(n):
            continue

        tag = _make_fx_tag(tag_prefix=tag_prefix, seq=seq, n=n, aten_op_counter=aten_op_counter)
        seq += 1

        with g.inserting_after(n):
            dbg = g.call_function(torch.ops.xla.write_mlir_debuginfo, args=(n, tag))

        # 메타 복사 (shape/type 추론 유지 도움)
        dbg.meta = dict(n.meta)

        # n의 모든 사용처를 dbg로 교체 (dbg 자기 자신 제외)
        n.replace_all_uses_with(dbg)
        dbg.args = (n, tag)

    g.lint()
    gm.recompile()
    return gm


def _export_with_optional_debuginfo(
    model: torch.nn.Module,
    inputs: tuple,
    name: str,
    enable: bool,
) -> "torch.export.ExportedProgram":
    """
    enable=False: 기존처럼 바로 export
    enable=True :
      1) export → ep0
      2) ep0.graph_module을 in-place로 수정하여 debuginfo 삽입 (FX 고유 ID 포함)
    """
    ep0 = torch.export.export(model, inputs)
    if not enable:
        return ep0

    # GraphModule을 in-place로 수정
    _instrument_graphmodule_with_xla_debuginfo(ep0.graph_module, tag_prefix=f"{name}")
    
    return ep0


# ─────────────────────────────────────────────────────────────────────────────
# ExportedProgram 텍스트 저장
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
        code_txt = getattr(gm, "code", None)
        if code_txt is None:
            code_txt = repr(gm)
        with py_path.open("w", encoding="utf-8") as f:
            f.write(str(code_txt))
    except Exception as e:
        with py_path.open("w", encoding="utf-8") as f:
            f.write(f"# failed to dump GraphModule code: {type(e).__name__}: {e}\n")

    # FX Graph tabular 저장
    try:
        g = ep.graph
        txt: str
        if hasattr(g, "tabular") and callable(getattr(g, "tabular")):
            txt = g.tabular()  # type: ignore[call-arg]
        else:
            buf = StringIO()
            with redirect_stdout(buf):
                g.print_tabular()  # type: ignore[attr-defined]
            txt = buf.getvalue()
        with fx_path.open("w", encoding="utf-8") as f:
            f.write(txt if txt.strip() else "# empty FX graph tabular output\n")
    except Exception as e:
        with fx_path.open("w", encoding="utf-8") as f:
            f.write(f"# failed to dump FX graph: {type(e).__name__}: {e}\n")





# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="*", help="model keys; omit = all discovered")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--csv", action="store_true", help="append csv log")
    ap.add_argument(
        "--xla_debuginfo",
        action="store_true",
        help="auto-insert torch.ops.xla.write_mlir_debuginfo after tensor-producing FX nodes (includes FX unique id)",
    )
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

            inputs = make_inputs(dummy)

            # export 단계: 필요 시 debuginfo 삽입 후 재-export
            ep = _export_with_optional_debuginfo(model, inputs, name=name, enable=args.xla_debuginfo)

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
