#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_table.py
────────────────────────────────────────────────────────────────────────────
ATen-to-StableHLO 매핑 테이블 추출 도구

Tagged StableHLO MLIR 파일을 파싱하여 ATen 연산과 StableHLO 연산의 매핑을 추출합니다.

구현 전략:
1. MLIR 파싱: StableHLO operations와 location metadata 추출
2. Tag 파싱: fx_id, aten_qualname, occurrence 정보 추출
3. SSA 그래프 구축: use-def 관계 분석
4. FX 노드 그룹핑: fx_id가 동일한 operations만 그룹핑 (backward closure 없음)
5. Constant 제외: stablehlo.constant는 패턴에서 제외
6. Region 경계 탐지: 입력/출력 경계 계산
7. 매핑 테이블 생성: JSON 형식으로 저장
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 구조
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DebugTag:
    """디버그 태그 정보"""
    model_name: str = ""
    fx_id: str = ""
    fx_node_name: str = ""
    op_type: str = ""
    target: str = ""
    aten_occurrence: str = ""  # 예: "relu_0", "conv2d_1"
    stack_trace: str = ""
    raw_tag: str = ""
    
    @property
    def aten_op_name(self) -> str:
        """ATen operation 이름 (occurrence 번호 제외)"""
        if self.aten_occurrence:
            return self.aten_occurrence.rsplit('_', 1)[0]
        return ""
    
    @property
    def occurrence_index(self) -> int:
        """Occurrence 인덱스"""
        if self.aten_occurrence and '_' in self.aten_occurrence:
            try:
                return int(self.aten_occurrence.rsplit('_', 1)[1])
            except (ValueError, IndexError):
                return -1
        return -1


@dataclass
class StableHLOOp:
    """StableHLO operation 정보"""
    op_name: str  # 예: "stablehlo.convolution"
    result_names: list[str] = field(default_factory=list)  # SSA value names
    operand_names: list[str] = field(default_factory=list)  # 사용하는 value names
    attributes: dict[str, Any] = field(default_factory=dict)
    tag: DebugTag | None = None
    line_number: int = -1
    raw_line: str = ""
    
    def __hash__(self) -> int:
        """객체 identity 기반 해싱 (set에서 사용 가능하도록)"""
        return id(self)
    
    def __eq__(self, other: object) -> bool:
        """객체 identity 기반 비교"""
        return self is other


@dataclass
class FXNodeRegion:
    """FX 노드에 대응하는 StableHLO subgraph region"""
    fx_id: str
    aten_op_name: str
    occurrence_index: int
    operations: list[StableHLOOp] = field(default_factory=list)
    input_values: set[str] = field(default_factory=set)  # region 외부에서 들어오는 값
    output_values: set[str] = field(default_factory=set)  # region 외부로 나가는 값
    
    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화를 위한 딕셔너리 변환"""
        return {
            "fx_id": self.fx_id,
            "aten_op_name": self.aten_op_name,
            "occurrence_index": self.occurrence_index,
            "operations": [
                {
                    "op_name": op.op_name,
                    "result_names": op.result_names,
                    "operand_names": op.operand_names,
                    "line_number": op.line_number,
                }
                for op in self.operations
            ],
            "input_values": sorted(self.input_values),
            "output_values": sorted(self.output_values),
        }


@dataclass
class MappingTable:
    """ATen-to-StableHLO 매핑 테이블"""
    model_name: str
    regions: list[FXNodeRegion] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화를 위한 딕셔너리 변환"""
        return {
            "model_name": self.model_name,
            "regions": [r.to_dict() for r in self.regions],
            "summary": self._generate_summary(),
        }
    
    def _generate_summary(self) -> dict[str, Any]:
        """매핑 요약 정보 생성"""
        aten_ops = defaultdict(int)
        for region in self.regions:
            aten_ops[region.aten_op_name] += 1
        
        return {
            "total_regions": len(self.regions),
            "aten_operations": dict(aten_ops),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MLIR 파싱
# ─────────────────────────────────────────────────────────────────────────────
# 허용 노이즈 op: 다른 fx_id 구간에 흡수 가능한 보조 operations
ALLOWLIST_OPS = {
    "stablehlo.constant",
    "stablehlo.broadcast_in_dim",
    "stablehlo.reshape",
    "stablehlo.transpose",
    "stablehlo.convert",
    "stablehlo.bitcast_convert",
}


def parse_debug_tag(loc_str: str) -> DebugTag | None:
    """
    Location 문자열에서 디버그 태그 파싱
    
    예: "conv|fx000003|linear|call_function|aten.linear.default|linear_0"
    """
    if not loc_str or loc_str in ("[unknown]", "unknown"):
        return None
    
    # 따옴표 제거
    loc_str = loc_str.strip('"\'')
    
    # 파이프로 분리
    parts = loc_str.split('|')
    if len(parts) < 2:
        return None
    
    tag = DebugTag(raw_tag=loc_str)
    
    # 각 필드 파싱 (순서: model_name|fx_id|fx_node_name|op_type|target|aten_occurrence|stack_trace)
    if len(parts) >= 1:
        tag.model_name = parts[0]
    if len(parts) >= 2:
        tag.fx_id = parts[1]
    if len(parts) >= 3:
        tag.fx_node_name = parts[2]
    if len(parts) >= 4:
        tag.op_type = parts[3]
    if len(parts) >= 5:
        tag.target = parts[4]
    if len(parts) >= 6:
        tag.aten_occurrence = parts[5]
    if len(parts) >= 7:
        tag.stack_trace = parts[6]
    
    return tag


def extract_ssa_values(text: str) -> list[str]:
    """
    MLIR 텍스트에서 SSA value 이름 추출
    예: "%0", "%1", "%arg0"
    """
    return re.findall(r'%[\w.]+', text)


def parse_stablehlo_mlir(mlir_path: Path) -> list[StableHLOOp]:
    """
    StableHLO MLIR 파일 파싱
    
    각 operation의 결과, 피연산자, 태그 정보를 추출합니다.
    """
    operations = []
    
    with mlir_path.open('r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith('//') or stripped.startswith('#'):
            continue
        
        # stablehlo operation 찾기
        if 'stablehlo.' not in stripped:
            continue
        
        # Operation 이름 추출
        match = re.search(r'(stablehlo\.\w+)', stripped)
        if not match:
            continue
        
        op_name = match.group(1)
        
        # 결과 값 추출 (= 왼쪽)
        result_names = []
        if '=' in stripped:
            lhs = stripped.split('=')[0]
            result_names = extract_ssa_values(lhs)
        
        # 피연산자 추출 (괄호 안)
        operand_names = []
        paren_match = re.search(r'\((.*?)\)', stripped)
        if paren_match:
            operands_str = paren_match.group(1)
            operand_names = extract_ssa_values(operands_str)
        
        # Location 태그 추출
        tag = None
        loc_match = re.search(r'"([^"]+)"$', stripped)
        if loc_match:
            loc_str = loc_match.group(1)
            tag = parse_debug_tag(loc_str)
        
        op = StableHLOOp(
            op_name=op_name,
            result_names=result_names,
            operand_names=operand_names,
            tag=tag,
            line_number=line_num,
            raw_line=stripped,
        )
        operations.append(op)
    
    return operations


# ─────────────────────────────────────────────────────────────────────────────
# SSA 그래프 구축 및 분석
# ─────────────────────────────────────────────────────────────────────────────
def build_ssa_graph(operations: list[StableHLOOp]) -> tuple[dict[str, StableHLOOp], dict[str, list[StableHLOOp]]]:
    """
    SSA use-def 그래프 구축
    
    Returns:
        - value_defs: value 이름 -> 정의하는 operation
        - value_uses: value 이름 -> 사용하는 operations 리스트
    """
    value_defs: dict[str, StableHLOOp] = {}
    value_uses: dict[str, list[StableHLOOp]] = defaultdict(list)
    
    for op in operations:
        # 정의
        for result in op.result_names:
            value_defs[result] = op
        
        # 사용
        for operand in op.operand_names:
            value_uses[operand].append(op)
    
    return value_defs, value_uses


# ─────────────────────────────────────────────────────────────────────────────
# FX 노드 그룹핑 및 Region 추출
# ─────────────────────────────────────────────────────────────────────────────


def compute_region_boundaries(
    region_ops: set[StableHLOOp],
    all_ops: list[StableHLOOp],
    value_defs: dict[str, StableHLOOp],
    value_uses: dict[str, list[StableHLOOp]],
) -> tuple[set[str], set[str]]:
    """
    Region의 입력/출력 경계 계산
    
    Returns:
        - input_values: region 외부에서 정의되어 region 내부에서 사용되는 값들
        - output_values: region 내부에서 정의되어 region 외부에서 사용되는 값들
    """
    input_values = set()
    output_values = set()
    
    # Region 내부에서 정의된 모든 값
    defined_in_region = set()
    for op in region_ops:
        defined_in_region.update(op.result_names)
    
    # 입력 경계: region 내부 op가 사용하는 값 중 region 외부에서 정의된 것
    for op in region_ops:
        for operand in op.operand_names:
            if operand not in defined_in_region:
                input_values.add(operand)
    
    # 출력 경계: region 내부에서 정의된 값 중 region 외부에서 사용되는 것
    for value in defined_in_region:
        if value in value_uses:
            for use_op in value_uses[value]:
                if use_op not in region_ops:
                    output_values.add(value)
                    break
    
    return input_values, output_values


# ─────────────────────────────────────────────────────────────────────────────
# FX 노드 그룹핑 및 Region 추출
# ─────────────────────────────────────────────────────────────────────────────
def group_operations_by_fx_id(operations: list[StableHLOOp]) -> dict[str, list[StableHLOOp]]:
    """FX ID별로 operations 그룹핑"""
    groups = defaultdict(list)
    
    for op in operations:
        if op.tag and op.tag.fx_id:
            groups[op.tag.fx_id].append(op)
    
    return groups


def extract_regions(operations: list[StableHLOOp]) -> list[FXNodeRegion]:
    """
    Tagged operations로부터 FX node regions 추출
    
    전략:
    - fx_id가 동일한 operations만 하나의 region으로 그룹핑
    - Backward closure 사용하지 않음 (이전 연산 포함 안 함)
    - Constant operations는 패턴에서 제외
    """
    value_defs, value_uses = build_ssa_graph(operations)
    fx_groups = group_operations_by_fx_id(operations)
    
    regions = []
    
    for fx_id, all_ops_with_fx_id in sorted(fx_groups.items()):
        if not all_ops_with_fx_id:
            continue
        
        # 첫 번째 op의 태그에서 ATen 정보 추출
        first_tag = all_ops_with_fx_id[0].tag
        if not first_tag:
            continue
        
        aten_op_name = first_tag.aten_op_name
        occurrence_index = first_tag.occurrence_index
        
        # Constant operations 제외
        region_ops = [
            op for op in all_ops_with_fx_id
            if op.op_name != "stablehlo.constant"
        ]
        
        # Constant만 있는 경우 스킵
        if not region_ops:
            continue
        
        # 경계 계산 (fx_id가 같은 ops만 사용)
        input_values, output_values = compute_region_boundaries(
            set(region_ops), operations, value_defs, value_uses
        )
        
        region = FXNodeRegion(
            fx_id=fx_id,
            aten_op_name=aten_op_name,
            occurrence_index=occurrence_index,
            operations=sorted(region_ops, key=lambda op: op.line_number),
            input_values=input_values,
            output_values=output_values,
        )
        regions.append(region)
    
    return regions


# ─────────────────────────────────────────────────────────────────────────────
# 메인 추출 로직
# ─────────────────────────────────────────────────────────────────────────────
def extract_mapping_table(mlir_path: Path, model_name: str | None = None) -> MappingTable:
    """
    Tagged StableHLO MLIR 파일로부터 매핑 테이블 추출
    """
    print(f"[1/4] Parsing MLIR file: {mlir_path}")
    operations = parse_stablehlo_mlir(mlir_path)
    print(f"      Found {len(operations)} StableHLO operations")
    
    # 태그가 있는 operations 카운트
    tagged_ops = [op for op in operations if op.tag and op.tag.fx_id]
    print(f"      {len(tagged_ops)} operations have debug tags")
    
    print(f"[2/4] Building SSA use-def graph")
    value_defs, value_uses = build_ssa_graph(operations)
    print(f"      {len(value_defs)} SSA values defined")
    
    print(f"[3/4] Extracting FX node regions")
    regions = extract_regions(operations)
    print(f"      Extracted {len(regions)} regions")
    
    # 모델 이름 추출
    if model_name is None and regions:
        model_name = regions[0].fx_id.split('_')[0] if '_' in regions[0].fx_id else "unknown"
        if regions[0].operations and regions[0].operations[0].tag:
            model_name = regions[0].operations[0].tag.model_name or model_name
    
    print(f"[4/4] Generating mapping table")
    table = MappingTable(model_name=model_name or "unknown", regions=regions)
    
    return table


def save_mapping_table(table: MappingTable, output_path: Path) -> None:
    """매핑 테이블을 JSON 파일로 저장"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(table.to_dict(), f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Mapping table saved to: {output_path}")
    print(f"  Model: {table.model_name}")
    print(f"  Total regions: {len(table.regions)}")
    
    # ATen ops 요약 출력
    aten_ops = defaultdict(int)
    for region in table.regions:
        aten_ops[region.aten_op_name] += 1
    
    print(f"\n  ATen operations:")
    for aten_op, count in sorted(aten_ops.items()):
        print(f"    - {aten_op}: {count} occurrence(s)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ATen-to-StableHLO mapping table from tagged MLIR files"
    )
    parser.add_argument(
        "mlir_file",
        type=Path,
        help="Path to tagged StableHLO MLIR file (e.g., results/xla/StableHLO/conv_stablehlo/functions/forward.mlir)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output JSON file path (default: <mlir_dir>/mapping_table.json)",
    )
    parser.add_argument(
        "-m", "--model-name",
        type=str,
        help="Model name (auto-detected from tags if not specified)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    if not args.mlir_file.exists():
        print(f"Error: MLIR file not found: {args.mlir_file}")
        return
    
    # 출력 경로 결정
    if args.output:
        output_path = args.output
    else:
        # MLIR 파일과 같은 디렉토리에 저장
        output_path = args.mlir_file.parent.parent / "mapping_table.json"
    
    # 매핑 테이블 추출
    table = extract_mapping_table(args.mlir_file, args.model_name)
    
    # 저장
    save_mapping_table(table, output_path)


if __name__ == "__main__":
    main()
