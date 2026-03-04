#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_mappings.py
────────────────────────────────────────────────────────────────────────────
동일한 ATen 연산이 다른 StableHLO 패턴으로 lowering되는 경우를 찾는 분석 도구
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Any


def load_mapping_table(json_path: Path) -> dict[str, Any]:
    """매핑 테이블 JSON 로드"""
    with json_path.open('r', encoding='utf-8') as f:
        return json.load(f)


def extract_operation_pattern(region: dict[str, Any]) -> tuple[str, ...]:
    """Region의 StableHLO operation 패턴 추출 (순서 유지)"""
    return tuple(op['op_name'] for op in region['operations'])


def analyze_aten_lowering_patterns(tables: dict[str, dict[str, Any]]) -> dict[str, dict[str, list[tuple]]]:
    """
    각 ATen operation이 어떤 StableHLO 패턴으로 lowering되는지 분석
    
    Returns:
        {aten_op_name: {model_name: [patterns]}}
    """
    aten_patterns = defaultdict(lambda: defaultdict(list))
    
    for model_name, table in tables.items():
        for region in table['regions']:
            aten_op = region['aten_op_name']
            pattern = extract_operation_pattern(region)
            aten_patterns[aten_op][model_name].append(pattern)
    
    return aten_patterns


def find_lowering_variations(aten_patterns: dict[str, dict[str, list[tuple]]]) -> dict[str, Any]:
    """
    동일한 ATen op가 다른 패턴으로 lowering되는 경우 찾기
    """
    variations = {}
    
    for aten_op, model_patterns in aten_patterns.items():
        # 모든 모델에서 나타난 고유 패턴 수집
        all_patterns = set()
        for patterns in model_patterns.values():
            all_patterns.update(patterns)
        
        # 고유 패턴이 2개 이상이면 variation 있음
        if len(all_patterns) > 1:
            variations[aten_op] = {
                'unique_patterns': len(all_patterns),
                'patterns': list(all_patterns),
                'by_model': dict(model_patterns),
            }
    
    return variations


def print_analysis(aten_patterns: dict[str, dict[str, list[tuple]]], variations: dict[str, Any]) -> None:
    """분석 결과 출력"""
    print("=" * 80)
    print("ATen Operation Lowering Pattern Analysis")
    print("=" * 80)
    
    # 1. 전체 ATen operations 요약
    print(f"\n총 {len(aten_patterns)}개의 ATen operations 발견\n")
    
    # 2. 일관된 lowering (모든 occurrence가 동일한 패턴)
    consistent_ops = {op: patterns for op, patterns in aten_patterns.items() if op not in variations}
    
    print(f"일관된 Lowering 패턴 ({len(consistent_ops)}개):")
    print("-" * 80)
    for aten_op in sorted(consistent_ops.keys()):
        model_patterns = aten_patterns[aten_op]
        # 첫 번째 모델의 첫 번째 패턴 가져오기
        first_model = list(model_patterns.keys())[0]
        pattern = model_patterns[first_model][0]
        
        # 총 occurrence 수
        total_occurrences = sum(len(patterns) for patterns in model_patterns.values())
        
        print(f"\n  {aten_op} ({total_occurrences} occurrences)")
        print(f"    → {' → '.join(pattern)}")
    
    # 3. Variation이 있는 경우
    if variations:
        print(f"\n\n다양한 Lowering 패턴 ({len(variations)}개):")
        print("=" * 80)
        
        for aten_op, var_info in sorted(variations.items()):
            print(f"\n{aten_op}:")
            print(f"  고유 패턴 수: {var_info['unique_patterns']}")
            
            for i, pattern in enumerate(var_info['patterns'], 1):
                print(f"\n  패턴 {i}: {' → '.join(pattern)}")
                
                # 이 패턴을 사용하는 모델과 occurrence 수
                models_using = []
                for model, patterns in var_info['by_model'].items():
                    count = patterns.count(pattern)
                    if count > 0:
                        models_using.append(f"{model}({count})")
                
                print(f"    사용 모델: {', '.join(models_using)}")
    else:
        print("\n\n✓ 모든 ATen operations이 일관된 lowering 패턴을 가지고 있습니다.")
    
    print("\n" + "=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ATen-to-StableHLO lowering patterns across models"
    )
    parser.add_argument(
        "mapping_files",
        nargs="+",
        type=Path,
        help="Mapping table JSON files to compare",
    )
    
    args = parser.parse_args()
    
    # 매핑 테이블 로드
    tables = {}
    for json_path in args.mapping_files:
        if not json_path.exists():
            print(f"Warning: {json_path} not found, skipping")
            continue
        
        table = load_mapping_table(json_path)
        model_name = table['model_name']
        tables[model_name] = table
        print(f"Loaded: {model_name} ({len(table['regions'])} regions)")
    
    if not tables:
        print("Error: No valid mapping tables loaded")
        return
    
    print()
    
    # 분석 수행
    aten_patterns = analyze_aten_lowering_patterns(tables)
    variations = find_lowering_variations(aten_patterns)
    
    # 결과 출력
    print_analysis(aten_patterns, variations)


if __name__ == "__main__":
    main()
