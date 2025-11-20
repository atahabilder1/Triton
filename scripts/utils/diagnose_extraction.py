#!/usr/bin/env python3
"""
Diagnose Static and Dynamic Feature Extraction Success Rates
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.slither_wrapper import SlitherWrapper
from tools.mythril_wrapper import MythrilWrapper
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.WARNING)

def diagnose_dataset(contracts_dir: str, max_samples: int = 50):
    """Test extraction success rates"""

    contracts_path = Path(contracts_dir)
    slither = SlitherWrapper(timeout=60)
    mythril = MythrilWrapper(timeout=60, max_depth=12)

    results = {
        'total': 0,
        'static_success': 0,
        'dynamic_success': 0,
        'static_empty': 0,
        'dynamic_empty': 0,
        'static_errors': [],
        'dynamic_errors': []
    }

    # Load contracts
    sol_files = []
    for vuln_dir in contracts_path.glob('*'):
        if vuln_dir.is_dir():
            sol_files.extend(list(vuln_dir.glob('*.sol'))[:10])  # 10 per category

    sol_files = sol_files[:max_samples]

    print(f"\n{'='*80}")
    print(f"DIAGNOSING {len(sol_files)} CONTRACTS")
    print(f"{'='*80}\n")

    for sol_file in tqdm(sol_files, desc="Testing"):
        results['total'] += 1

        try:
            with open(sol_file, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
        except:
            continue

        # Test Slither
        slither_result = slither.analyze_contract(source_code)
        if slither_result.get('success'):
            pdg = slither_result.get('pdg')
            if pdg and pdg.number_of_nodes() > 0:
                results['static_success'] += 1
            else:
                results['static_empty'] += 1
        else:
            error = slither_result.get('error', 'Unknown')
            results['static_errors'].append(error[:100])

        # Test Mythril
        mythril_result = mythril.analyze_contract(source_code)
        if mythril_result.get('success'):
            traces = mythril_result.get('execution_traces', [])
            if traces and any(len(t.get('steps', [])) > 0 for t in traces):
                results['dynamic_success'] += 1
            else:
                results['dynamic_empty'] += 1
        else:
            error = mythril_result.get('error', 'Unknown')
            results['dynamic_errors'].append(error[:100])

    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}\n")

    print(f"Total contracts tested: {results['total']}")
    print(f"\nSTATIC ANALYSIS (Slither):")
    print(f"  ✓ Success: {results['static_success']}/{results['total']} ({100*results['static_success']/results['total']:.1f}%)")
    print(f"  ∅ Empty PDG: {results['static_empty']}/{results['total']} ({100*results['static_empty']/results['total']:.1f}%)")
    print(f"  ✗ Failures: {len(results['static_errors'])}/{results['total']} ({100*len(results['static_errors'])/results['total']:.1f}%)")

    print(f"\nDYNAMIC ANALYSIS (Mythril):")
    print(f"  ✓ Success: {results['dynamic_success']}/{results['total']} ({100*results['dynamic_success']/results['total']:.1f}%)")
    print(f"  ∅ Empty traces: {results['dynamic_empty']}/{results['total']} ({100*results['dynamic_empty']/results['total']:.1f}%)")
    print(f"  ✗ Failures: {len(results['dynamic_errors'])}/{results['total']} ({100*len(results['dynamic_errors'])/results['total']:.1f}%)")

    # Show common errors
    if results['static_errors']:
        print(f"\nTop Static Errors:")
        from collections import Counter
        for error, count in Counter(results['static_errors']).most_common(3):
            print(f"  - {error} ({count}x)")

    if results['dynamic_errors']:
        print(f"\nTop Dynamic Errors:")
        from collections import Counter
        for error, count in Counter(results['dynamic_errors']).most_common(3):
            print(f"  - {error} ({count}x)")

    print(f"\n{'='*80}\n")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/datasets/smartbugs-curated/dataset")
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()

    diagnose_dataset(args.dataset, args.max_samples)
