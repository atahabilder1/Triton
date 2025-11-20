#!/usr/bin/env python3
"""
Test PDG extraction on the FULL FORGE dataset (6,432 contracts)
This will verify our PDG improvements work on the complete dataset
"""
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.slither_wrapper import extract_static_features

def test_pdg_extraction_full_dataset():
    """Test PDG extraction on all 6,432 contracts"""

    dataset_dir = Path("data/datasets/forge_full_cleaned")

    print("=" * 80)
    print("PDG EXTRACTION TEST - FULL FORGE DATASET")
    print("=" * 80)
    print()

    # Stats tracking
    total_contracts = 0
    successful_pdgs = 0
    failed_pdgs = 0

    results_by_class = defaultdict(lambda: {'total': 0, 'success': 0, 'failed': 0})
    failure_reasons = Counter()

    # Test each split
    for split in ['train', 'val', 'test']:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            print(f"⚠️  Split directory not found: {split_dir}")
            continue

        print(f"\n{'=' * 80}")
        print(f"Testing {split.upper()} split")
        print(f"{'=' * 80}")

        # Process each vulnerability class
        for vuln_dir in sorted(split_dir.iterdir()):
            if not vuln_dir.is_dir():
                continue

            vuln_class = vuln_dir.name
            contracts = list(vuln_dir.glob("*.sol"))

            print(f"\n  {vuln_class}: {len(contracts)} contracts")

            class_success = 0
            class_failed = 0

            for i, contract_path in enumerate(contracts):
                total_contracts += 1
                results_by_class[vuln_class]['total'] += 1

                # Progress indicator every 100 contracts
                if (i + 1) % 100 == 0:
                    print(f"    Progress: {i+1}/{len(contracts)} contracts processed...")

                # Extract PDG
                with open(contract_path, 'r', encoding='utf-8', errors='ignore') as f:
                    source_code = f.read()

                pdg = extract_static_features(source_code, contract_path=str(contract_path))

                if pdg and pdg.get('nodes') and len(pdg['nodes']) > 0:
                    successful_pdgs += 1
                    class_success += 1
                    results_by_class[vuln_class]['success'] += 1
                else:
                    failed_pdgs += 1
                    class_failed += 1
                    results_by_class[vuln_class]['failed'] += 1

                    # Track failure reason if available
                    if pdg and 'error' in pdg:
                        failure_reasons[pdg['error'][:100]] += 1

            success_rate = 100 * class_success / len(contracts) if contracts else 0
            print(f"    ✓ Success: {class_success}/{len(contracts)} ({success_rate:.1f}%)")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS - FULL DATASET PDG EXTRACTION")
    print("=" * 80)
    print(f"\nTotal contracts tested: {total_contracts}")
    print(f"Successful PDG extractions: {successful_pdgs}")
    print(f"Failed extractions: {failed_pdgs}")

    success_rate = 100 * successful_pdgs / total_contracts if total_contracts > 0 else 0
    print(f"\n{'=' * 80}")
    print(f"OVERALL SUCCESS RATE: {success_rate:.2f}%")
    print(f"{'=' * 80}")

    # Per-class breakdown
    print("\n" + "=" * 80)
    print("SUCCESS RATE BY VULNERABILITY CLASS")
    print("=" * 80)
    print(f"{'Class':<35} {'Total':>8} {'Success':>8} {'Failed':>8} {'Rate':>8}")
    print("-" * 80)

    for vuln_class in sorted(results_by_class.keys()):
        stats = results_by_class[vuln_class]
        rate = 100 * stats['success'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{vuln_class:<35} {stats['total']:>8} {stats['success']:>8} {stats['failed']:>8} {rate:>7.1f}%")

    # Top failure reasons
    if failure_reasons:
        print("\n" + "=" * 80)
        print("TOP 10 FAILURE REASONS")
        print("=" * 80)
        for reason, count in failure_reasons.most_common(10):
            pct = 100 * count / failed_pdgs if failed_pdgs > 0 else 0
            print(f"  [{count:4d}] ({pct:5.1f}%) {reason}")

    # Assessment
    print("\n" + "=" * 80)
    print("ASSESSMENT")
    print("=" * 80)

    if success_rate >= 50:
        print(f"✅ EXCELLENT: {success_rate:.1f}% success rate!")
        print(f"   Estimated valid training samples: {int(successful_pdgs * 0.7)}")
        print("   This is sufficient for deep learning!")
        print("\n   ➡️  READY TO TRAIN")
    elif success_rate >= 30:
        print(f"⚠️  ACCEPTABLE: {success_rate:.1f}% success rate")
        print(f"   Estimated valid training samples: {int(successful_pdgs * 0.7)}")
        print("   May need more improvements for optimal results")
    else:
        print(f"❌ INSUFFICIENT: {success_rate:.1f}% success rate")
        print("   Need to investigate and fix PDG extraction issues")
        print("\n   ➡️  DO NOT TRAIN YET - Fix PDG extraction first!")

    # Save detailed results
    results = {
        'total_contracts': total_contracts,
        'successful_pdgs': successful_pdgs,
        'failed_pdgs': failed_pdgs,
        'success_rate': success_rate,
        'by_class': dict(results_by_class),
        'top_failures': dict(failure_reasons.most_common(20))
    }

    output_file = Path("logs/pdg_test_full_dataset.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 Detailed results saved to: {output_file}")
    print()

    return success_rate >= 50

if __name__ == "__main__":
    success = test_pdg_extraction_full_dataset()
    sys.exit(0 if success else 1)
