#!/usr/bin/env python3
"""
Combine All Labeled Datasets into One Unified Training Dataset

This script combines:
1. SmartBugs Curated (143 contracts)
2. SmartBugs Samples (10 contracts)
3. SolidiFI (50 safe contracts)
4. Not So Smart Contracts (25 contracts)

Total: 228 labeled contracts

Output: data/datasets/combined_labeled/
"""

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
import random

# Set random seed for reproducibility
random.seed(42)

# Define project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "datasets"
OUTPUT_DIR = DATA_DIR / "combined_labeled"

# Define the 10 vulnerability classes
VULNERABILITY_CLASSES = [
    'access_control',
    'arithmetic',
    'bad_randomness',
    'denial_of_service',
    'front_running',
    'reentrancy',
    'short_addresses',
    'time_manipulation',
    'unchecked_low_level_calls',
    'safe'
]

# Mapping from dataset-specific names to standard class names
CLASS_MAPPING = {
    # SmartBugs naming
    'access_control': 'access_control',
    'arithmetic': 'arithmetic',
    'bad_randomness': 'bad_randomness',
    'denial_of_service': 'denial_of_service',
    'front_running': 'front_running',
    'reentrancy': 'reentrancy',
    'short_addresses': 'short_addresses',
    'time_manipulation': 'time_manipulation',
    'unchecked_low_level_calls': 'unchecked_low_level_calls',
    'other': 'safe',

    # Not So Smart Contracts naming
    'integer_overflow': 'arithmetic',
    'race_condition': 'front_running',
    'unchecked_external_call': 'unchecked_low_level_calls',
    'honeypots': 'access_control',
    'unprotected_function': 'access_control',
    'wrong_constructor_name': 'access_control',
    'forced_ether_reception': 'safe',
    'incorrect_interface': 'safe',
    'variable shadowing': 'safe',

    # SolidiFI (all safe)
    'contracts': 'safe',
}


def create_output_directories():
    """Create output directory structure"""
    print(f"Creating output directory: {OUTPUT_DIR}")

    # Remove old combined dataset if exists
    if OUTPUT_DIR.exists():
        print(f"  Removing old combined dataset...")
        shutil.rmtree(OUTPUT_DIR)

    # Create main directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for each class
    for vuln_class in VULNERABILITY_CLASSES:
        (OUTPUT_DIR / vuln_class).mkdir(exist_ok=True)

    print(f"  ✓ Created {len(VULNERABILITY_CLASSES)} class directories")


def copy_smartbugs_curated():
    """Copy SmartBugs Curated contracts"""
    print("\n" + "="*80)
    print("Dataset 1: SmartBugs Curated")
    print("="*80)

    source_dir = DATA_DIR / "smartbugs-curated" / "dataset"

    if not source_dir.exists():
        print(f"  ⚠ Warning: {source_dir} not found!")
        return 0

    stats = defaultdict(int)

    for vuln_dir in source_dir.iterdir():
        if not vuln_dir.is_dir():
            continue

        vuln_type = vuln_dir.name
        mapped_class = CLASS_MAPPING.get(vuln_type, vuln_type)

        contracts = list(vuln_dir.glob("*.sol"))

        for contract_file in contracts:
            dest_file = OUTPUT_DIR / mapped_class / f"smartbugs_curated_{contract_file.name}"
            shutil.copy2(contract_file, dest_file)
            stats[mapped_class] += 1

    total = sum(stats.values())
    print(f"  Copied {total} contracts:")
    for vuln_class in sorted(stats.keys()):
        print(f"    {vuln_class:30s}: {stats[vuln_class]:3d} contracts")

    return total


def copy_smartbugs_samples():
    """Copy SmartBugs Samples contracts"""
    print("\n" + "="*80)
    print("Dataset 2: SmartBugs Samples")
    print("="*80)

    source_dir = DATA_DIR / "smartbugs" / "samples"

    if not source_dir.exists():
        print(f"  ⚠ Warning: {source_dir} not found!")
        return 0

    # Read vulnerabilities.json
    vuln_json = source_dir / "vulnerabilities.json"
    if not vuln_json.exists():
        print(f"  ⚠ Warning: {vuln_json} not found!")
        return 0

    with open(vuln_json, 'r') as f:
        vulnerabilities = json.load(f)

    stats = defaultdict(int)

    # Create mapping of contract name to vulnerability type
    contract_vulns = {}
    for entry in vulnerabilities:
        contract_name = entry['name']
        if entry['vulnerabilities']:
            vuln_category = entry['vulnerabilities'][0]['category']
            contract_vulns[contract_name] = CLASS_MAPPING.get(vuln_category, vuln_category)
        else:
            contract_vulns[contract_name] = 'safe'

    # Find and copy contracts
    for contract_name, vuln_class in contract_vulns.items():
        # Search for contract in all version directories
        found = False
        for version_dir in source_dir.iterdir():
            if not version_dir.is_dir() or version_dir.name == 'README.md':
                continue

            contract_file = version_dir / contract_name
            if contract_file.exists():
                dest_file = OUTPUT_DIR / vuln_class / f"smartbugs_samples_{contract_name}"
                shutil.copy2(contract_file, dest_file)
                stats[vuln_class] += 1
                found = True
                break

        if not found:
            print(f"  ⚠ Warning: {contract_name} not found")

    total = sum(stats.values())
    print(f"  Copied {total} contracts:")
    for vuln_class in sorted(stats.keys()):
        print(f"    {vuln_class:30s}: {stats[vuln_class]:3d} contracts")

    return total


def copy_solidifi():
    """Copy SolidiFI safe contracts"""
    print("\n" + "="*80)
    print("Dataset 3: SolidiFI (Safe Contracts)")
    print("="*80)

    source_dir = DATA_DIR / "solidifi" / "contracts"

    if not source_dir.exists():
        print(f"  ⚠ Warning: {source_dir} not found!")
        return 0

    contracts = list(source_dir.glob("*.sol"))

    stats = defaultdict(int)

    for contract_file in contracts:
        dest_file = OUTPUT_DIR / "safe" / f"solidifi_{contract_file.name}"
        shutil.copy2(contract_file, dest_file)
        stats['safe'] += 1

    total = sum(stats.values())
    print(f"  Copied {total} contracts:")
    for vuln_class in sorted(stats.keys()):
        print(f"    {vuln_class:30s}: {stats[vuln_class]:3d} contracts")

    return total


def copy_not_so_smart_contracts():
    """Copy Not So Smart Contracts"""
    print("\n" + "="*80)
    print("Dataset 4: Not So Smart Contracts")
    print("="*80)

    source_dir = DATA_DIR / "audits" / "not_so_smart_contracts"

    if not source_dir.exists():
        print(f"  ⚠ Warning: {source_dir} not found!")
        return 0

    stats = defaultdict(int)

    for vuln_dir in source_dir.iterdir():
        if not vuln_dir.is_dir():
            continue

        vuln_type = vuln_dir.name
        mapped_class = CLASS_MAPPING.get(vuln_type, 'safe')

        # Find all .sol files recursively
        contracts = list(vuln_dir.rglob("*.sol"))

        for contract_file in contracts:
            dest_file = OUTPUT_DIR / mapped_class / f"not_so_smart_{contract_file.name}"
            shutil.copy2(contract_file, dest_file)
            stats[mapped_class] += 1

    total = sum(stats.values())
    print(f"  Copied {total} contracts:")
    for vuln_class in sorted(stats.keys()):
        print(f"    {vuln_class:30s}: {stats[vuln_class]:3d} contracts")

    return total


def generate_summary():
    """Generate dataset summary"""
    print("\n" + "="*80)
    print("COMBINED DATASET SUMMARY")
    print("="*80)

    summary = {
        'total_contracts': 0,
        'classes': {},
        'datasets_combined': [
            'SmartBugs Curated',
            'SmartBugs Samples',
            'SolidiFI',
            'Not So Smart Contracts'
        ]
    }

    # Count contracts per class
    for vuln_class in VULNERABILITY_CLASSES:
        class_dir = OUTPUT_DIR / vuln_class
        contracts = list(class_dir.glob("*.sol"))
        count = len(contracts)

        summary['classes'][vuln_class] = {
            'count': count,
            'percentage': 0.0,
            'contracts': [c.name for c in contracts]
        }
        summary['total_contracts'] += count

    # Calculate percentages
    for vuln_class in summary['classes']:
        count = summary['classes'][vuln_class]['count']
        summary['classes'][vuln_class]['percentage'] = (count / summary['total_contracts'] * 100) if summary['total_contracts'] > 0 else 0

    # Save JSON summary
    summary_file = OUTPUT_DIR / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal Contracts: {summary['total_contracts']}")
    print("\nClass Distribution:")
    print("-" * 80)
    print(f"{'Class':<35} {'Count':>8} {'Percentage':>12}")
    print("-" * 80)

    vulnerable_count = 0
    safe_count = 0

    for vuln_class in sorted(summary['classes'].keys()):
        count = summary['classes'][vuln_class]['count']
        pct = summary['classes'][vuln_class]['percentage']
        print(f"{vuln_class:<35} {count:>8} {pct:>11.2f}%")

        if vuln_class == 'safe':
            safe_count = count
        else:
            vulnerable_count += count

    print("-" * 80)
    print(f"{'VULNERABLE (all types)':<35} {vulnerable_count:>8} {vulnerable_count/summary['total_contracts']*100:>11.2f}%")
    print(f"{'SAFE':<35} {safe_count:>8} {safe_count/summary['total_contracts']*100:>11.2f}%")
    print("=" * 80)

    print(f"\n✓ Summary saved to: {summary_file}")

    return summary


def create_train_val_test_splits(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Split dataset into train/val/test sets"""
    print("\n" + "="*80)
    print("Creating Train/Val/Test Splits")
    print("="*80)
    print(f"Train: {train_ratio*100:.0f}% | Val: {val_ratio*100:.0f}% | Test: {test_ratio*100:.0f}%")

    splits = {
        'train': defaultdict(list),
        'val': defaultdict(list),
        'test': defaultdict(list)
    }

    # For each class, split contracts
    for vuln_class in VULNERABILITY_CLASSES:
        class_dir = OUTPUT_DIR / vuln_class
        contracts = list(class_dir.glob("*.sol"))

        if len(contracts) == 0:
            continue

        # Shuffle contracts
        random.shuffle(contracts)

        # Calculate split sizes
        n_total = len(contracts)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Split
        train_contracts = contracts[:n_train]
        val_contracts = contracts[n_train:n_train+n_val]
        test_contracts = contracts[n_train+n_val:]

        splits['train'][vuln_class] = [c.name for c in train_contracts]
        splits['val'][vuln_class] = [c.name for c in val_contracts]
        splits['test'][vuln_class] = [c.name for c in test_contracts]

        print(f"\n{vuln_class}:")
        print(f"  Total: {n_total} | Train: {len(train_contracts)} | Val: {len(val_contracts)} | Test: {len(test_contracts)}")

    # Save splits to JSON
    splits_file = OUTPUT_DIR / "train_val_test_splits.json"
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"\n✓ Splits saved to: {splits_file}")

    return splits


def main():
    """Main function"""
    print("\n" + "="*80)
    print("COMBINE ALL LABELED DATASETS")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)

    # Create directories
    create_output_directories()

    # Copy datasets
    total = 0
    total += copy_smartbugs_curated()
    total += copy_smartbugs_samples()
    total += copy_solidifi()
    total += copy_not_so_smart_contracts()

    # Generate summary
    summary = generate_summary()

    # Create splits
    splits = create_train_val_test_splits()

    # Final message
    print("\n" + "="*80)
    print("✓ DATASET COMBINATION COMPLETE!")
    print("="*80)
    print(f"\nCombined {total} labeled contracts into:")
    print(f"  {OUTPUT_DIR}")
    print(f"\nDataset includes:")
    print(f"  - {len(VULNERABILITY_CLASSES)} vulnerability classes")
    print(f"  - Train/Val/Test splits (70/15/15)")
    print(f"  - Complete metadata in JSON files")
    print(f"\nNext steps:")
    print(f"  1. Review: cat {OUTPUT_DIR}/dataset_summary.json")
    print(f"  2. Train: python scripts/train_complete_pipeline.py --train-dir {OUTPUT_DIR}")
    print(f"  3. Test: python scripts/test_dataset_performance.py --custom-dir {OUTPUT_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
