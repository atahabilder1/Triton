#!/usr/bin/env python3
"""
Reorganize Combined Dataset into Physical Train/Val/Test Folders

This script:
1. Reads the train_val_test_splits.json
2. Creates train/, val/, test/ folders
3. Copies contracts to appropriate folders based on splits
4. Maintains class subdirectories in each split
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "combined_labeled"
SPLITS_FILE = DATASET_DIR / "train_val_test_splits.json"

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


def create_split_directories():
    """Create train/val/test directory structure"""
    print("\n" + "="*100)
    print("📁 CREATING TRAIN/VAL/TEST FOLDER STRUCTURE")
    print("="*100)

    for split in ['train', 'val', 'test']:
        split_dir = DATASET_DIR / split

        if split_dir.exists():
            print(f"\n⚠️  {split}/ already exists. Removing old structure...")
            shutil.rmtree(split_dir)

        split_dir.mkdir()
        print(f"\n✓ Created {split}/ directory")

        # Create class subdirectories
        for vuln_class in VULNERABILITY_CLASSES:
            class_dir = split_dir / vuln_class
            class_dir.mkdir()

        print(f"  ✓ Created {len(VULNERABILITY_CLASSES)} class subdirectories")


def copy_contracts_to_splits():
    """Copy contracts to train/val/test folders based on splits"""
    print("\n" + "="*100)
    print("📋 LOADING SPLIT ASSIGNMENTS")
    print("="*100)

    # Load splits
    with open(SPLITS_FILE, 'r') as f:
        splits = json.load(f)

    print(f"\n✓ Loaded split assignments from: {SPLITS_FILE}")

    # Statistics
    stats = {
        'train': defaultdict(int),
        'val': defaultdict(int),
        'test': defaultdict(int)
    }

    print("\n" + "="*100)
    print("📦 COPYING CONTRACTS TO SPLITS")
    print("="*100)

    # Copy contracts for each split
    for split_name in ['train', 'val', 'test']:
        print(f"\n{split_name.upper()} SET:")
        print("-"*100)

        split_data = splits[split_name]

        for vuln_class in VULNERABILITY_CLASSES:
            contracts = split_data.get(vuln_class, [])

            if not contracts:
                print(f"  {vuln_class:<35}: 0 contracts (empty)")
                continue

            for contract_name in contracts:
                # Source: flat class directory
                source_file = DATASET_DIR / vuln_class / contract_name

                # Destination: split/class directory
                dest_file = DATASET_DIR / split_name / vuln_class / contract_name

                if not source_file.exists():
                    print(f"  ⚠️  WARNING: {contract_name} not found in {vuln_class}/")
                    continue

                # Copy file
                shutil.copy2(source_file, dest_file)
                stats[split_name][vuln_class] += 1

            count = stats[split_name][vuln_class]
            print(f"  {vuln_class:<35}: {count:>3} contracts")

    return stats


def verify_splits(stats):
    """Verify that all contracts were copied correctly"""
    print("\n" + "="*100)
    print("✅ VERIFICATION")
    print("="*100)

    # Count total contracts
    train_total = sum(stats['train'].values())
    val_total = sum(stats['val'].values())
    test_total = sum(stats['test'].values())
    total = train_total + val_total + test_total

    print(f"\n📊 Split Statistics:")
    print(f"  Train: {train_total:>3} contracts ({train_total/total*100:>5.1f}%)")
    print(f"  Val:   {val_total:>3} contracts ({val_total/total*100:>5.1f}%)")
    print(f"  Test:  {test_total:>3} contracts ({test_total/total*100:>5.1f}%)")
    print(f"  TOTAL: {total:>3} contracts")

    # Verify counts match
    expected_total = 228
    if total == expected_total:
        print(f"\n✓ Success! All {expected_total} contracts copied correctly.")
    else:
        print(f"\n⚠️  Warning: Expected {expected_total} contracts, but got {total}!")

    # Check for empty classes
    print(f"\n🔍 Checking for empty classes:")
    for split_name in ['train', 'val', 'test']:
        empty_classes = [cls for cls in VULNERABILITY_CLASSES if stats[split_name].get(cls, 0) == 0]
        if empty_classes:
            print(f"  {split_name}: Empty classes: {', '.join(empty_classes)}")
        else:
            print(f"  {split_name}: ✓ All classes have samples")


def print_new_structure():
    """Print the new directory structure"""
    print("\n" + "="*100)
    print("📂 NEW DIRECTORY STRUCTURE")
    print("="*100)

    print(f"\n{DATASET_DIR}/")
    print("├── train/")
    for vuln_class in VULNERABILITY_CLASSES[:3]:
        train_count = len(list((DATASET_DIR / "train" / vuln_class).glob("*.sol")))
        print(f"│   ├── {vuln_class}/ ({train_count} contracts)")
    print("│   └── ... (all 10 classes)")
    print("│")
    print("├── val/")
    for vuln_class in VULNERABILITY_CLASSES[:3]:
        val_count = len(list((DATASET_DIR / "val" / vuln_class).glob("*.sol")))
        print(f"│   ├── {vuln_class}/ ({val_count} contracts)")
    print("│   └── ... (all 10 classes)")
    print("│")
    print("├── test/")
    for vuln_class in VULNERABILITY_CLASSES[:3]:
        test_count = len(list((DATASET_DIR / "test" / vuln_class).glob("*.sol")))
        print(f"│   ├── {vuln_class}/ ({test_count} contracts)")
    print("│   └── ... (all 10 classes)")
    print("│")
    print("├── access_control/ (original flat structure - BACKUP)")
    print("├── arithmetic/")
    print("└── ... (all 10 classes)")


def print_usage_instructions():
    """Print instructions for using the new structure"""
    print("\n" + "="*100)
    print("🎓 HOW TO USE THE NEW STRUCTURE")
    print("="*100)

    print("\n✅ Training (on train set):")
    print("   python scripts/train_complete_pipeline.py \\")
    print("       --train-dir data/datasets/combined_labeled/train \\")
    print("       --num-epochs 20 \\")
    print("       --batch-size 4")

    print("\n✅ Validation (during training):")
    print("   The script will automatically use train/ for training")
    print("   You can add --val-dir if you want to specify validation separately")

    print("\n✅ Testing (after training):")
    print("   python scripts/test_dataset_performance.py \\")
    print("       --dataset custom \\")
    print("       --custom-dir data/datasets/combined_labeled/test")

    print("\n✅ Full Pipeline:")
    print("   1. Train: Use train/ folder (155 contracts)")
    print("   2. Validate: Use val/ folder during training (29 contracts)")
    print("   3. Test: Use test/ folder for final evaluation (44 contracts)")

    print("\n📋 Note:")
    print("   • Original flat structure (access_control/, arithmetic/, etc.) is kept as backup")
    print("   • train/val/test splits are now physical folders")
    print("   • Same split every time = reproducible results!")


def main():
    """Main function"""
    print("\n" + "="*100)
    print("🔀 REORGANIZE DATASET INTO TRAIN/VAL/TEST FOLDERS")
    print("="*100)
    print(f"\nDataset directory: {DATASET_DIR}")
    print(f"Splits file: {SPLITS_FILE}")
    print("="*100)

    # Check if splits file exists
    if not SPLITS_FILE.exists():
        print(f"\n❌ Error: {SPLITS_FILE} not found!")
        print("   Please run combine_labeled_datasets.py first.")
        return

    # Create directories
    create_split_directories()

    # Copy contracts
    stats = copy_contracts_to_splits()

    # Verify
    verify_splits(stats)

    # Print structure
    print_new_structure()

    # Print usage
    print_usage_instructions()

    print("\n" + "="*100)
    print("✅ DATASET REORGANIZATION COMPLETE!")
    print("="*100)
    print("\nNext step: Start training!")
    print("  python scripts/train_complete_pipeline.py \\")
    print("      --train-dir data/datasets/combined_labeled/train \\")
    print("      --num-epochs 20")
    print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    main()
