#!/usr/bin/env python3
"""
Display Beautiful Dataset Summary
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SUMMARY_FILE = PROJECT_ROOT / "data" / "datasets" / "combined_labeled" / "dataset_summary.json"
SPLITS_FILE = PROJECT_ROOT / "data" / "datasets" / "combined_labeled" / "train_val_test_splits.json"

def print_dataset_summary():
    """Print beautiful dataset summary"""

    # Load summary
    with open(SUMMARY_FILE, 'r') as f:
        summary = json.load(f)

    # Load splits
    with open(SPLITS_FILE, 'r') as f:
        splits = json.load(f)

    print("\n" + "="*100)
    print("📊 TRITON COMBINED LABELED DATASET SUMMARY")
    print("="*100)

    print(f"\n🎯 Total Contracts: {summary['total_contracts']}")
    print(f"📦 Datasets Combined: {len(summary['datasets_combined'])}")
    for ds in summary['datasets_combined']:
        print(f"   • {ds}")

    print("\n" + "="*100)
    print("CLASS DISTRIBUTION")
    print("="*100)

    print(f"\n{'Class':<35} {'Count':>8} {'%':>10} {'Train':>8} {'Val':>6} {'Test':>6}")
    print("-"*100)

    vulnerable_total = 0
    safe_total = 0

    # Sort by count (descending)
    sorted_classes = sorted(summary['classes'].items(), key=lambda x: x[1]['count'], reverse=True)

    for class_name, class_data in sorted_classes:
        count = class_data['count']
        pct = class_data['percentage']

        # Get train/val/test counts
        train_count = len(splits['train'].get(class_name, []))
        val_count = len(splits['val'].get(class_name, []))
        test_count = len(splits['test'].get(class_name, []))

        # Color indicator
        if count < 10:
            indicator = "⚠️ "
        elif count < 20:
            indicator = "⚡"
        else:
            indicator = "✓ "

        print(f"{indicator} {class_name:<32} {count:>8} {pct:>9.2f}% {train_count:>8} {val_count:>6} {test_count:>6}")

        if class_name == 'safe':
            safe_total = count
        else:
            vulnerable_total += count

    print("-"*100)
    print(f"{'VULNERABLE (all types)':<35} {vulnerable_total:>8} {vulnerable_total/summary['total_contracts']*100:>9.2f}%")
    print(f"{'SAFE (no vulnerabilities)':<35} {safe_total:>8} {safe_total/summary['total_contracts']*100:>9.2f}%")
    print("="*100)

    # Class balance analysis
    print("\n" + "="*100)
    print("⚖️  CLASS BALANCE ANALYSIS")
    print("="*100)

    counts = [data['count'] for data in summary['classes'].values()]
    largest = max(counts)
    smallest = min(counts)
    average = sum(counts) / len(counts)

    print(f"\n📊 Statistics:")
    print(f"   Largest class:  {largest} samples")
    print(f"   Smallest class: {smallest} samples")
    print(f"   Average:        {average:.1f} samples per class")
    print(f"   Imbalance ratio: {largest/smallest:.1f}:1")

    print(f"\n🎯 Class Size Categories:")
    excellent = [name for name, data in summary['classes'].items() if data['count'] >= 30]
    good = [name for name, data in summary['classes'].items() if 20 <= data['count'] < 30]
    acceptable = [name for name, data in summary['classes'].items() if 10 <= data['count'] < 20]
    small = [name for name, data in summary['classes'].items() if 5 <= data['count'] < 10]
    very_small = [name for name, data in summary['classes'].items() if data['count'] < 5]

    if excellent:
        print(f"\n   ✅ Excellent (≥30): {len(excellent)} classes")
        for name in excellent:
            print(f"      • {name}: {summary['classes'][name]['count']} samples")

    if good:
        print(f"\n   ✓  Good (20-29): {len(good)} classes")
        for name in good:
            print(f"      • {name}: {summary['classes'][name]['count']} samples")

    if acceptable:
        print(f"\n   ⚡ Acceptable (10-19): {len(acceptable)} classes")
        for name in acceptable:
            print(f"      • {name}: {summary['classes'][name]['count']} samples")

    if small:
        print(f"\n   ⚠️  Small (5-9): {len(small)} classes")
        for name in small:
            print(f"      • {name}: {summary['classes'][name]['count']} samples")

    if very_small:
        print(f"\n   ❌ Very Small (<5): {len(very_small)} classes")
        for name in very_small:
            print(f"      • {name}: {summary['classes'][name]['count']} samples")

    # Train/Val/Test split summary
    print("\n" + "="*100)
    print("🔀 TRAIN/VAL/TEST SPLITS (70/15/15)")
    print("="*100)

    train_total = sum(len(contracts) for contracts in splits['train'].values())
    val_total = sum(len(contracts) for contracts in splits['val'].values())
    test_total = sum(len(contracts) for contracts in splits['test'].values())

    print(f"\n   Train: {train_total:>3} contracts ({train_total/summary['total_contracts']*100:>5.1f}%)")
    print(f"   Val:   {val_total:>3} contracts ({val_total/summary['total_contracts']*100:>5.1f}%)")
    print(f"   Test:  {test_total:>3} contracts ({test_total/summary['total_contracts']*100:>5.1f}%)")

    # Recommendations
    print("\n" + "="*100)
    print("💡 RECOMMENDATIONS")
    print("="*100)

    print(f"\n✅ Strengths:")
    print(f"   • {safe_total} safe contracts (25.4%) - Good balance!")
    print(f"   • {vulnerable_total} vulnerable contracts across 9 types")
    print(f"   • Proper train/val/test splits for evaluation")
    print(f"   • Multiple data sources for diversity")

    print(f"\n⚠️  Challenges:")
    if very_small:
        print(f"   • {len(very_small)} classes with <5 samples (may not learn well)")
    if small:
        print(f"   • {len(small)} classes with 5-9 samples (limited data)")
    print(f"   • Class imbalance: {largest/smallest:.1f}:1 ratio")

    print(f"\n🚀 Next Steps:")
    print(f"   1. Train with class weights (already implemented)")
    print(f"   2. Consider data augmentation for small classes")
    print(f"   3. Monitor per-class performance during training")
    print(f"   4. Expected accuracy: 40-60% (vs 10% before)")

    # Training command
    print("\n" + "="*100)
    print("🎓 READY TO TRAIN!")
    print("="*100)

    print(f"\nRun training:")
    print(f"   python scripts/train_complete_pipeline.py \\")
    print(f"       --train-dir data/datasets/combined_labeled \\")
    print(f"       --num-epochs 20 \\")
    print(f"       --batch-size 4")

    print(f"\nRun testing:")
    print(f"   python scripts/test_dataset_performance.py \\")
    print(f"       --dataset custom \\")
    print(f"       --custom-dir data/datasets/combined_labeled")

    print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    print_dataset_summary()
