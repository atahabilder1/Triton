#!/usr/bin/env python3
"""
Training Monitor - Real-time performance tracking
Shows detailed metrics and comparisons during training
"""

import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List
from collections import defaultdict

def print_performance_table(history: Dict):
    """Print a nice performance comparison table"""

    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    # Static Encoder
    if 'static' in history:
        print("\n📊 STATIC ENCODER (PDG + GAT)")
        print("-" * 60)
        print(f"{'Epoch':<10} {'Train Loss':<15} {'Train Acc':<15} {'Val Loss':<15} {'Val Acc':<15}")
        print("-" * 60)
        for epoch, metrics in enumerate(history['static'], 1):
            print(f"{epoch:<10} {metrics['train_loss']:<15.4f} {metrics['train_acc']:<15.2f}% {metrics['val_loss']:<15.4f} {metrics['val_acc']:<15.2f}%")

        best_epoch = max(range(len(history['static'])), key=lambda i: history['static'][i]['val_acc'])
        print(f"\n✓ Best: Epoch {best_epoch+1} - Val Acc: {history['static'][best_epoch]['val_acc']:.2f}%")

    # Dynamic Encoder
    if 'dynamic' in history:
        print("\n📊 DYNAMIC ENCODER (Traces + LSTM)")
        print("-" * 60)
        print(f"{'Epoch':<10} {'Train Loss':<15} {'Train Acc':<15} {'Val Loss':<15} {'Val Acc':<15}")
        print("-" * 60)
        for epoch, metrics in enumerate(history['dynamic'], 1):
            print(f"{epoch:<10} {metrics['train_loss']:<15.4f} {metrics['train_acc']:<15.2f}% {metrics['val_loss']:<15.4f} {metrics['val_acc']:<15.2f}%")

        best_epoch = max(range(len(history['dynamic'])), key=lambda i: history['dynamic'][i]['val_acc'])
        print(f"\n✓ Best: Epoch {best_epoch+1} - Val Acc: {history['dynamic'][best_epoch]['val_acc']:.2f}%")

    # Semantic Encoder
    if 'semantic' in history:
        print("\n📊 SEMANTIC ENCODER (Code + GraphCodeBERT)")
        print("-" * 60)
        print(f"{'Epoch':<10} {'Train Loss':<15} {'Train Acc':<15} {'Val Loss':<15} {'Val Acc':<15}")
        print("-" * 60)
        for epoch, metrics in enumerate(history['semantic'], 1):
            print(f"{epoch:<10} {metrics['train_loss']:<15.4f} {metrics['train_acc']:<15.2f}% {metrics['val_loss']:<15.4f} {metrics['val_acc']:<15.2f}%")

        best_epoch = max(range(len(history['semantic'])), key=lambda i: history['semantic'][i]['val_acc'])
        print(f"\n✓ Best: Epoch {best_epoch+1} - Val Acc: {history['semantic'][best_epoch]['val_acc']:.2f}%")

    # Fusion
    if 'fusion' in history:
        print("\n📊 FUSION MODULE (All Modalities)")
        print("-" * 60)
        print(f"{'Epoch':<10} {'Train Loss':<15} {'Train Acc':<15} {'Val Loss':<15} {'Val Acc':<15}")
        print("-" * 60)
        for epoch, metrics in enumerate(history['fusion'], 1):
            print(f"{epoch:<10} {metrics['train_loss']:<15.4f} {metrics['train_acc']:<15.2f}% {metrics['val_loss']:<15.4f} {metrics['val_acc']:<15.2f}%")

        best_epoch = max(range(len(history['fusion'])), key=lambda i: history['fusion'][i]['val_acc'])
        print(f"\n✓ Best: Epoch {best_epoch+1} - Val Acc: {history['fusion'][best_epoch]['val_acc']:.2f}%")

    # Overall Comparison
    print("\n" + "=" * 80)
    print("OVERALL COMPARISON (Best Validation Accuracy)")
    print("=" * 80)
    print(f"{'Component':<30} {'Best Val Acc':<20} {'Improvement':<20}")
    print("-" * 80)

    results = {}
    if 'static' in history:
        best_acc = max(m['val_acc'] for m in history['static'])
        results['Static (PDG)'] = best_acc
        print(f"{'Static (PDG)':<30} {best_acc:<20.2f}% {'-':<20}")

    if 'dynamic' in history:
        best_acc = max(m['val_acc'] for m in history['dynamic'])
        results['Dynamic (Traces)'] = best_acc
        print(f"{'Dynamic (Traces)':<30} {best_acc:<20.2f}% {'-':<20}")

    if 'semantic' in history:
        best_acc = max(m['val_acc'] for m in history['semantic'])
        results['Semantic (Code)'] = best_acc
        baseline = best_acc
        print(f"{'Semantic (Code)':<30} {best_acc:<20.2f}% {'(baseline)':<20}")

    if 'fusion' in history:
        best_acc = max(m['val_acc'] for m in history['fusion'])
        results['Fusion (All)'] = best_acc
        if 'semantic' in results:
            improvement = best_acc - results['Semantic (Code)']
            print(f"{'Fusion (All)':<30} {best_acc:<20.2f}% {f'+{improvement:.2f}%':<20}")
        else:
            print(f"{'Fusion (All)':<30} {best_acc:<20.2f}% {'-':<20}")

    print("=" * 80)

def monitor_checkpoints(checkpoint_dir: str = "models/checkpoints"):
    """Monitor and display training progress from saved checkpoints"""

    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return

    print("\n" + "=" * 80)
    print("TRITON TRAINING MONITOR")
    print("=" * 80)
    print(f"Monitoring: {checkpoint_dir}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check for saved models
    models = {
        'static': checkpoint_path / 'static_encoder_best.pt',
        'dynamic': checkpoint_path / 'dynamic_encoder_best.pt',
        'semantic': checkpoint_path / 'semantic_encoder_best.pt',
        'fusion': checkpoint_path / 'fusion_module_best.pt'
    }

    print("\n📁 Saved Models:")
    print("-" * 60)

    import torch

    history = defaultdict(list)

    for name, model_path in models.items():
        if model_path.exists():
            # Load checkpoint
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                metadata = checkpoint.get('metadata', {})

                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"✓ {name.capitalize():<15} - {size_mb:>6.1f} MB - Val Acc: {metadata.get('val_acc', 0):.2f}%")

                # Build history (for demo, just show final epoch)
                history[name].append({
                    'train_loss': 0,
                    'train_acc': 0,
                    'val_loss': metadata.get('val_loss', 0),
                    'val_acc': metadata.get('val_acc', 0)
                })
            except Exception as e:
                print(f"✗ {name.capitalize():<15} - Error loading: {e}")
        else:
            print(f"✗ {name.capitalize():<15} - Not trained yet")

    print("\n" + "=" * 80)

def create_progress_summary():
    """Create a summary of training progress"""

    print("\n📈 TRAINING PROGRESS SUMMARY")
    print("=" * 80)
    print("""
When you train with --train-mode static, you'll see output like:

Epoch 1/10
Training Static Encoder: 100%|████████| 25/25 [00:45<00:00, loss=2.15]
Train Loss: 2.1500, Train Acc: 65.00%
Val Loss: 2.0800, Val Acc: 70.00%
✓ Saved best static encoder (val_loss: 2.0800)

Epoch 2/10
Training Static Encoder: 100%|████████| 25/25 [00:43<00:00, loss=2.05]
Train Loss: 2.0500, Train Acc: 68.00%
Val Loss: 1.9500, Val Acc: 72.00%
✓ Saved best static encoder (val_loss: 1.9500)
    """)

    print("\n🎯 What These Metrics Mean:")
    print("-" * 60)
    print("Train Loss:   How well the model fits training data (lower = better)")
    print("Train Acc:    % correct on training set (higher = better)")
    print("Val Loss:     How well model generalizes (lower = better)")
    print("Val Acc:      % correct on validation set (higher = better)")
    print("")
    print("✓ Symbol:     Model improved and was saved!")
    print("=" * 80)

def show_tool_usage():
    """Show what tools are used for each encoder"""

    print("\n🔧 ANALYSIS TOOLS USED")
    print("=" * 80)
    print(f"{'Encoder':<20} {'Tool':<15} {'Extracts':<30} {'Purpose':<30}")
    print("-" * 80)
    print(f"{'Static':<20} {'Slither ✓':<15} {'PDG (Graph)':<30} {'Graph structure for GAT':<30}")
    print(f"{'Dynamic':<20} {'Mythril ✓':<15} {'Execution Traces':<30} {'Opcode sequences for LSTM':<30}")
    print(f"{'Semantic':<20} {'None':<15} {'Source Code':<30} {'Raw Solidity for BERT':<30}")
    print("=" * 80)

    print("\n💡 Notes:")
    print("  - Slither: Extracts control/data flow graphs (PDG)")
    print("  - Mythril: Generates symbolic execution traces")
    print("  - Both results are cached to speed up training")
    print("  - Semantic encoder works without external tools!")

def main():
    print("\n" + "=" * 80)
    print("🔍 TRITON TRAINING PERFORMANCE MONITOR")
    print("=" * 80)

    # Show what tools are used
    show_tool_usage()

    # Show example of what you'll see during training
    create_progress_summary()

    # Check current saved models
    monitor_checkpoints()

    print("\n" + "=" * 80)
    print("📝 TO TRACK TRAINING:")
    print("=" * 80)
    print("""
1. During training, watch the console output for metrics each epoch
2. After training, run this script to see saved models:
   python scripts/monitor_training.py
3. Compare performance across encoders
4. Check if fusion improves over individual encoders
    """)
    print("=" * 80)

if __name__ == "__main__":
    main()
