# Triton Quick Start Guide

## TL;DR - Train Everything Now

```bash
# 1. Test that everything works (30 seconds)
python scripts/quick_test_encoders.py

# 2. Train complete pipeline (recommended)
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs/samples \
    --train-mode all \
    --num-epochs 15 \
    --batch-size 4
```

## Common Training Commands

### Test Everything Works
```bash
python scripts/quick_test_encoders.py
```
**Output:** Confirms all encoders and fusion work correctly

### Train Semantic Encoder Only (No External Tools Needed)
```bash
python scripts/train_complete_pipeline.py \
    --train-mode semantic \
    --num-epochs 5
```
**Time:** ~20 minutes (100 contracts)

### Train All Encoders Individually
```bash
# Static (requires: pip install slither-analyzer)
python scripts/train_complete_pipeline.py --train-mode static --num-epochs 10

# Dynamic (requires: pip install mythril)
python scripts/train_complete_pipeline.py --train-mode dynamic --num-epochs 10

# Semantic (no external tools)
python scripts/train_complete_pipeline.py --train-mode semantic --num-epochs 5
```

### Train Complete Fusion Pipeline
```bash
python scripts/train_complete_pipeline.py --train-mode fusion --num-epochs 15
```

### Train Everything Sequentially
```bash
python scripts/train_complete_pipeline.py --train-mode all --num-epochs 15
```
**What it does:**
1. Trains static encoder (10 epochs)
2. Trains dynamic encoder (10 epochs)
3. Trains semantic encoder (5 epochs)
4. Trains fusion module (15 epochs)

## Quick Testing (Small Dataset)

```bash
# Test with only 20 contracts, 2 epochs
python scripts/train_complete_pipeline.py \
    --max-samples 20 \
    --num-epochs 2 \
    --batch-size 2 \
    --train-mode semantic
```
**Time:** ~2 minutes

## Check Training Results

```bash
# View saved models
ls -lh models/checkpoints/

# Expected files:
# - static_encoder_best.pt
# - dynamic_encoder_best.pt
# - semantic_encoder_best.pt
# - fusion_module_best.pt
```

## Troubleshooting

### Error: "No such file or directory: 'slither'"
```bash
# Install Slither
pip install slither-analyzer

# Or train without static analysis
python scripts/train_complete_pipeline.py --train-mode semantic
```

### Error: "CUDA out of memory"
```bash
# Use smaller batch size
python scripts/train_complete_pipeline.py --batch-size 2

# Or use CPU
python scripts/train_complete_pipeline.py --device cpu
```

### Training is too slow
```bash
# Use fewer samples for testing
python scripts/train_complete_pipeline.py --max-samples 50

# Skip encoder tests
python scripts/train_complete_pipeline.py --skip-tests
```

## Full Command Reference

```bash
python scripts/train_complete_pipeline.py \
    --train-dir PATH              # Default: data/datasets/smartbugs/samples
    --output-dir PATH             # Default: models/checkpoints
    --train-mode MODE             # static|dynamic|semantic|fusion|all
    --batch-size N                # Default: 4
    --num-epochs N                # Default: 10
    --learning-rate LR            # Default: 0.001
    --max-samples N               # Limit dataset size (for testing)
    --device DEVICE               # cuda|cpu (auto-detect if not specified)
    --skip-tests                  # Don't run encoder tests first
```

## What Each Mode Does

| Mode | What It Trains | Output Files | Requires |
|------|---------------|--------------|----------|
| `semantic` | GraphCodeBERT only | `semantic_encoder_best.pt` | Nothing |
| `static` | PDG+GAT only | `static_encoder_best.pt` | Slither |
| `dynamic` | Trace+LSTM only | `dynamic_encoder_best.pt` | Mythril |
| `fusion` | All + fusion | All 4 checkpoint files | Slither + Mythril |
| `all` | Sequential training | All 4 checkpoint files | Slither + Mythril |

## Recommended Workflow

### For Development/Testing
```bash
# 1. Quick test
python scripts/quick_test_encoders.py

# 2. Small dataset test
python scripts/train_complete_pipeline.py \
    --max-samples 20 \
    --num-epochs 2 \
    --train-mode semantic
```

### For Full Training
```bash
# Install tools
pip install slither-analyzer mythril

# Train complete pipeline
python scripts/train_complete_pipeline.py \
    --train-mode all \
    --num-epochs 15
```

## Expected Training Times

| Component | Samples | Epochs | Time (GPU) | Time (CPU) |
|-----------|---------|--------|------------|------------|
| Semantic | 100 | 5 | 20 min | 60 min |
| Static | 100 | 10 | 30 min | 90 min |
| Dynamic | 100 | 10 | 45 min | 120 min |
| Fusion | 100 | 15 | 90 min | 240 min |

## Dimension Flow

```
PDG Graph    ──→ Static  (768) ─┐
Traces       ──→ Dynamic (512) ─┤
Source Code  ──→ Semantic(768) ─┴─→ Fusion ──→ 768 ──→ 10 classes
```

## After Training

```bash
# Test your trained models
python scripts/test_triton.py --model-dir models/checkpoints

# View model sizes
ls -lh models/checkpoints/

# Expected sizes:
# - Semantic: ~493 MB (GraphCodeBERT is large)
# - Static:   ~22 MB
# - Dynamic:  ~26 MB
# - Fusion:   ~38 MB
```

## Need Help?

See detailed documentation:
- `TRAINING_GUIDE.md` - Complete training guide
- `TRAINING_SUMMARY.md` - Implementation summary
- `README.md` - Project overview

## One-Liner Complete Training

```bash
python scripts/quick_test_encoders.py && \
python scripts/train_complete_pipeline.py --train-mode all --num-epochs 15
```

That's it! 🚀
