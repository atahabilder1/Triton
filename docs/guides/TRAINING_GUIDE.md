# Triton Training Guide

Complete guide for training all components of the Triton vulnerability detection system.

## Overview

Triton consists of four main components that need to be trained:
1. **Static Encoder** (768-dim) - Analyzes PDG structures using GAT
2. **Dynamic Encoder** (512-dim) - Processes execution traces using LSTM
3. **Semantic Encoder** (768-dim) - Fine-tunes GraphCodeBERT on Solidity code
4. **Fusion Module** (768-dim output) - Combines all three modalities

## Quick Start

### 1. Test All Encoders

First, verify that all encoders work correctly:

```bash
python scripts/quick_test_encoders.py
```

Expected output:
```
ALL TESTS PASSED! ✓

Dimension Summary:
  Static Encoder:   768
  Dynamic Encoder:  512
  Semantic Encoder: 768
  Fusion Output:    768
```

### 2. Train Individual Encoders

Train each encoder separately to understand their individual performance:

#### Train Static Encoder Only
```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs/samples \
    --train-mode static \
    --num-epochs 10 \
    --batch-size 8
```

#### Train Dynamic Encoder Only
```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs/samples \
    --train-mode dynamic \
    --num-epochs 10 \
    --batch-size 8
```

#### Train Semantic Encoder Only
```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs/samples \
    --train-mode semantic \
    --num-epochs 5 \
    --batch-size 8
```

### 3. Train Complete Fusion Pipeline

Train all encoders together with the fusion module:

```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs/samples \
    --train-mode all \
    --num-epochs 15 \
    --batch-size 4
```

## Training Modes

The training script supports different modes:

| Mode | Components Trained | Use Case |
|------|-------------------|----------|
| `static` | Static Encoder only | Test PDG-based analysis |
| `dynamic` | Dynamic Encoder only | Test trace-based analysis |
| `semantic` | Semantic Encoder only | Test code understanding |
| `fusion` | All + Fusion Module | End-to-end training |
| `all` | Sequential: static → dynamic → semantic → fusion | Complete pipeline |

## Dataset Structure

The script supports two dataset structures:

### SmartBugs Structure (Organized by Vulnerability Type)
```
data/datasets/smartbugs/samples/
├── reentrancy/
│   ├── contract1.sol
│   └── contract2.sol
├── access_control/
│   ├── contract3.sol
│   └── contract4.sol
└── arithmetic/
    └── contract5.sol
```

### Flat Structure
```
data/datasets/my_contracts/
├── contract1.sol
├── contract2.sol
└── contract3.sol
```

## Advanced Options

### Caching

The script uses caching to avoid re-running Slither/Mythril on the same contracts:

```bash
# Enable caching (default)
python scripts/train_complete_pipeline.py --train-dir data/datasets/smartbugs/samples

# Clear cache before training
rm -rf data/cache/*
```

Cache files are stored in `data/cache/` and contain:
- PDG graphs from Slither
- Execution traces from Mythril

### Limited Training (For Testing)

```bash
# Train on only 50 contracts for 3 epochs
python scripts/train_complete_pipeline.py \
    --max-samples 50 \
    --num-epochs 3 \
    --batch-size 4
```

### Skip Encoder Tests

```bash
# Skip individual encoder tests before training
python scripts/train_complete_pipeline.py \
    --skip-tests \
    --train-mode all
```

### GPU/CPU Selection

```bash
# Force CPU
python scripts/train_complete_pipeline.py --device cpu

# Use CUDA (default if available)
python scripts/train_complete_pipeline.py --device cuda
```

## Training Pipeline Stages

### Stage 1: Static Encoder Training

**What it does:**
- Extracts PDGs from contracts using Slither
- Trains GAT to learn graph representations
- Predicts vulnerabilities from structural patterns

**Output:**
- `models/checkpoints/static_encoder_best.pt`

**Metrics:**
- Training accuracy on graph-based classification
- Validation loss and accuracy

### Stage 2: Dynamic Encoder Training

**What it does:**
- Generates execution traces using Mythril
- Trains LSTM to learn sequential patterns
- Detects runtime vulnerabilities

**Output:**
- `models/checkpoints/dynamic_encoder_best.pt`

**Metrics:**
- Training accuracy on trace-based classification
- Validation loss and accuracy

### Stage 3: Semantic Encoder Training

**What it does:**
- Fine-tunes GraphCodeBERT on Solidity code
- Learns high-level vulnerability patterns
- Understands code semantics

**Output:**
- `models/checkpoints/semantic_encoder_best.pt`

**Metrics:**
- Training accuracy on semantic classification
- Validation loss and accuracy

### Stage 4: Fusion Module Training

**What it does:**
- Combines all three modalities
- Learns adaptive weighting per vulnerability type
- End-to-end optimization

**Output:**
- `models/checkpoints/fusion_module_best.pt`
- `models/checkpoints/static_encoder_fusion_best.pt`
- `models/checkpoints/dynamic_encoder_fusion_best.pt`
- `models/checkpoints/semantic_encoder_fusion_best.pt`

**Metrics:**
- Training accuracy on fused predictions
- Validation loss and accuracy
- Modality weight distributions

## Modality Dimensions

Understanding the dimension flow:

```
Input → Encoder → Output → Fusion → Final
```

### Static Path
```
PDG Graph → GAT (128→256→768) → 768-dim features
```

### Dynamic Path
```
Execution Trace → LSTM (128→256→512) → 512-dim features
```

### Semantic Path
```
Solidity Code → GraphCodeBERT (768) → 768-dim features
```

### Fusion
```
Static (768) ─┐
Dynamic (512) ─┼→ Projection (512) → Cross-Attention → Weighted Fusion → 768-dim output
Semantic (768)─┘
```

## Expected Results

### Individual Encoder Performance

| Encoder | Expected Val Accuracy | Training Time (10 epochs) |
|---------|----------------------|---------------------------|
| Static | 60-70% | ~30 min (100 contracts) |
| Dynamic | 55-65% | ~45 min (100 contracts) |
| Semantic | 70-80% | ~20 min (100 contracts) |

### Fusion Performance

| Metric | Expected Value |
|--------|---------------|
| Validation Accuracy | 75-85% |
| False Positive Rate | <10% |
| Training Time | ~90 min (100 contracts, 15 epochs) |

## Troubleshooting

### Issue: Slither/Mythril Not Found

```
ERROR - Slither analysis error: [Errno 2] No such file or directory: 'slither'
ERROR - Mythril analysis error: [Errno 2] No such file or directory: 'myth'
```

**Solution:**
- These errors are expected when training only the semantic encoder
- For static/dynamic/fusion training, install tools:
  ```bash
  pip install slither-analyzer
  pip install mythril
  ```

### Issue: CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce batch size: `--batch-size 2`
- Use CPU: `--device cpu`
- Train encoders individually instead of fusion

### Issue: Training Very Slow

**Solution:**
- Enable caching (default)
- Use `--max-samples` to limit dataset size during development
- Increase batch size if you have enough GPU memory
- Use `--skip-tests` to skip encoder verification

### Issue: Low Accuracy

**Possible causes:**
1. **Dataset too small** - Need at least 100 contracts per vulnerability type
2. **Imbalanced classes** - Check label distribution in logs
3. **Not enough epochs** - Try 15-20 epochs for fusion
4. **Learning rate too high** - Default 0.001 should work, try 0.0005

## Monitoring Training

Watch the training progress:

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# View training logs
tail -f training.log

# Check saved models
ls -lh models/checkpoints/
```

## Next Steps After Training

1. **Evaluate on test set:**
   ```bash
   python scripts/test_triton.py --model-dir models/checkpoints
   ```

2. **Compare encoders:**
   - Check which encoder performs best individually
   - Analyze modality weights in fusion

3. **Analyze errors:**
   - Review false positives
   - Identify missed vulnerabilities
   - Improve data preprocessing

4. **Fine-tune hyperparameters:**
   - Learning rates per component
   - Fusion layer sizes
   - Attention heads

## Configuration Reference

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-dir` | `data/datasets/smartbugs/samples` | Path to training data |
| `--output-dir` | `models/checkpoints` | Where to save models |
| `--batch-size` | 4 | Batch size for training |
| `--num-epochs` | 10 | Number of epochs |
| `--learning-rate` | 0.001 | Base learning rate |
| `--max-samples` | None | Limit number of contracts |
| `--device` | auto | `cuda` or `cpu` |
| `--skip-tests` | False | Skip encoder tests |
| `--train-mode` | `all` | What to train |

### Learning Rate Schedule

Different components use different learning rates:

| Component | Learning Rate Multiplier | Reason |
|-----------|-------------------------|--------|
| Static Encoder | 0.5× | Training from scratch |
| Dynamic Encoder | 0.5× | Training from scratch |
| Semantic Encoder | 0.1× | Fine-tuning pretrained model |
| Fusion Module | 1.0× | Training from scratch |

## Performance Optimization Tips

1. **Use caching** - Saves Slither/Mythril results
2. **Batch size** - Larger = faster, but needs more memory
3. **Parallel data loading** - Set `num_workers > 0` if stable
4. **Mixed precision** - Add `torch.cuda.amp` for faster training
5. **Gradient accumulation** - For effective larger batch sizes

## Summary

```bash
# Complete training workflow:

# 1. Test everything works
python scripts/quick_test_encoders.py

# 2. Train semantic encoder (fastest)
python scripts/train_complete_pipeline.py --train-mode semantic --num-epochs 5

# 3. Train complete pipeline
python scripts/train_complete_pipeline.py --train-mode all --num-epochs 15

# 4. Check results
ls -lh models/checkpoints/
```

Your models will be saved in `models/checkpoints/` and ready for inference!
