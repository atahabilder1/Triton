# Triton Training Guide

## Quick Start - Unified Training Interface

Triton now has a unified training system with configuration management!

### Simple Usage

```bash
# Train static encoder (PDG-based GAT)
./start_training.sh static

# Train dynamic encoder (Execution traces)
./start_training.sh dynamic

# Train semantic encoder (CodeBERT)
./start_training.sh semantic

# Train full multi-modal system
./start_training.sh full
```

All settings come from `config.yaml` - no need to specify paths!

### With Custom Parameters

```bash
# Override specific settings
./start_training.sh static --batch-size 32 --epochs 100

# Use custom config file
./start_training.sh full --config custom_config.yaml
```

## Training Types

### 1. Static Training
- **Model**: Graph Attention Network (GAT)
- **Input**: Program Dependence Graphs (PDG) from Slither
- **Use Case**: Detect vulnerabilities from code structure
- **Command**: `./start_training.sh static`

**Config section**: `training.static` in config.yaml

### 2. Dynamic Training
- **Model**: Trace-based Neural Network
- **Input**: Execution traces from Mythril
- **Use Case**: Detect vulnerabilities from runtime behavior
- **Command**: `./start_training.sh dynamic`
- **Status**: ⚠️  Template ready, needs implementation

**Config section**: `training.dynamic` in config.yaml

### 3. Semantic Training
- **Model**: CodeBERT Transformer
- **Input**: Source code tokens
- **Use Case**: Detect vulnerabilities from code semantics
- **Command**: `./start_training.sh semantic`
- **Status**: ⚠️  Template ready, needs implementation

**Config section**: `training.semantic` in config.yaml

### 4. Full Multi-Modal Training
- **Model**: All encoders + Cross-Modal Fusion
- **Input**: PDG + Traces + Source Code
- **Use Case**: Maximum accuracy using all modalities
- **Command**: `./start_training.sh full`
- **Trains**: Static → Dynamic → Semantic → Fusion (end-to-end)

**Config section**: `training.full` in config.yaml

## Configuration

### Edit Dataset Paths (One Time Setup)

Edit `config.yaml`:
```yaml
data:
  train_dir: "data/datasets/your_dataset/train"
  val_dir: "data/datasets/your_dataset/val"
  test_dir: "data/datasets/your_dataset/test"
```

### Adjust Training Settings

Edit `config.yaml` for each training type:

```yaml
training:
  # Static-only training
  static:
    batch_size: 16
    learning_rate: 0.001
    num_epochs: 50
    early_stopping_patience: 5

  # Dynamic-only training
  dynamic:
    batch_size: 16
    learning_rate: 0.001
    num_epochs: 50

  # Semantic-only training
  semantic:
    batch_size: 8          # Smaller batch for transformers
    learning_rate: 0.00005  # Lower LR for fine-tuning
    num_epochs: 30

  # Full multi-modal training
  full:
    batch_size: 8
    learning_rate: 0.0001
    num_epochs: 100
    pretrain_epochs: 20
    fusion_epochs: 80
```

## Directory Structure

```
scripts/train/
├── static/
│   ├── train_static_optimized.py  ✅ Config-enabled
│   └── train_static_only.py
├── dynamic/
│   └── train_dynamic.py            ⚠️  Template only
├── semantic/
│   └── train_semantic.py           ⚠️  Template only
└── full/
    └── train_complete_pipeline.py  ✅ Minimal config support
```

## Training Scripts

### Static Training (Fully Implemented)

```bash
# Use config defaults
python scripts/train/static/train_static_optimized.py

# Override settings
python scripts/train/static/train_static_optimized.py \
    --batch-size 32 \
    --learning-rate 0.0005 \
    --num-epochs 100
```

### Dynamic Training (Template)

```bash
python scripts/train/dynamic/train_dynamic.py
```

Currently shows configuration and ready for implementation.

### Semantic Training (Template)

```bash
python scripts/train/semantic/train_semantic.py
```

Currently shows configuration and ready for implementation.

### Full Training

```bash
python scripts/train/full/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --test-dir data/datasets/forge_balanced_accurate/test
```

## Monitoring Training

All training runs create logs and TensorBoard data:

```bash
# View logs
tail -f logs/static_training_*.log

# Launch TensorBoard
tensorboard --logdir runs/

# Quick status
./scripts/monitoring/quick_status.sh
```

## Output Files

After training completes:

```
models/checkpoints/
├── static_encoder_best.pt
├── dynamic_encoder_best.pt
├── semantic_encoder_best.pt
├── fusion_module_best.pt
└── test_results_*.txt

runs/
├── static_optimized_*/
├── dynamic_*/
├── semantic_*/
└── triton_*/          # Full training

logs/
├── static_training_*.log
├── dynamic_training_*.log
├── semantic_training_*.log
└── full_training_*.log
```

## Tips

1. **Start with static training** - It's the most developed
2. **Check GPU availability** - Training will auto-detect and use GPU
3. **Use caching** - PDG/trace extraction is cached for faster re-runs
4. **Monitor early** - Use TensorBoard to see if model is learning
5. **Adjust batch size** - Increase if you have more GPU memory
6. **Use early stopping** - Configured in config.yaml to prevent overfitting

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` in config.yaml
- Reduce `num_workers` if using CPU

### Slow Training
- Enable GPU if available
- Increase `num_workers` for faster data loading
- Enable caching (`use_cache: true`)

### Poor Accuracy
- Increase `num_epochs`
- Adjust `learning_rate`
- Check class distribution in dataset
- Ensure `use_class_weights: true` for imbalanced data

## Next Steps

1. **Implement dynamic training** - Complete `scripts/train/dynamic/train_dynamic.py`
2. **Implement semantic training** - Complete `scripts/train/semantic/train_semantic.py`
3. **Full pipeline** - Update `train_complete_pipeline.py` to use config fully
4. **Experiment** - Try different hyperparameters in config.yaml
5. **Compare** - Run all 4 modes and compare results
