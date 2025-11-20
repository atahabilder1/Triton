# Configuration System Update

## What Changed

The training script `scripts/train/static/train_static_optimized.py` now uses centralized configuration from `config.yaml`.

## Before vs After

### Before (Hardcoded):
```bash
python scripts/train/static/train_static_optimized.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --test-dir data/datasets/forge_balanced_accurate/test \
    --batch-size 16 \
    --learning-rate 0.001 \
    --num-epochs 50
```

### After (Config-Based):
```bash
# All settings come from config.yaml!
python scripts/train/static/train_static_optimized.py
```

Or override specific values:
```bash
python scripts/train/static/train_static_optimized.py \
    --batch-size 32 \
    --learning-rate 0.0005
```

## Quick Start

### 1. Edit Your Dataset Path (One Time Setup)

Edit `config.yaml`:
```yaml
data:
  train_dir: "data/datasets/your_dataset/train"
  val_dir: "data/datasets/your_dataset/val"
  test_dir: "data/datasets/your_dataset/test"
```

### 2. Run Training

```bash
# Simple - uses all defaults from config
./start_static_training_simple.sh

# Or run directly
python scripts/train/static/train_static_optimized.py
```

### 3. Experiment with Hyperparameters

Edit `config.yaml` to try different settings:
```yaml
training:
  batch_size: 32        # Try larger batch
  learning_rate: 0.0005 # Try different LR
  num_epochs: 100       # Train longer
```

No code changes needed!

## Key Benefits

1. **Single Source of Truth** - All paths in `config.yaml`
2. **No Hardcoded Paths** - Easy to switch datasets
3. **Version Controlled** - Track config changes in git
4. **Environment Flexibility** - Different configs for different machines
5. **Command-line Override** - Can still override any setting

## Files Modified

- ✅ `scripts/train/static/train_static_optimized.py` - Updated to use config
- ✅ `config.yaml` - Central configuration file
- ✅ `utils/config.py` - Configuration loader utility
- ✅ `start_static_training_simple.sh` - Simple launcher script
- ✅ `docs/CONFIG_GUIDE.md` - Full documentation

## Next Steps

To update other training scripts:
```python
from utils.config import get_config

config = get_config()
train_dir = config.train_dir
batch_size = config.batch_size
```

See `docs/CONFIG_GUIDE.md` for complete examples.
