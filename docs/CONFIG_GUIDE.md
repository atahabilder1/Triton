# Configuration Guide

Triton uses a centralized YAML configuration file (`config.yaml`) for easy management of paths, hyperparameters, and settings.

## Quick Start

### 1. Edit Configuration

Edit `config.yaml` in the project root to customize:
- Dataset paths
- Model checkpoint locations
- Training hyperparameters
- Architecture settings

```yaml
# Example: Change dataset paths
data:
  train_dir: "data/datasets/my_custom_dataset/train"
  val_dir: "data/datasets/my_custom_dataset/val"
  test_dir: "data/datasets/my_custom_dataset/test"

# Example: Adjust training settings
training:
  batch_size: 32
  learning_rate: 0.0005
  num_epochs: 100
```

### 2. Use in Python Scripts

```python
from utils.config import get_config, load_config

# Option 1: Auto-load default config.yaml
config = get_config()

# Option 2: Load specific config file
config = load_config("path/to/custom_config.yaml")

# Access configuration values
train_dir = config.train_dir
batch_size = config.batch_size
learning_rate = config.learning_rate

# Or use dot notation
cache_dir = config.get('data.cache_dir')
static_model = config.get('models.static_encoder')
```

### 3. Use in Shell Scripts

```bash
#!/bin/bash
# No changes needed - Python scripts will read config.yaml automatically
python scripts/train/static/train_static_optimized.py
```

## Configuration Sections

### Data Paths
All dataset and cache directories:
```yaml
data:
  base_dir: "data"
  train_dir: "data/datasets/forge_balanced_accurate/train"
  val_dir: "data/datasets/forge_balanced_accurate/val"
  test_dir: "data/datasets/forge_balanced_accurate/test"
  cache_dir: "data/cache"
```

### Model Paths
Checkpoint locations for all models:
```yaml
models:
  checkpoints_dir: "models/checkpoints"
  static_encoder: "models/checkpoints/static_encoder_best.pt"
  dynamic_encoder: "models/checkpoints/dynamic_encoder_best.pt"
  semantic_encoder: "models/checkpoints/semantic_encoder_best.pt"
```

### Training Settings
All hyperparameters in one place:
```yaml
training:
  batch_size: 16
  learning_rate: 0.001
  num_epochs: 50
  early_stopping_patience: 5
  num_workers: 4
```

### Architecture Configuration
Model-specific settings:
```yaml
architecture:
  static_encoder:
    hidden_dim: 256
    output_dim: 768
    dropout: 0.2
```

## Example: Update Training Script

### Before (Hardcoded):
```python
train_dataset = StaticDataset(
    contracts_dir="data/datasets/forge_balanced_accurate/train",
    cache_dir="data/cache"
)

detector = StaticVulnerabilityDetector(
    output_dir="models/checkpoints",
    batch_size=16,
    learning_rate=0.001
)
```

### After (Using Config):
```python
from utils.config import get_config

config = get_config()

train_dataset = StaticDataset(
    contracts_dir=str(config.train_dir),
    cache_dir=str(config.cache_dir)
)

detector = StaticVulnerabilityDetector(
    output_dir=str(config.checkpoints_dir),
    batch_size=config.batch_size,
    learning_rate=config.learning_rate
)
```

## Benefits

1. **Single Source of Truth** - All paths and settings in one file
2. **Easy Experimentation** - Change hyperparameters without editing code
3. **Environment Flexibility** - Different configs for dev/prod
4. **Version Control Friendly** - Track configuration changes in git
5. **No Hardcoded Paths** - Paths adapt to your system

## Advanced Usage

### Multiple Environments

Create different config files for different environments:

```bash
config.yaml              # Default
config.dev.yaml         # Development
config.production.yaml  # Production
config.gpu_server.yaml  # High-performance GPU server
```

Load specific config:
```python
from utils.config import load_config

config = load_config("config.production.yaml")
```

### Override from Command Line

You can still override config values via command-line args:

```python
import argparse
from utils.config import get_config

config = get_config()
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=config.batch_size)
parser.add_argument("--learning-rate", type=float, default=config.learning_rate)
args = parser.parse_args()
```

### Access Nested Values

Use dot notation for nested configuration:
```python
# Get specific architecture setting
hidden_dim = config.get('architecture.static_encoder.hidden_dim')

# Get with default value
max_samples = config.get('processing.max_samples', default=None)
```

## Tips

1. **Use relative paths** in config.yaml (relative to project root)
2. **Keep sensitive data out** - Don't commit API keys or credentials
3. **Document custom settings** - Add comments in YAML for clarity
4. **Validate paths** - Config loader will create missing directories when needed
5. **Version your config** - Track changes in git to see experiment history
