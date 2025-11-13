# Dataset Guide - What Dataset Is Used for Training & Testing?

## Default Dataset: SmartBugs-Curated

**By default, the training script uses:**
```
data/datasets/smartbugs-curated/dataset/
```

This is specified in the training script (line 527):
```python
parser.add_argument(
    "--train-dir",
    default="data/datasets/smartbugs/samples",  # Can be changed!
    help="Training data directory"
)
```

## Current Dataset: SmartBugs-Curated

### 📊 Dataset Statistics

**Total Contracts: 143**

| Vulnerability Type | # Contracts | % of Dataset |
|-------------------|-------------|--------------|
| Unchecked Low Level Calls | 52 | 36.4% |
| Reentrancy | 31 | 21.7% |
| Access Control | 18 | 12.6% |
| Arithmetic (Overflow/Underflow) | 15 | 10.5% |
| Bad Randomness | 8 | 5.6% |
| Denial of Service | 6 | 4.2% |
| Time Manipulation | 5 | 3.5% |
| Front Running | 4 | 2.8% |
| Other | 3 | 2.1% |
| Short Addresses | 1 | 0.7% |

### 📁 Dataset Structure

```
data/datasets/smartbugs-curated/dataset/
├── access_control/           (18 contracts)
│   ├── 0x01f8c4e3fa3edeb29e514cba738d87ce8c091d3f.sol
│   ├── 0x23a91059fdc9579a9fbd0edc5f2ea0bfdb70deb4.sol
│   └── ...
├── arithmetic/               (15 contracts)
│   ├── overflow_simple_add.sol
│   ├── integer_overflow_mul.sol
│   └── ...
├── reentrancy/              (31 contracts)
│   ├── DAO.sol
│   ├── simple_dao.sol
│   └── ...
├── unchecked_low_level_calls/ (52 contracts)
│   └── ...
└── [other vulnerability types...]
```

## How Training/Testing Split Works

### Automatic 80/20 Split

The training script automatically splits the dataset:

```python
# From train_complete_pipeline.py, lines 594-596
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
```

**For 143 contracts:**
- **Training set: 114 contracts** (80%)
- **Validation set: 29 contracts** (20%)

### What This Means

1. **Training Set (114 contracts)**
   - Used to train the model
   - Model sees these during backpropagation
   - Used to update weights

2. **Validation Set (29 contracts)**
   - Model never sees these during training
   - Used to evaluate generalization
   - Used to select best model (early stopping)

3. **Test Set**
   - Currently using validation set for testing
   - For production: should use separate held-out test set

## Available Datasets in Your Project

You have **multiple datasets** available:

### 1. SmartBugs-Curated ✅ (Default)
```
Path: data/datasets/smartbugs-curated/dataset/
Contracts: 143
Structure: Organized by vulnerability type
Quality: Curated, labeled
Best for: Training individual encoders
```

### 2. SmartBugs Samples
```
Path: data/datasets/smartbugs/samples/
Contracts: 50
Structure: Organized by Solidity version (0.4.x, 0.5.17, etc.)
Quality: Sample set for testing
Best for: Quick experiments
```

### 3. SmartBugs Wild
```
Path: data/datasets/smartbugs_wild/contracts/
Contracts: 47,398
Structure: Flat directory of real-world contracts
Quality: Unlabeled, diverse
Best for: Pre-training or large-scale testing
```

### 4. Other Datasets
```
- FORGE-Artifacts: Research dataset
- Securify: Academic dataset
- Solidifi: Verification dataset
- Audits: Audit reports with contracts
```

## How to Change the Dataset

### Option 1: Use Command Line Argument

```bash
# Use SmartBugs-Curated (recommended for training)
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs-curated/dataset

# Use SmartBugs Samples (quick testing)
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs/samples

# Use your own dataset
python scripts/train_complete_pipeline.py \
    --train-dir /path/to/your/contracts
```

### Option 2: Modify Default in Script

Edit `scripts/train_complete_pipeline.py` line 527:
```python
parser.add_argument(
    "--train-dir",
    default="data/datasets/smartbugs-curated/dataset",  # Change this!
    help="Training data directory"
)
```

## Recommended Dataset Strategy

### For Development/Testing (Small Dataset)
```bash
# Use SmartBugs Samples (50 contracts)
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs/samples \
    --max-samples 20 \
    --num-epochs 2 \
    --batch-size 2
```
**Time: ~5 minutes**
**Purpose: Verify everything works**

### For Training Individual Encoders (Medium Dataset)
```bash
# Use SmartBugs-Curated (143 contracts)
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs-curated/dataset \
    --train-mode all \
    --num-epochs 10
```
**Time: ~2 hours**
**Purpose: Train and evaluate encoders**

### For Production Training (Large Dataset)
```bash
# Use SmartBugs Wild (47,398 contracts)
# Note: You'll need to label this dataset first
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs_wild/contracts \
    --train-mode all \
    --num-epochs 20 \
    --batch-size 8
```
**Time: Several days**
**Purpose: Production-quality model**

## Current Default Configuration

**When you run:**
```bash
python scripts/train_complete_pipeline.py --train-mode all
```

**What happens:**

1. **Dataset**: SmartBugs Samples (50 contracts)
   - Path: `data/datasets/smartbugs/samples`
   - Why: Default in script (can be changed)

2. **Split**: 80/20 automatic split
   - Training: 40 contracts
   - Validation: 10 contracts

3. **Processing**:
   - Static: Slither extracts PDG from each contract
   - Dynamic: Mythril generates execution traces
   - Semantic: Raw source code tokenized
   - Results cached in `data/cache/`

## Recommended: Use SmartBugs-Curated

**I recommend changing the default to SmartBugs-Curated** because:

✅ **Better organized** - Contracts grouped by vulnerability type
✅ **More contracts** - 143 vs 50 (better for training)
✅ **Labeled data** - Clear vulnerability classifications
✅ **Balanced classes** - Good distribution across vuln types

### Quick Fix

Run training with explicit path:
```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs-curated/dataset \
    --train-mode all \
    --num-epochs 15
```

**Or update the default permanently:**

```python
# In scripts/train_complete_pipeline.py, line 527-530
parser.add_argument(
    "--train-dir",
    default="data/datasets/smartbugs-curated/dataset",  # ← Change this line
    help="Training data directory"
)
```

## Dataset Statistics Summary

| Dataset | Contracts | Labeled | Organized | Recommended For |
|---------|-----------|---------|-----------|-----------------|
| **SmartBugs-Curated** | **143** | ✅ Yes | ✅ By vuln type | **Training** |
| SmartBugs Samples | 50 | ✅ Yes | By Solidity version | Quick tests |
| SmartBugs Wild | 47,398 | ❌ No | Flat directory | Pre-training |
| FORGE | ~1,000 | ✅ Yes | By artifact | Research |

## Training/Testing Workflow

```
┌─────────────────────────────────────────────────────┐
│ Load Dataset (143 contracts from SmartBugs-Curated) │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│ Extract Features (PDG, Traces, Code)                │
│ - Slither: Extract PDG graphs                       │
│ - Mythril: Generate execution traces                │
│ - Cache results in data/cache/                      │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│ Split Dataset (80/20)                               │
│ - Training: 114 contracts                           │
│ - Validation: 29 contracts                          │
└──────────────────┬──────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌─────────────────┐  ┌──────────────────┐
│ Train on 114    │  │ Validate on 29   │
│ contracts       │  │ contracts        │
│ (update weights)│  │ (check accuracy) │
└─────────────────┘  └──────────────────┘
         │                   │
         └─────────┬─────────┘
                   ▼
         ┌─────────────────────┐
         │ Save Best Model     │
         │ (highest val acc)   │
         └─────────────────────┘
```

## Example Training Run

```bash
# Train on SmartBugs-Curated dataset
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs-curated/dataset \
    --train-mode all \
    --num-epochs 15 \
    --batch-size 4

# Output:
# Loading contracts from data/datasets/smartbugs-curated/dataset...
# Loaded 143 contracts total
# Label distribution:
#   unchecked_low_level_calls: 52 contracts
#   reentrancy: 31 contracts
#   access_control: 18 contracts
#   ...
# Training samples: 114
# Validation samples: 29
#
# [Training proceeds...]
```

## Summary

**📌 Current Default Dataset:**
- Path: `data/datasets/smartbugs/samples`
- Size: 50 contracts
- Split: 40 training / 10 validation

**📌 Recommended Dataset:**
- Path: `data/datasets/smartbugs-curated/dataset`
- Size: 143 contracts
- Split: 114 training / 29 validation
- Better organized and more diverse

**📌 To Use Recommended Dataset:**
```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs-curated/dataset \
    --train-mode all
```

**📌 Training/Testing Split:**
- Automatic 80/20 split
- Training set: Train model weights
- Validation set: Evaluate and select best model
- Random split ensures fair evaluation
