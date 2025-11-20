# Dataset Preprocessing Workflow

## Current State of Your Datasets

| Dataset | Size | Interfaces Removed? | Flattened? | Ready? |
|---------|------|-------------------|------------|--------|
| `combined_labeled` | 228 | ✅ Yes (curated) | ✅ Yes | ✅ **Ready!** |
| `forge_filtered` | 3,746 | ✅ **Yes** | ❌ **No** | ⚠️ Needs flattening |
| `forge_balanced_accurate` | 7,013 | ❌ No (43% interfaces) | ❌ No | ❌ Needs both |

---

## Understanding the Issues

### Issue 1: Abstract Contracts & Interfaces ✅ SOLVED in forge_filtered
**Problem**: Contracts like this fail to compile:
```solidity
interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
}
```

**Solution**: `forge_filtered` already removed these!
- Removed 79.4% interfaces
- Removed 0.9% abstract contracts
- Removed 14.4% too-small contracts

### Issue 2: Missing Dependencies ❌ STILL EXISTS in forge_filtered
**Problem**: Contracts with imports that aren't resolved:
```solidity
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";  // ← Missing!
import "./SafeMath.sol";  // ← Missing!

contract MyToken is ERC20 {
    // ... code ...
}
```

**Solution**: Need to **flatten** - combine all imports into one file

---

## 🎯 Recommended Preprocessing Path

### Path A: Quick Start (Use combined_labeled)
**For**: Testing your pipeline, quick experiments

```bash
# Already clean and ready!
./start_training.sh static --train-dir data/datasets/combined_labeled/train
```

**Pros**: Immediate, no preprocessing
**Cons**: Only 228 contracts - too small for serious training

---

### Path B: Medium Scale (Preprocess forge_filtered) ⭐ **RECOMMENDED**
**For**: Serious training with good quality/size balance

```bash
# Step 1: Flatten contracts (resolve imports)
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_filtered/train \
    --output data/datasets/forge_filtered_flat/train \
    --tool simple

python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_filtered/val \
    --output data/datasets/forge_filtered_flat/val \
    --tool simple

python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_filtered/test \
    --output data/datasets/forge_filtered_flat/test \
    --tool simple

# Step 2: Verify (check that PDG/AST extraction works)
./verify_contracts.sh data/datasets/forge_filtered_flat/train --max 100

# Step 3: If >80% success, train!
./start_training.sh static --train-dir data/datasets/forge_filtered_flat/train
```

**Pros**:
- 3,746 contracts (good size)
- Already filtered (no interfaces/abstract)
- Only needs flattening

**Cons**:
- Flattening may fail for some contracts
- Still need to verify

---

### Path C: Maximum Scale (Preprocess forge_balanced_accurate)
**For**: Maximum training data

```bash
# Step 1: Filter (remove interfaces/abstract)
python scripts/dataset/filter_dataset.py \
    --input-dir data/datasets/forge_balanced_accurate \
    --output-dir data/datasets/forge_custom_filtered

# Step 2: Flatten
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_custom_filtered/train \
    --output data/datasets/forge_custom_flat/train \
    --tool simple

# Step 3: Validate (remove contracts that still don't compile)
python scripts/dataset/validate_contracts.py \
    data/datasets/forge_custom_flat/train \
    --output-dir data/datasets/forge_custom_clean/train \
    --copy-valid

# Step 4: Verify
./verify_contracts.sh data/datasets/forge_custom_clean/train --max 100

# Step 5: Train
./start_training.sh static --train-dir data/datasets/forge_custom_clean/train
```

**Pros**:
- Maximum contracts (~4,000+ after all filtering)
- Custom control over each step

**Cons**:
- Most time-consuming
- Multiple preprocessing steps
- May lose more contracts in validation

---

## 📊 Expected Results After Each Path

### Path A: combined_labeled (No Preprocessing)
- **Contracts**: 228
- **Expected PDG Success**: 95%+
- **Expected AST Success**: 98%+
- **Time to Train**: Fast
- **Model Quality**: Baseline

### Path B: forge_filtered + Flattening ⭐
- **Contracts**: ~3,000-3,500 (some may fail flattening)
- **Expected PDG Success**: 85-90%
- **Expected AST Success**: 90-95%
- **Time to Train**: Medium
- **Model Quality**: **Good**

### Path C: forge_balanced_accurate (Full Pipeline)
- **Contracts**: ~4,000-5,000 (after all filtering)
- **Expected PDG Success**: 80-85%
- **Expected AST Success**: 85-90%
- **Time to Train**: Slow
- **Model Quality**: Best (if successful)

---

## 🚀 Step-by-Step: Recommended Path B

### Step 1: Flatten forge_filtered
```bash
cd /home/anik/code/Triton

# Flatten train set
echo "Flattening train set..."
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_filtered/train \
    --output data/datasets/forge_filtered_flat/train \
    --tool simple \
    --batch

# Flatten val set
echo "Flattening val set..."
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_filtered/val \
    --output data/datasets/forge_filtered_flat/val \
    --tool simple \
    --batch

# Flatten test set
echo "Flattening test set..."
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_filtered/test \
    --output data/datasets/forge_filtered_flat/test \
    --tool simple \
    --batch
```

**What this does**:
- Reads each contract
- Finds all imports
- Combines them into a single file
- Copies to output directory

**Expected**: ~80-90% success rate (some imports may not resolve)

### Step 2: Verify Flattened Dataset
```bash
# Verify 100 random contracts from train set
./verify_contracts.sh data/datasets/forge_filtered_flat/train --max 100 --output forge_filtered_flat_report.json
```

**Look for**:
- PDG Success: Should be **85%+**
- AST Success: Should be **90%+**
- Both Success: Should be **80%+**

If these numbers are good, proceed to training!

### Step 3: Train Model
```bash
# Start static encoder training
./start_training.sh static --train-dir data/datasets/forge_filtered_flat/train

# Monitor in another terminal
./scripts/monitor_training.sh
tensorboard --logdir runs/
```

---

## 🔧 Troubleshooting

### If Flattening Fails for Many Contracts

Try different flattening tools:

```bash
# Try Forge (better for newer contracts)
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_filtered/train \
    --output data/datasets/forge_filtered_flat_forge/train \
    --tool forge

# Try sol-merger (better for older contracts)
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_filtered/train \
    --output data/datasets/forge_filtered_flat_merger/train \
    --tool sol-merger
```

### If PDG/AST Success Rate is Low (<70%)

Add validation step:

```bash
# Validate flattened contracts
python scripts/dataset/validate_contracts.py \
    data/datasets/forge_filtered_flat/train \
    --output-dir data/datasets/forge_filtered_validated/train \
    --copy-valid

# Re-verify
./verify_contracts.sh data/datasets/forge_filtered_validated/train --max 100
```

### If Training Fails

Check:
1. **GPU memory**: Reduce batch_size in config.yaml
2. **Timeouts**: Increase slither_timeout in config.yaml
3. **Contract size**: Some contracts may be too large

---

## 📋 Quick Commands Reference

```bash
# 1. Quick test on combined_labeled
./start_training.sh static --train-dir data/datasets/combined_labeled/train

# 2. Flatten forge_filtered
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_filtered/train \
    --output data/datasets/forge_filtered_flat/train \
    --tool simple --batch

# 3. Verify flattened dataset
./verify_contracts.sh data/datasets/forge_filtered_flat/train --max 100

# 4. Train on flattened dataset
./start_training.sh static --train-dir data/datasets/forge_filtered_flat/train

# 5. Monitor training
./scripts/monitor_training.sh
tensorboard --logdir runs/
```

---

## 💡 My Recommendation

**Start with Path B** (forge_filtered + flattening):

1. ✅ **Already filtered** - no interfaces or abstract contracts
2. ⚠️ **Needs flattening** - but this is just one step
3. ✅ **Good size** - 3,746 contracts (enough for deep learning)
4. ✅ **Expected success** - 85%+ after flattening
5. ✅ **Faster than Path C** - one preprocessing step vs three

**Commands**:
```bash
# Flatten
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_filtered/train \
    --output data/datasets/forge_filtered_flat/train \
    --tool simple --batch

# Verify
./verify_contracts.sh data/datasets/forge_filtered_flat/train --max 100

# Train
./start_training.sh static --train-dir data/datasets/forge_filtered_flat/train
```

If flattening doesn't work well, fall back to Path A (combined_labeled) for testing.

---

## 📊 Summary Table

| Dataset | Step 1 | Step 2 | Step 3 | Total Steps | Expected Final Size |
|---------|--------|--------|--------|-------------|---------------------|
| **combined_labeled** | ✅ Ready | - | - | **0** | 228 |
| **forge_filtered** | Flatten | Verify | Train | **2** | ~3,000 |
| **forge_balanced** | Filter | Flatten | Validate + Verify | **4** | ~4,000 |

**Recommendation**: Use **forge_filtered** - good balance of quality and effort!

---

**Last Updated**: November 19, 2025
**Status**: forge_filtered needs flattening only
