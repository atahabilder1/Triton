# Dataset Quick Guide

## 📁 Your 3 Main Training Datasets

### 1️⃣ combined_labeled (High Quality, Small)
**Path**: `data/datasets/combined_labeled/`
- **Size**: 228 contracts
- **Quality**: ⭐⭐⭐⭐⭐ Manually curated
- **Abstract Contracts**: ❌ No (removed)
- **Flattened**: ✅ Yes
- **Ready to Train**: ✅ **YES!**

**Best for**: Quick testing, pipeline validation

```bash
./start_training.sh static --train-dir data/datasets/combined_labeled/train
```

---

### 2️⃣ forge_no_abstract_not_flattened (Medium, Filtered) ⭐ **RECOMMENDED**
**Path**: `data/datasets/forge_no_abstract_not_flattened/`
- **Size**: 3,746 contracts
- **Quality**: ⭐⭐⭐⭐ Auto-filtered (high quality)
- **Abstract Contracts**: ❌ **No (removed)**
- **Interfaces**: ❌ **No (removed)**
- **Tiny Stubs**: ❌ **No (removed)**
- **Flattened**: ❌ **Not yet** (needs flattening)
- **Ready to Train**: ⚠️ **After flattening**

**What was filtered out**:
- 79.4% interfaces
- 14.4% contracts with <10 lines
- 3.7% small libraries
- 1.7% contracts with no implementations
- 0.9% abstract contracts

**Best for**: Serious training (after flattening)

```bash
# Step 1: Flatten (resolve imports)
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_no_abstract_not_flattened/train \
    --output data/datasets/forge_clean/train \
    --tool simple --batch

# Step 2: Verify
./verify_contracts.sh data/datasets/forge_clean/train --max 100

# Step 3: Train
./start_training.sh static --train-dir data/datasets/forge_clean/train
```

---

### 3️⃣ forge_balanced_accurate (Large, Raw)
**Path**: `data/datasets/forge_balanced_accurate/`
- **Size**: 7,013 contracts
- **Quality**: ⭐⭐⭐ Auto-labeled
- **Abstract Contracts**: ✅ Yes (43% are interfaces!)
- **Flattened**: ❌ No
- **Ready to Train**: ❌ **Needs filtering + flattening**

**Best for**: Maximum scale (but requires most preprocessing)

---

## 🎯 Quick Decision Tree

### Want to test your pipeline?
→ Use **combined_labeled** (228 contracts, ready now)

### Want serious training with minimal work?
→ Use **forge_no_abstract_not_flattened** + flattening (3,746 contracts)

### Want maximum training data?
→ Use **forge_balanced_accurate** + filtering + flattening (7,013 contracts)

---

## 📊 Comparison Table

| Feature | combined_labeled | forge_no_abstract_not_flattened | forge_balanced_accurate |
|---------|------------------|--------------------------------|-------------------------|
| **Contracts** | 228 | 3,746 | 7,013 |
| **Abstract removed?** | ✅ Yes | ✅ **Yes** | ❌ No |
| **Interfaces removed?** | ✅ Yes | ✅ **Yes** | ❌ No (43%!) |
| **Flattened?** | ✅ Yes | ❌ No | ❌ No |
| **Steps to train** | **0** | **1** (flatten) | **2** (filter + flatten) |
| **Expected success** | 95%+ | 85-90% | 60-70% (80%+ after prep) |
| **Best for** | Testing | **Training** | Max scale |

---

## 🚀 Recommended Workflow

### Day 1: Quick Test
```bash
# Test on combined_labeled (ready immediately)
./start_training.sh static --train-dir data/datasets/combined_labeled/train
```

### Day 2: Real Training
```bash
# Use forge_no_abstract_not_flattened (best balance)

# 1. Flatten
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_no_abstract_not_flattened/train \
    --output data/datasets/forge_clean/train \
    --tool simple --batch

# 2. Verify
./verify_contracts.sh data/datasets/forge_clean/train --max 100

# 3. Train (if verification >80% success)
./start_training.sh static --train-dir data/datasets/forge_clean/train
```

---

## ✅ Key Insights

### forge_no_abstract_not_flattened is BEST because:

1. **No Abstract Contracts** ✅
   - Already removed (0.9% of dataset)
   - Won't fail compilation

2. **No Interfaces** ✅
   - Already removed (79.4% of removals!)
   - Won't produce empty PDGs

3. **No Tiny Stubs** ✅
   - Minimum 10 lines of code
   - All have meaningful implementations

4. **Still Large** ✅
   - 3,746 contracts (vs 228 in combined_labeled)
   - Enough for deep learning

5. **Only Needs Flattening** ✅
   - One preprocessing step vs two
   - Faster to get started

---

## 📁 Folder Structure (All Datasets)

All datasets use this structure (which is **correct**!):

```
dataset_name/
├── train/
│   ├── access_control/      (label 0)
│   ├── arithmetic/           (label 1)
│   ├── bad_randomness/       (label 2)
│   ├── denial_of_service/    (label 3)
│   ├── front_running/        (label 4)
│   ├── reentrancy/           (label 5)
│   ├── short_addresses/      (label 6)
│   ├── time_manipulation/    (label 7)
│   ├── unchecked_low_level_calls/ (label 8)
│   ├── other/                (label 9)
│   └── safe/                 (label 10)
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

**Why this is good**:
- Folder name = Label (automatic labeling)
- PyTorch shuffles batches (mixed types per batch)
- Easy to verify balance
- Standard ML practice

See `docs/FOLDER_STRUCTURE_MATTERS.md` for details!

---

## 🔧 All Dataset Locations

| Dataset | Contracts | Purpose |
|---------|-----------|---------|
| **combined_labeled** | 228 | Quick testing |
| **forge_no_abstract_not_flattened** | 3,746 | **Training (recommended)** |
| **forge_balanced_accurate** | 7,013 | Max scale training |
| FORGE-Artifacts | 78,223 | Research/insights |
| smartbugs-curated | 143 | (used in combined_labeled) |
| smartbugs | 50 | (used in combined_labeled) |
| solidifi | 50 | (used in combined_labeled) |
| audits | 25 | (used in combined_labeled) |
| securify | 381 | Not currently used |

---

## 💾 Quick Commands

```bash
# 1. Quick test
./start_training.sh static --train-dir data/datasets/combined_labeled/train

# 2. Flatten forge_no_abstract_not_flattened
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_no_abstract_not_flattened/train \
    --output data/datasets/forge_clean/train \
    --tool simple --batch

# 3. Verify flattened dataset
./verify_contracts.sh data/datasets/forge_clean/train --max 100

# 4. Train on flattened dataset
./start_training.sh static --train-dir data/datasets/forge_clean/train

# 5. Monitor training
./scripts/monitor_training.sh
tensorboard --logdir runs/
```

---

## 📚 Documentation

- **DATASET_COMPARISON.md** - Detailed comparison of all datasets
- **FORGE_FILTERED_EXPLANATION.md** - What filtering was done
- **DATASET_PREPROCESSING_WORKFLOW.md** - Complete preprocessing guide
- **FOLDER_STRUCTURE_MATTERS.md** - Why folder structure is correct
- **FORGE_INSIGHTS.md** - Analysis of FORGE audit data
- **ALL_DATASETS_SUMMARY.md** - Complete overview

---

## 🎓 Summary

**Dataset Renamed**: `forge_filtered` → `forge_no_abstract_not_flattened`

**Why the new name**:
- "no_abstract" = interfaces and abstract contracts removed ✅
- "not_flattened" = still has imports (needs flattening) ⚠️

**Your Best Option**: `forge_no_abstract_not_flattened`
- Already filtered (no interfaces/abstract)
- Just needs flattening (one step)
- 3,746 contracts (good size)
- Expected 85-90% success rate

**Start Here**:
```bash
# Flatten
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_no_abstract_not_flattened/train \
    --output data/datasets/forge_clean/train \
    --tool simple --batch

# Verify
./verify_contracts.sh data/datasets/forge_clean/train --max 100

# Train
./start_training.sh static --train-dir data/datasets/forge_clean/train
```

Good luck with your training! 🚀
