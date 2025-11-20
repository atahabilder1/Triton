# Complete Dataset Summary

## 📊 All Dataset Locations in Your Project

You have **8 different dataset locations** with **3 main training-ready datasets**:

---

## 🎯 Main Training Datasets (3)

### 1. `combined_labeled` ⭐ Best for Testing
**Location**: `data/datasets/combined_labeled/`
- **Total Contracts**: 228
- **Solidity Files**: 456 (train/val/test copies)
- **Size**: 3.6 MB
- **Created By**: `scripts/dataset/combine_labeled_datasets.py`
- **Sources**: SmartBugs Curated + SmartBugs Samples + SolidiFI + Not So Smart Contracts
- **Quality**: ✅ **Highest** - manually curated and validated

**Structure**:
```
combined_labeled/
├── train/          (70% - ~160 contracts)
├── val/            (15% - ~34 contracts)
├── test/           (15% - ~34 contracts)
└── By class directories:
    ├── access_control/         29 contracts
    ├── reentrancy/             54 contracts
    ├── safe/                   60 contracts
    ├── arithmetic/             17 contracts
    ├── unchecked_low_level_calls/ 30 contracts
    ├── bad_randomness/         10 contracts
    ├── denial_of_service/       9 contracts
    ├── time_manipulation/      11 contracts
    ├── front_running/           6 contracts
    └── short_addresses/         2 contracts
```

**Pros**:
- ✅ Highest quality - manually validated
- ✅ Clean compilation - no dependencies issues
- ✅ Well-documented vulnerabilities
- ✅ Ready to use immediately

**Cons**:
- ❌ Very small - only 228 contracts
- ❌ Imbalanced - some classes have <10 samples
- ❌ Limited diversity - academic examples

**Best For**: Quick testing, pipeline validation, baseline models

---

### 2. `forge_balanced_accurate` ⭐ Best for Training
**Location**: `data/datasets/forge_balanced_accurate/`
- **Total Contracts**: 7,013
- **Solidity Files**: 6,575
- **Size**: 66 MB
- **Created By**: `scripts/dataset/prepare_forge_dataset_accurate.py`
- **Source**: FORGE-Artifacts with CWE → vulnerability class mapping
- **Quality**: ⚠️ **Medium** - automatic labeling

**Structure**:
```
forge_balanced_accurate/
├── train/          (70% - 4,909 contracts)
│   ├── safe/                      700 contracts
│   ├── access_control/            700 contracts
│   ├── arithmetic/                700 contracts
│   ├── unchecked_low_level_calls/ 700 contracts
│   ├── reentrancy/                560 contracts
│   ├── other/                     700 contracts
│   ├── denial_of_service/         350 contracts
│   ├── time_manipulation/         210 contracts
│   ├── front_running/             147 contracts
│   ├── bad_randomness/            112 contracts
│   └── short_addresses/            30 contracts
├── val/            (15% - 1,051 contracts)
└── test/           (15% - 1,053 contracts)
```

**Class Distribution**:
| Class | Train | Val | Test | Total | Status |
|-------|-------|-----|------|-------|--------|
| safe | 700 | 150 | 150 | 1,000 | ✅ Well-balanced |
| access_control | 700 | 150 | 150 | 1,000 | ✅ Well-balanced |
| arithmetic | 700 | 150 | 150 | 1,000 | ✅ Well-balanced |
| unchecked_calls | 700 | 150 | 150 | 1,000 | ✅ Well-balanced |
| reentrancy | 560 | 120 | 120 | 800 | ✅ Well-balanced |
| other | 700 | 150 | 150 | 1,000 | ✅ Well-balanced |
| denial_of_service | 350 | 75 | 75 | 500 | ⚠️ Moderate |
| time_manipulation | 210 | 45 | 45 | 300 | ⚠️ Moderate |
| front_running | 147 | 31 | 32 | 210 | ⚠️ Under-represented |
| bad_randomness | 112 | 24 | 24 | 160 | ❌ Under-represented |
| short_addresses | 30 | 6 | 7 | 43 | ❌ Very rare |

**Pros**:
- ✅ Large scale - 7,013 contracts
- ✅ Balanced - most classes 500-1,000 samples
- ✅ Real-world - from actual audits
- ✅ Train/val/test pre-split

**Cons**:
- ❌ Has interfaces and abstract contracts
- ❌ Missing dependencies (imports not resolved)
- ❌ Automatic labeling (CWE mapping may be noisy)
- ❌ Contains low-severity issues (code quality)
- ⚠️ **Needs preprocessing**: flatten + validate

**Best For**: Large-scale training after preprocessing

---

### 3. `forge_filtered` ⭐ Filtered Version
**Location**: `data/datasets/forge_filtered/`
- **Total Contracts**: ~3,746 (estimated from train split)
- **Solidity Files**: 3,746
- **Size**: 33 MB
- **Created By**: `scripts/dataset/filter_dataset.py` (likely)
- **Source**: Filtered from forge_balanced_accurate
- **Quality**: ⚠️ **Medium-High** - filtered subset

**Structure**:
```
forge_filtered/
├── train/          (~2,600 contracts)
│   ├── access_control/            322 contracts
│   ├── arithmetic/                396 contracts
│   ├── unchecked_low_level_calls/ 360 contracts
│   ├── other/                     359 contracts
│   ├── safe/                      333 contracts
│   ├── reentrancy/                327 contracts
│   ├── denial_of_service/         188 contracts
│   ├── time_manipulation/         129 contracts
│   ├── bad_randomness/             84 contracts
│   ├── front_running/              81 contracts
│   └── short_addresses/            17 contracts
├── val/
└── test/
```

**Observations**:
- 📉 ~53% smaller than forge_balanced_accurate (3,746 vs 7,013)
- 📊 Still relatively balanced across main classes
- 🤔 Likely filtered to remove problematic contracts

**Pros**:
- ✅ Medium size - 3,746 contracts
- ✅ Cleaner than forge_balanced_accurate (filtered)
- ✅ More balanced than combined_labeled
- ✅ Still has good class representation

**Cons**:
- ❓ No documentation on what was filtered
- ❓ No summary JSON to explain filtering criteria
- ⚠️ May still need preprocessing

**Best For**: Compromise between quality and scale

---

## 📂 Source Datasets (5)

These are the original datasets used to create the training sets above:

### 4. `FORGE-Artifacts` (Original FORGE)
**Location**: `data/datasets/FORGE-Artifacts/`
- **Contracts**: 78,223 .sol files in `dataset/contracts/`
- **Audit Reports**: 6,454 JSON files in `dataset/results/`
- **Size**: ~1.5 GB
- **Type**: Raw audit data with CWE codes

**What's Inside**:
- Original smart contract files (with dependencies)
- Audit report JSONs with:
  - CWE codes
  - Severity levels
  - Vulnerability descriptions
  - Contract metadata (compiler version, blockchain, address)

**Best For**: Research, insights, creating custom datasets

---

### 5. `smartbugs-curated`
**Location**: `data/datasets/smartbugs-curated/`
- **Contracts**: 143 .sol files
- **Size**: 1.5 MB
- **Type**: Curated vulnerable contracts
- **Classes**: 9 vulnerability types

**Used In**: `combined_labeled` dataset

---

### 6. `smartbugs`
**Location**: `data/datasets/smartbugs/`
- **Contracts**: 50 .sol files
- **Size**: 85 MB
- **Type**: SmartBugs sample contracts

**Used In**: `combined_labeled` dataset

---

### 7. `solidifi`
**Location**: `data/datasets/solidifi/`
- **Contracts**: 50 .sol files
- **Size**: 9.7 MB
- **Type**: Safe contracts (no vulnerabilities)

**Used In**: `combined_labeled` dataset (as "safe" class)

---

### 8. `audits`
**Location**: `data/datasets/audits/`
- **Contracts**: 25 .sol files
- **Size**: 1.6 MB
- **Type**: Not So Smart Contracts

**Used In**: `combined_labeled` dataset

---

### 9. `securify`
**Location**: `data/datasets/securify/`
- **Contracts**: 381 .sol files
- **Size**: 6.4 MB
- **Type**: Securify benchmark dataset
- **Status**: ❓ Not currently used in training datasets

---

## 📈 Dataset Comparison

| Dataset | Contracts | Size | Quality | Balance | Ready? | Best For |
|---------|-----------|------|---------|---------|--------|----------|
| **combined_labeled** | 228 | 3.6 MB | ⭐⭐⭐⭐⭐ | ❌ Poor | ✅ Yes | Testing |
| **forge_balanced_accurate** | 7,013 | 66 MB | ⭐⭐⭐ | ✅ Good | ⚠️ Needs prep | Training |
| **forge_filtered** | 3,746 | 33 MB | ⭐⭐⭐⭐ | ✅ Good | ⚠️ Maybe | Compromise |
| FORGE-Artifacts | 78,223 | 1.5 GB | Raw | N/A | ❌ No | Research |
| smartbugs-curated | 143 | 1.5 MB | ⭐⭐⭐⭐⭐ | ❌ Poor | ✅ Yes | - |
| smartbugs | 50 | 85 MB | ⭐⭐⭐⭐ | ❌ Poor | ✅ Yes | - |
| solidifi | 50 | 9.7 MB | ⭐⭐⭐⭐⭐ | N/A | ✅ Yes | - |
| audits | 25 | 1.6 MB | ⭐⭐⭐⭐ | ❌ Poor | ✅ Yes | - |
| securify | 381 | 6.4 MB | ⭐⭐⭐ | ❓ | ⚠️ | Unused |

---

## 🎯 Which Dataset Should You Use?

### Scenario 1: Quick Testing
```bash
# Use combined_labeled (228 contracts, high quality)
./start_training.sh static --train-dir data/datasets/combined_labeled/train
```
**Why**: Clean, validated, ready to use

### Scenario 2: Serious Training
```bash
# Option A: Use forge_filtered (medium size, filtered)
./verify_contracts.sh data/datasets/forge_filtered/train --max 100
./start_training.sh static --train-dir data/datasets/forge_filtered/train
```
**Why**: Good compromise - cleaner than forge_balanced_accurate

```bash
# Option B: Use forge_balanced_accurate (large, needs preprocessing)
# 1. Flatten
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_balanced_accurate/train \
    --output data/datasets/forge_flattened/train

# 2. Validate
python scripts/dataset/validate_contracts.py \
    data/datasets/forge_flattened/train \
    --output-dir data/datasets/forge_clean/train \
    --copy-valid

# 3. Verify
./verify_contracts.sh data/datasets/forge_clean/train --max 100

# 4. Train
./start_training.sh static --train-dir data/datasets/forge_clean/train
```
**Why**: Maximum scale, but requires work

### Scenario 3: Research/Analysis
```bash
# Use FORGE-Artifacts
python3 scripts/dataset/analyze_forge_audits.py \
    --forge-dir data/datasets/FORGE-Artifacts \
    --output insights.json
```
**Why**: Rich metadata, real-world insights

---

## 🔍 Dataset Hierarchy

```
FORGE-Artifacts (78,223 contracts - Raw)
    ↓ [prepare_forge_dataset_accurate.py]
    ↓ [CWE → Vulnerability mapping]
    ↓ [Balance classes]
    ↓
forge_balanced_accurate (7,013 contracts)
    ↓ [filter_dataset.py?]
    ↓ [Remove problematic contracts]
    ↓
forge_filtered (3,746 contracts)


SmartBugs + SolidiFI + Not So Smart + Audits
    ↓ [combine_labeled_datasets.py]
    ↓ [Manual curation]
    ↓
combined_labeled (228 contracts)
```

---

## 💡 Recommendations

### 1. Start with `combined_labeled`
**Why**: Test your pipeline on clean, validated data first
```bash
./start_training.sh static --train-dir data/datasets/combined_labeled/train
```

### 2. Verify `forge_filtered`
**Why**: It's already filtered - might be ready to use!
```bash
./verify_contracts.sh data/datasets/forge_filtered/train --max 100
```
If verification shows >80% success, use it directly!

### 3. Investigate `forge_filtered`
**Why**: Understand what filtering was done
```bash
# Create summary
python3 scripts/dataset/show_dataset_summary.py data/datasets/forge_filtered

# Compare to forge_balanced_accurate
# What contracts were removed? Why?
```

### 4. Consider `securify`
**Why**: 381 contracts unused - might be valuable
```bash
# Analyze securify dataset
ls -R data/datasets/securify/
# Integrate if it has labels
```

---

## 🚀 Recommended Workflow

### Day 1: Quick Start
```bash
# Test on combined_labeled
./start_training.sh static --train-dir data/datasets/combined_labeled/train
```
**Goal**: Verify your training pipeline works

### Day 2: Scale Up (Option A - Easy)
```bash
# Verify forge_filtered
./verify_contracts.sh data/datasets/forge_filtered/train --max 100

# If >80% success, train directly
./start_training.sh static --train-dir data/datasets/forge_filtered/train
```
**Goal**: Train on medium-sized, filtered dataset

### Day 2: Scale Up (Option B - More Work)
```bash
# Preprocess forge_balanced_accurate
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_balanced_accurate/train \
    --output data/datasets/forge_flattened/train

python scripts/dataset/validate_contracts.py \
    data/datasets/forge_flattened/train \
    --output-dir data/datasets/forge_clean/train \
    --copy-valid

./verify_contracts.sh data/datasets/forge_clean/train --max 100
./start_training.sh static --train-dir data/datasets/forge_clean/train
```
**Goal**: Maximum training data

### Day 3: Research
```bash
# Understand FORGE insights
python3 scripts/dataset/analyze_forge_audits.py \
    --forge-dir data/datasets/FORGE-Artifacts

# Create custom filtered dataset based on insights
```

---

## 📝 Summary

**Total Dataset Locations**: 8
- **3 Training-ready**: combined_labeled, forge_balanced_accurate, forge_filtered
- **5 Source datasets**: FORGE-Artifacts, smartbugs-curated, smartbugs, solidifi, audits, securify

**Total Unique Contracts**:
- **Combined_labeled**: 228 (curated, high quality)
- **Forge_filtered**: 3,746 (filtered, medium quality)
- **Forge_balanced_accurate**: 7,013 (balanced, needs preprocessing)
- **FORGE-Artifacts**: 78,223 (raw, for research)

**Recommended Path**:
1. Test: `combined_labeled` (228)
2. Train: `forge_filtered` (3,746) ← **Check this first!**
3. Scale: `forge_balanced_accurate` (7,013) after preprocessing

**Key Finding**: You have `forge_filtered` which is 53% the size of `forge_balanced_accurate` - this suggests it's already been preprocessed/filtered. **Check this dataset first** before doing more work!
