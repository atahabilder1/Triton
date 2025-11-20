# Quick Dataset Reference

## 📁 Your 3 Datasets Explained

### 1️⃣ FORGE-Artifacts (Raw Audit Data)
**Path**: `data/datasets/FORGE-Artifacts/`
- **Size**: 6,454 audit reports + 6,618 contracts
- **What**: Original FORGE dataset with audit JSONs
- **Best for**: Research, insights, understanding vulnerabilities
- **Status**: ❌ Not ready for training (has interfaces, missing dependencies)

**Key Insight from Analysis:**
- Only 18 unique CWEs (not 303!)
- 32% are CWE-710 (code quality, not security)
- 56% have multiple vulnerabilities (multi-label problem)
- 55% are low severity
- 63% are BSC contracts, only 11% Ethereum

### 2️⃣ combined_labeled (High-Quality Curated)
**Path**: `data/datasets/combined_labeled/`
- **Size**: 228 contracts
- **What**: Combined SmartBugs + SolidiFI + Not So Smart Contracts
- **Best for**: Quick testing, validation
- **Status**: ✅ Ready for training

**Distribution:**
- Reentrancy: 54 (23.7%)
- Safe: 60 (26.3%)
- Access Control: 29 (12.7%)
- Unchecked Calls: 30 (13.2%)
- Other classes: <20 each

**Pros**: Clean, validated, compiles well
**Cons**: Too small for deep learning

### 3️⃣ forge_balanced_accurate (Large Automatic)
**Path**: `data/datasets/forge_balanced_accurate/`
- **Size**: 7,013 contracts (70/15/15 split)
- **What**: FORGE with automatic CWE→class mapping
- **Best for**: Large-scale training (after preprocessing)
- **Status**: ⚠️ Needs preprocessing (flatten + validate)

**Distribution:**
- Well-represented: safe (1000), access_control (1000), arithmetic (1000), unchecked_calls (1000), reentrancy (800)
- Moderate: denial_of_service (500), time_manipulation (300)
- Under-represented: bad_randomness (160), short_addresses (43)

**Pros**: Large scale, balanced, real-world
**Cons**: Has interfaces, missing dependencies, automatic labels may be noisy

---

## 🎯 Which Dataset Should I Use?

### For Quick Testing
```bash
./start_training.sh static --train-dir data/datasets/combined_labeled/train
```
✅ Small, clean, validated - good for testing your pipeline

### For Serious Training
```bash
# First, preprocess
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_balanced_accurate/train \
    --output data/datasets/forge_flattened/train

python scripts/dataset/validate_contracts.py \
    data/datasets/forge_flattened/train \
    --output-dir data/datasets/forge_clean/train \
    --copy-valid

# Then verify
./verify_contracts.sh data/datasets/forge_clean/train --max 100

# Finally, train
./start_training.sh static --train-dir data/datasets/forge_clean/train
```

### For Research/Insights
```bash
# Analyze FORGE audit reports
python3 scripts/dataset/analyze_forge_audits.py \
    --forge-dir data/datasets/FORGE-Artifacts \
    --output forge_insights.json
```

---

## 📊 Quick Comparison

| Feature | FORGE-Artifacts | combined_labeled | forge_balanced_accurate |
|---------|----------------|------------------|------------------------|
| **Contracts** | 6,618 | 228 | 7,013 |
| **Labels** | CWE in JSON | Manual | Auto CWE→class |
| **Ready?** | ❌ No | ✅ Yes | ⚠️ Needs prep |
| **Quality** | Raw | High | Medium |
| **Use for** | Insights | Testing | Training |

---

## 🔧 Key Scripts

### Preprocessing
```bash
# Flatten contracts
scripts/dataset/flatten_contracts.py

# Validate contracts
scripts/dataset/validate_contracts.py

# Verify PDG/AST extraction
./verify_contracts.sh <dir>
```

### Analysis
```bash
# Analyze FORGE audits
scripts/dataset/analyze_forge_audits.py

# Show dataset summary
scripts/dataset/show_dataset_summary.py
```

### Training
```bash
# Unified training launcher
./start_training.sh <static|dynamic|semantic|full> [options]
```

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `DATASET_COMPARISON.md` | Detailed comparison of all 3 datasets |
| `FORGE_INSIGHTS.md` | Analysis results from FORGE audit data |
| `DATASET_CREATION.md` | How forge_balanced_accurate was created |
| `VERIFICATION_GUIDE.md` | How to verify PDG/AST extraction |
| `FLATTENING_GUIDE.md` | How to flatten Solidity contracts |
| `DATASET_PREPROCESSING.md` | Complete preprocessing workflow |
| `TRAINING_GUIDE.md` | How to train models |

---

## ⚠️ Key Findings from FORGE Analysis

1. **Only 18 CWEs** in actual data (not 303 from mapping)
2. **CWE-710 dominates** (32% - code quality, not security)
3. **56% are multi-label** (have multiple vulnerabilities)
4. **55% are low severity** (not exploitable)
5. **63% are BSC** (not Ethereum)

**Recommendation**: Filter FORGE by severity, exclude CWE-710, consider multi-label

---

## 🚀 Recommended Workflow

### 1. Start Small (Day 1)
```bash
# Test your pipeline on clean data
./start_training.sh static --train-dir data/datasets/combined_labeled/train
```

### 2. Understand Your Data (Day 2)
```bash
# Analyze FORGE to understand what you're working with
python3 scripts/dataset/analyze_forge_audits.py \
    --forge-dir data/datasets/FORGE-Artifacts \
    --output insights.json
```

### 3. Preprocess Large Dataset (Day 3)
```bash
# Flatten
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_balanced_accurate/train \
    --output data/datasets/forge_flattened/train

# Validate
python scripts/dataset/validate_contracts.py \
    data/datasets/forge_flattened/train \
    --output-dir data/datasets/forge_clean/train \
    --copy-valid

# Verify
./verify_contracts.sh data/datasets/forge_clean/train --max 100
```

### 4. Train at Scale (Day 4+)
```bash
# Train on cleaned FORGE
./start_training.sh static --train-dir data/datasets/forge_clean/train
```

---

## 💡 Pro Tips

1. **Always verify before training**:
   ```bash
   ./verify_contracts.sh <dataset> --max 100
   ```
   Ensure >80% success rate for PDG/AST extraction

2. **Check class balance**:
   ```bash
   python scripts/dataset/show_dataset_summary.py <dataset>
   ```

3. **Start with static encoder**:
   ```bash
   ./start_training.sh static
   ```
   Fastest to train, good baseline

4. **Use config.yaml**:
   - All paths and hyperparameters centralized
   - Easy to experiment

5. **Monitor training**:
   ```bash
   # In another terminal
   tensorboard --logdir runs/
   ```

---

## 🆘 Troubleshooting

### Low PDG/AST success rate (<80%)
```bash
# Increase timeouts in config.yaml
processing:
  slither_timeout: 120  # from 60
  solc_timeout: 60      # from 30
```

### Abstract contracts breaking training
```bash
# Use validation script
python scripts/dataset/validate_contracts.py \
    <input> --output-dir <output> --copy-valid
```

### Missing dependencies
```bash
# Flatten contracts
python scripts/dataset/flatten_contracts.py <input> --output <output>
```

### Training fails with CUDA errors
```bash
# Reduce batch size in config.yaml
training:
  static:
    batch_size: 8  # from 16
```

---

## 📞 Quick Commands Cheat Sheet

```bash
# Verify dataset
./verify_contracts.sh data/datasets/forge_balanced_accurate/train --max 100

# Flatten contracts
python scripts/dataset/flatten_contracts.py <input> --output <output>

# Validate contracts
python scripts/dataset/validate_contracts.py <input> --output-dir <output> --copy-valid

# Analyze FORGE
python3 scripts/dataset/analyze_forge_audits.py --forge-dir data/datasets/FORGE-Artifacts

# Train model
./start_training.sh static --train-dir <dataset>

# Monitor training
./scripts/monitor_training.sh
tensorboard --logdir runs/

# Check training status
./scripts/quick_status.sh
```

---

**Last Updated**: November 19, 2025
**Analysis Based On**: 6,454 FORGE audit reports
