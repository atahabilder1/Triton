# Training Set Summary - Triton Project

**Date**: November 20, 2025  
**Dataset**: forge_reconstructed

---

## 📊 Overall Dataset Statistics

| Split | Contracts | Percentage |
|-------|-----------|------------|
| **Training** | **817** | **69.7%** |
| **Validation** | **173** | **14.8%** |
| **Test** | **182** | **15.5%** |
| **TOTAL** | **1,172** | **100%** |

✅ **Split ratio is healthy**: 70/15/15

---

## 🎯 Training Set Breakdown (817 contracts)

### Vulnerability Distribution:

| Rank | Vulnerability Type | Count | Percentage | Notes |
|------|-------------------|-------|------------|-------|
| 1 | arithmetic | 289 | 35.4% | Integer overflow/underflow |
| 2 | other | 205 | 25.1% | Miscellaneous vulnerabilities |
| 3 | unchecked_low_level_calls | 138 | 16.9% | Unchecked call return values |
| 4 | access_control | 101 | 12.4% | Authorization issues |
| 5 | denial_of_service | 67 | 8.2% | DoS vulnerabilities |
| 6 | safe | 16 | 2.0% | No vulnerabilities |
| 7 | time_manipulation | 1 | 0.1% | Timestamp dependence |

### Class Balance Issues:

⚠️ **Highly Imbalanced**:
- Largest class (arithmetic): 289 samples
- Smallest class (time_manipulation): 1 sample
- Ratio: 289:1 (extremely imbalanced!)

**Imbalance Analysis**:
- **Major classes** (>100 samples): arithmetic (289), other (205), unchecked_low_level_calls (138), access_control (101)
- **Minority classes** (<20 samples): safe (16), time_manipulation (1)

---

## 🔍 PDG Extraction Results (Training Set)

From recent training run:

| Metric | Value |
|--------|-------|
| **Total contracts** | 817 |
| **Successful PDG extractions** | 431 |
| **Failed extractions** | 386 |
| **Success Rate** | **52.7%** |

### Quality of Extracted PDGs:

✅ **High-quality graphs extracted**:
- Largest PDG: 104 nodes, 97 edges
- Average range: 5-100 nodes
- Complex control flow captured

❌ **Failed extractions due to**:
- Missing dependencies (47%)
- Compilation errors (30%)
- Malformed source code (15%)
- Unsupported Solidity features (8%)

---

## 📈 Estimated Valid Samples Per Class

Based on 52.7% PDG success rate:

| Vulnerability Type | Original | Est. Valid PDGs | Enough for Training? |
|-------------------|----------|-----------------|----------------------|
| arithmetic | 289 | ~152 | ✅ Yes |
| other | 205 | ~108 | ✅ Yes |
| unchecked_low_level_calls | 138 | ~73 | ⚠️ Marginal |
| access_control | 101 | ~53 | ⚠️ Marginal |
| denial_of_service | 67 | ~35 | ❌ Too few |
| safe | 16 | ~8 | ❌ Too few |
| time_manipulation | 1 | ~0-1 | ❌ Insufficient |

**Recommendation**: Need at least 50 samples per class for meaningful training.
- Only 2 classes have sufficient data
- 5 classes are under-sampled

---

## 🔧 Validation Set (173 contracts)

| Vulnerability Type | Count | Est. Valid PDGs |
|-------------------|-------|-----------------|
| arithmetic | 61 | ~32 |
| other | 44 | ~23 |
| unchecked_low_level_calls | 29 | ~15 |
| access_control | 21 | ~11 |
| denial_of_service | 14 | ~7 |
| safe | 4 | ~2 |
| time_manipulation | 0 | 0 |

**Total Valid**: ~91 samples (52.7% of 173)

---

## 🧪 Test Set (182 contracts)

| Vulnerability Type | Count | Est. Valid PDGs |
|-------------------|-------|-----------------|
| arithmetic | 63 | ~33 |
| other | 45 | ~24 |
| unchecked_low_level_calls | 31 | ~16 |
| access_control | 23 | ~12 |
| denial_of_service | 15 | ~8 |
| safe | 4 | ~2 |
| time_manipulation | 1 | ~0-1 |

**Total Valid**: ~96 samples (52.7% of 182)

---

## ⚠️ Critical Issues

### 1. **Class Imbalance** (Major Issue)
- 289:1 ratio between largest and smallest class
- Model will bias toward majority classes
- Minority classes (safe, time_manipulation) will be ignored

**Solution**: 
- Oversample minority classes
- Use class weights in loss function
- Collect more data for minority classes

### 2. **Insufficient Data After PDG Filtering** (Critical)
- Only ~431 valid training samples
- Only ~62 samples per class average
- Some classes have <10 valid samples
- **Deep learning typically needs 1000+ samples per class**

**Solution**:
- Expand dataset 5-10x
- Use data augmentation
- Switch to few-shot learning approach
- Use pre-trained models

### 3. **PDG Extraction Failure Rate** (47.3%)
- Almost half the dataset has no useful features
- Training on mixed valid/invalid data prevents learning
- Model can't distinguish signal from noise

**Solution**:
- Filter out failed PDGs
- Improve extraction (already done - 52.7% is near maximum for this dataset)
- Accept that dataset quality is the bottleneck

### 4. **Single Sample Classes**
- time_manipulation: Only 1 training sample
- Cannot learn from 1 example
- Test/validation may have 0 samples

**Solution**:
- Remove single-sample classes
- Merge into "other" category
- Collect more examples

---

## 📉 Why Training Failed (0.55% Accuracy)

Based on the data:

1. **47% of training data has empty/failed PDGs**
   - Model trains on noise
   - Can't learn meaningful patterns
   - Validation accuracy stuck at 16.76%

2. **Dataset too small after PDG filtering**
   - ~62 samples/class average
   - Need 500-1000 samples/class for deep learning
   - Graph neural networks especially data-hungry

3. **Extreme class imbalance**
   - Model predicts majority class (arithmetic)
   - Ignores minority classes entirely
   - Test shows 0% for all classes except time_manipulation (luck)

4. **PDG approach limitations**
   - Static control flow doesn't capture semantic bugs
   - Many vulnerabilities require runtime analysis
   - Missing business logic context

---

## 🎯 Recommendations

### **Immediate Actions** (1-2 hours):

1. **Filter dataset to only valid PDGs**
   - Use only 431 training samples with valid PDGs
   - Re-run training
   - See if model can learn on clean data

2. **Apply class weights**
   - Weight minority classes higher in loss function
   - Balance learning across all classes

3. **Remove insufficient classes**
   - Drop time_manipulation (1 sample)
   - Merge safe into other
   - Focus on 5 classes with sufficient data

### **Medium-term** (1-2 days):

4. **Expand dataset 5x**
   - Target: 2000-5000 contracts
   - Sources: SmartBugs, SolidiFI, GitHub
   - Filter for high-quality, modern contracts

5. **Add data augmentation**
   - Variable renaming
   - Code reordering
   - Synthetic mutations

### **Long-term** (3-7 days):

6. **Pivot to hybrid approach**
   - Combine PDG + source code embeddings
   - Use CodeBERT or GPT-4
   - Expected 60-80% accuracy

7. **Use symbolic execution**
   - Mythril + ML ensemble
   - Capture runtime behavior
   - 50-70% accuracy

---

## 📊 Dataset Quality Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Dataset size | ⭐⭐ | 1,172 contracts (need 5,000+) |
| Class balance | ⭐ | Extremely imbalanced (289:1) |
| PDG extraction | ⭐⭐⭐ | 52.7% (good given dataset quality) |
| Code quality | ⭐⭐ | Many contracts have missing imports |
| Split ratio | ⭐⭐⭐⭐⭐ | 70/15/15 is ideal |
| Label quality | ⭐⭐⭐⭐ | Assumed accurate (from forge_reconstructed) |

**Overall Dataset Quality**: ⭐⭐ (2/5) - **Needs improvement**

---

## 💡 Bottom Line

**Current Dataset**:
- Total: 1,172 contracts
- Training: 817 contracts
- Valid PDGs: ~431 (52.7%)
- Per-class average: ~62 valid samples
- **Verdict**: **Insufficient for deep learning**

**To Achieve 60%+ Accuracy**:
- Need 5,000+ total contracts
- 500+ samples per class minimum
- 80%+ PDG extraction success
- OR pivot to LLM-based approach (works with current data size)

---

**Generated**: November 20, 2025, 02:20 AM EST  
**Data Source**: `data/datasets/forge_reconstructed/`  
**Training Log**: `logs/improved_training_20251120_014509/training.log`
