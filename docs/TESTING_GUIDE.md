# Triton Testing Guide

**How to test all models and generate comprehensive reports**

---

## Quick Start

### Run Complete Test Suite (All 4 Modalities):

```bash
./run_full_test.sh
```

This will test:
1. **Static Encoder Only** - PDG-based graph analysis
2. **Dynamic Encoder Only** - Execution trace analysis
3. **Semantic Encoder Only** - Code semantics (CodeBERT)
4. **Fusion Model** - All 3 modalities combined

**Output:** `COMPREHENSIVE_TEST_REPORT.md` with full comparison

---

## Available Test Scripts

### 1. Comprehensive Test (Recommended)

```bash
python3 test_comprehensive_report.py --test-dir data/datasets/combined_labeled/test
```

**Features:**
- Tests all 4 modalities automatically
- Generates detailed Markdown report
- Per-class metrics (Precision, Recall, F1, TP/FP/FN)
- Side-by-side comparison table
- Best model identification per vulnerability type

**Output:**
- Console: Real-time progress and summary
- File: `COMPREHENSIVE_TEST_REPORT.md`

---

### 2. Individual Modality Test

```bash
python3 test_each_modality.py --test-dir data/datasets/combined_labeled/test
```

**Features:**
- Tests Static, Dynamic, Semantic separately
- Console output only (no file)
- Quick comparison table
- Useful for debugging individual encoders

**Options:**
```bash
--skip-static    # Skip static encoder (if Slither broken)
--skip-dynamic   # Skip dynamic encoder (if Mythril broken)
--skip-fusion    # Skip fusion model
```

---

### 3. Detailed Metrics Test

```bash
python3 test_models_detailed.py --test-dir data/datasets/combined_labeled/test
```

**Features:**
- Extremely detailed metrics
- TP, FP, FN, TN counts
- Confusion matrix analysis
- Detection counts per class

---

### 4. Safe Contract Detection Test

```bash
python3 test_with_safe_detection.py \
    --test-dir data/datasets/combined_labeled/test \
    --threshold 0.55
```

**Features:**
- Tests safe contract detection specifically
- Uses confidence threshold
- Shows precision/recall for safe class

---

## Test Dataset

**Location:** `data/datasets/combined_labeled/test/`

**Structure:**
```
test/
├── access_control/        (5 contracts)
├── arithmetic/            (4 contracts)
├── bad_randomness/        (2 contracts)
├── denial_of_service/     (2 contracts)
├── front_running/         (2 contracts)
├── reentrancy/            (7 contracts)
├── safe/                  (10 contracts)
├── short_addresses/       (1 contract)
├── time_manipulation/     (2 contracts)
└── unchecked_low_level_calls/ (9 contracts)
```

**Total:** 44 contracts (held out from training)

---

## Understanding the Report

### Overall Metrics Table:

| Metric | Description |
|--------|-------------|
| **Success Rate** | % of contracts successfully processed |
| **Accuracy** | % of correct predictions (TP+TN)/(TP+TN+FP+FN) |
| **Avg F1** | Average F1 score across all vulnerability types |
| **Avg Precision** | Average precision across all types |
| **Avg Recall** | Average recall across all types |
| **Correct/Total** | Number of correct predictions |

### Per-Class Metrics:

| Metric | Description | Formula |
|--------|-------------|---------|
| **Precision** | How many predicted vulnerabilities were correct | TP/(TP+FP) |
| **Recall** | How many actual vulnerabilities were found | TP/(TP+FN) |
| **F1** | Harmonic mean of precision and recall | 2×P×R/(P+R) |
| **TP** | True Positives - Correctly identified vulnerabilities | - |
| **FP** | False Positives - Safe contracts flagged as vulnerable | - |
| **FN** | False Negatives - Missed vulnerabilities | - |
| **Support** | Total contracts in this class in test set | TP+FN |

---

## Example Workflow

### After Training:

```bash
# 1. Run comprehensive test
./run_full_test.sh

# 2. View results
cat COMPREHENSIVE_TEST_REPORT.md

# 3. Analyze per-class performance
grep -A 15 "Per-Class Metrics" COMPREHENSIVE_TEST_REPORT.md
```

### Testing Individual Models:

```bash
# Test only semantic encoder (fastest)
python3 test_each_modality.py \
    --skip-static \
    --skip-dynamic \
    --skip-fusion

# Test only static encoder
python3 test_each_modality.py \
    --skip-dynamic \
    --skip-fusion
```

### Custom Test Directory:

```bash
# Test on different dataset
python3 test_comprehensive_report.py \
    --test-dir /path/to/your/contracts \
    --output custom_report.md
```

---

## Expected Performance

Based on Nov 5-6 testing with combined_labeled dataset:

| Model | Success Rate | Accuracy | Best At |
|-------|-------------|----------|---------|
| **Static Only** | 95.5% | 11.90% | access_control, structural patterns |
| **Dynamic Only** | 100% | 20.45% | unchecked_low_level_calls |
| **Semantic Only** | 100% | 50.00% | Most vulnerability types |
| **Fusion** | TBD | 55-65%* | Combining strengths of all 3 |

*Expected based on training, needs testing

---

## Troubleshooting

### Static Encoder Fails:

```bash
# Skip static if Slither has issues
./run_full_test.sh --skip-static

# Or test individually
python3 test_each_modality.py --skip-static
```

### Dynamic Encoder Slow:

```bash
# Mythril can be slow, reduce timeout
# Edit test script: mythril = MythrilWrapper(timeout=10)  # Reduce from 30
```

### CUDA Out of Memory:

```bash
# Reduce batch size in semantic encoder testing
# Edit test script: batch_size = 2  # Reduce from 4
```

### Missing Models:

Check that these files exist:
```
models/checkpoints/static_encoder_best.pt
models/checkpoints/dynamic_encoder_best.pt
models/checkpoints/semantic_encoder_best.pt
models/checkpoints/fusion_module_best.pt
models/checkpoints/static_encoder_fusion_best.pt
models/checkpoints/dynamic_encoder_fusion_best.pt
models/checkpoints/semantic_encoder_fusion_best.pt
```

---

## Advanced Options

### Skip Slow Encoders:

```bash
# Test only fast models (semantic)
python3 test_comprehensive_report.py \
    --skip-static \
    --skip-dynamic \
    --skip-fusion
```

### Custom Output File:

```bash
# Save to custom location
python3 test_comprehensive_report.py \
    --output reports/test_$(date +%Y%m%d).md
```

### Parallel Testing:

```bash
# Test each modality separately in parallel
python3 test_each_modality.py --skip-dynamic --skip-fusion > static_results.txt &
python3 test_each_modality.py --skip-static --skip-fusion > dynamic_results.txt &
wait
```

---

## Report Format

The comprehensive report includes:

1. **Overall Performance Comparison**
   - All 4 models side-by-side
   - Success rates, accuracy, F1 scores

2. **Per-Class F1 Comparison**
   - Each vulnerability type
   - Best model identified for each

3. **Detailed Per-Model Results**
   - Success rate breakdown
   - Per-class precision/recall/F1
   - TP/FP/FN/Support counts

4. **Timestamp & Metadata**
   - Test date/time
   - Number of contracts tested
   - Dataset location

---

## Interpreting Results

### Good Performance Indicators:

✅ **High Accuracy** (>50%) - Model makes correct predictions
✅ **High Precision** (>0.7) - Few false positives
✅ **High Recall** (>0.7) - Few false negatives
✅ **High F1** (>0.6) - Good balance of P&R
✅ **High Success Rate** (>95%) - Can process most contracts

### Areas for Improvement:

⚠️ **Low Accuracy** (<30%) - Needs more training
⚠️ **Low Precision** - Too many false alarms
⚠️ **Low Recall** - Missing vulnerabilities
⚠️ **Low F1 for rare classes** - Need more training data
⚠️ **Low Success Rate** - Tool extraction issues (Slither/Mythril)

---

## Next Steps After Testing

### If accuracy is low (<40%):

1. **Increase training epochs** (20 → 50)
2. **Add more training data** (228 → 500+ contracts)
3. **Tune hyperparameters** (learning rate, batch size)
4. **Implement focal loss** for class imbalance

### If specific classes perform poorly:

1. **Add more examples** of that vulnerability type
2. **Use data augmentation**
3. **Adjust class weights** in loss function
4. **Try ensemble methods**

### If fusion underperforms individual models:

1. **Retrain fusion module** with more epochs
2. **Adjust fusion architecture** (hidden dims)
3. **Try different fusion strategies** (attention, gating)

---

## Summary

**Quick Test:** `./run_full_test.sh` → `COMPREHENSIVE_TEST_REPORT.md`

**Options:** Skip encoders with `--skip-static`, `--skip-dynamic`, `--skip-fusion`

**Output:** Markdown report with accuracy, F1, precision, recall, TP/FP/FN

**Expected Time:** 10-15 minutes for 44 contracts

**Purpose:** Compare Static/Dynamic/Semantic/Fusion performance side-by-side
