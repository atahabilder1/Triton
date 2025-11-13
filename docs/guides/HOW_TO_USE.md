# 🚀 Triton: How to Train and Test

This guide explains the **complete training and testing process** for Triton vulnerability detection system.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding the Process](#understanding-the-process)
3. [Training](#training)
4. [Testing](#testing)
5. [Results](#results)
6. [Troubleshooting](#troubleshooting)

---

## 🎯 Quick Start

### Step 1: Train the Model
```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs-curated/dataset \
    --output-dir models/checkpoints \
    --batch-size 4 \
    --num-epochs 10 \
    --train-mode all
```

### Step 2: Test the Model
```bash
python scripts/test_dataset_performance.py \
    --dataset smartbugs \
    --output-dir results
```

That's it! You have **TWO Python scripts** for everything.

---

## 📖 Understanding the Process

### The Triton Pipeline

```
┌─────────────┐
│   TRAINING  │  → scripts/train_complete_pipeline.py
└─────────────┘
      ↓
1. Load labeled dataset (SmartBugs Curated - 143 contracts)
2. Extract features:
   - Static:   Control Flow Graphs using Slither
   - Dynamic:  Execution Traces using Mythril
   - Semantic: Code Understanding using CodeBERT
3. Train 4 components:
   Phase 1: Static Encoder    (CFG analysis)
   Phase 2: Dynamic Encoder   (Trace analysis)
   Phase 3: Semantic Encoder  (Code semantics)
   Phase 4: Fusion Module     (Combine all)
4. Save best models to models/checkpoints/

      ↓
┌─────────────┐
│   TESTING   │  → scripts/test_dataset_performance.py
└─────────────┘
      ↓
1. Load trained models from models/checkpoints/
2. Test on labeled dataset (143 contracts)
3. Calculate detection rate per vulnerability type
4. Generate reports:
   - JSON:     Detailed results
   - TXT:      Human-readable summary
   - Markdown: Tables for documentation

      ↓
┌─────────────┐
│   RESULTS   │
└─────────────┘
```

---

## 🎓 Training

### Training Script: `scripts/train_complete_pipeline.py`

This is the **ONE and ONLY** training script you need.

### Basic Usage:

```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs-curated/dataset \
    --output-dir models/checkpoints \
    --batch-size 4 \
    --num-epochs 10 \
    --train-mode all
```

### All Options:

```bash
python scripts/train_complete_pipeline.py --help
```

**Available Options:**
```
--train-dir         Path to training dataset (default: SmartBugs Curated)
--output-dir        Where to save trained models (default: models/checkpoints)
--batch-size        Batch size for training (default: 4)
--num-epochs        Number of epochs (default: 10)
--learning-rate     Learning rate (default: 0.001)
--max-samples       Limit number of samples (default: None = all)
--device            cuda or cpu (default: auto-detect)
--train-mode        What to train (default: all)
                    Options: all, static, dynamic, semantic, fusion
--skip-tests        Skip encoder tests before training
```

### Examples:

**Quick Test (3 epochs, 50 samples):**
```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs-curated/dataset \
    --num-epochs 3 \
    --max-samples 50
```

**Full Training:**
```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs-curated/dataset \
    --num-epochs 20 \
    --batch-size 4
```

**Train Only Fusion Module (after encoders are trained):**
```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs-curated/dataset \
    --train-mode fusion \
    --num-epochs 10
```

### What Training Does:

**Phase 1: Static Encoder** (~15-30 min)
- Learns from Control Flow Graphs
- Identifies suspicious code patterns
- Saves: `models/checkpoints/static_encoder_best.pt`

**Phase 2: Dynamic Encoder** (~15-30 min)
- Learns from Execution Traces
- Identifies dangerous execution paths
- Saves: `models/checkpoints/dynamic_encoder_best.pt`

**Phase 3: Semantic Encoder** (~15-30 min)
- Fine-tunes CodeBERT on vulnerability detection
- Learns code semantics
- Saves: `models/checkpoints/semantic_encoder_best.pt`

**Phase 4: Fusion Module** (~20-40 min)
- Combines all three encoders
- Learns optimal feature fusion
- Saves: `models/checkpoints/fusion_module_best.pt`

**Total Time:** 1-2 hours for 143 contracts, 10 epochs

### Training Output:

```
models/checkpoints/
├── static_encoder_best.pt         (22 MB)
├── dynamic_encoder_best.pt        (29 MB)
├── semantic_encoder_best.pt       (493 MB - CodeBERT weights)
└── fusion_module_best.pt          (38 MB)
```

### Monitoring Training Progress:

Training shows real-time progress:
```
PHASE 1: Training Static Encoder
Epoch 1/10
Training Static Encoder: 100%|████████| 29/29 [00:45<00:00, loss=2.34]
Train Loss: 2.3421, Train Acc: 25.60%
Val Loss: 2.2150, Val Acc: 31.03%
✓ Saved best static encoder (val_loss: 2.2150)
```

---

## 🧪 Testing

### Testing Script: `scripts/test_dataset_performance.py`

This is the **ONE and ONLY** testing script you need.

### Basic Usage:

```bash
python scripts/test_dataset_performance.py \
    --dataset smartbugs \
    --output-dir results
```

### All Options:

```bash
python scripts/test_dataset_performance.py --help
```

**Available Options:**
```
--dataset       Which dataset to test
                Options: smartbugs, solidifi, all, custom
                default: smartbugs
--output-dir    Where to save results (default: results)
--custom-dir    Path to custom contracts (required if --dataset custom)
```

### Examples:

**Test on SmartBugs Curated (labeled data):**
```bash
python scripts/test_dataset_performance.py --dataset smartbugs
```

**Test on All Datasets:**
```bash
python scripts/test_dataset_performance.py --dataset all
```

**Test on Custom Contracts:**
```bash
python scripts/test_dataset_performance.py \
    --dataset custom \
    --custom-dir /path/to/your/contracts \
    --output-dir results/custom
```

### What Testing Does:

1. **Load Models** (~10 seconds)
   - Loads all 4 trained models from `models/checkpoints/`

2. **Analyze Contracts** (~5-10 minutes for 143 contracts)
   - For each contract:
     - Extract static features (Slither)
     - Extract dynamic features (Mythril)
     - Extract semantic features (CodeBERT)
     - Run through fusion module
     - Predict vulnerability type
     - Compare with ground truth (if labeled)

3. **Generate Reports** (~1 second)
   - Creates 3 report files
   - Prints summary to console

**Total Time:** 5-10 minutes

### Testing Output:

```
results/
├── triton_test_results_YYYYMMDD_HHMMSS.json    (Detailed results)
├── triton_test_summary_YYYYMMDD_HHMMSS.txt     (Human-readable summary)
└── triton_results_table_YYYYMMDD_HHMMSS.md     (Markdown table)
```

### Console Output Example:

```
====================================================================================================
TRITON VULNERABILITY DETECTION - SUMMARY
====================================================================================================

SMARTBUGS Dataset - Vulnerability Detection:
----------------------------------------------------------------------------------------------------
Vulnerability Type             | Total    | Detected   | Missed   | Detection %
----------------------------------------------------------------------------------------------------
Reentrancy                     | 31       | 10         | 21       |      32.26%
Arithmetic                     | 15       | 3          | 12       |      20.00%
Access Control                 | 18       | 2          | 16       |      11.11%
Unchecked Low Level Calls      | 52       | 0          | 52       |       0.00%
...
----------------------------------------------------------------------------------------------------
TOTAL                          | 143      | 15         | 128      |      10.49%
====================================================================================================
```

---

## 📊 Results

### Viewing Results:

**Summary (Text):**
```bash
cat results/triton_test_summary_*.txt
```

**Table (Markdown):**
```bash
cat results/triton_results_table_*.md
```

**Detailed (JSON):**
```bash
cat results/triton_test_results_*.json | jq
```

### Understanding Metrics:

**Detection Rate** = (Detected / Total) × 100%
- How many vulnerabilities were found

**Precision** = True Positives / (True Positives + False Positives)
- Of detected vulnerabilities, how many are real

**Recall** = True Positives / (True Positives + False Negatives)
- Of all real vulnerabilities, how many were detected

**F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
- Balanced measure of precision and recall

### Current Performance (Nov 5, 2025):

```
Overall Detection Rate: 10.49%

Best Performing:
  Reentrancy:      32.26%
  Arithmetic:      20.00%
  Access Control:  11.11%

Not Detected (0%):
  Unchecked Low Level Calls
  Bad Randomness
  Denial of Service
  Front Running
  Time Manipulation
  Short Addresses
  Other
```

### Why Performance is Low:

1. **Small Dataset**: Only 218 labeled contracts
   - Deep learning typically needs 10,000+ samples
   - Current ratio: 22,936 parameters per sample

2. **Class Imbalance**: Largest class has 52 contracts, smallest has 1
   - Model struggles with rare classes

3. **Complex Architecture**: Multi-modal fusion is powerful but data-hungry

**Solution:** Need data augmentation or more labeled data

---

## 🔧 Troubleshooting

### Issue: Import Error

**Error:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
# Activate virtual environment
source triton_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Issue: Slither/Mythril Failures

**Error:**
```
ERROR - Slither analysis failed: Unknown error
```

**Check:**
```bash
# Verify Slither
slither --version

# Verify Mythril
myth version

# Install if missing
pip install slither-analyzer mythril
```

---

### Issue: Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Reduce batch size
python scripts/train_complete_pipeline.py \
    --batch-size 2 \
    --num-epochs 10
```

---

### Issue: No Models Found

**Error:**
```
WARNING - Checkpoint directory not found: models/checkpoints
WARNING - Using untrained models
```

**Solution:**
```bash
# Run training first
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs-curated/dataset \
    --num-epochs 10
```

---

### Issue: Dataset Not Found

**Error:**
```
ERROR - SmartBugs Curated dataset not found
```

**Solution:**
```bash
# Check dataset path
ls data/datasets/smartbugs-curated/dataset/

# If missing, verify dataset location
# Default should be: data/datasets/smartbugs-curated/dataset/
```

---

### Issue: Want Real-Time Logs

**Solution:**
```bash
# Save output to log file while watching
python -u scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs-curated/dataset \
    --num-epochs 10 \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log
```

The `-u` flag disables Python buffering for real-time output.

---

## 📁 File Structure

```
Triton/
│
├── scripts/
│   ├── train_complete_pipeline.py       # ← TRAINING SCRIPT
│   └── test_dataset_performance.py      # ← TESTING SCRIPT
│
├── models/
│   └── checkpoints/                  # ← Trained models saved here
│       ├── static_encoder_best.pt
│       ├── dynamic_encoder_best.pt
│       ├── semantic_encoder_best.pt
│       └── fusion_module_best.pt
│
├── data/
│   └── datasets/
│       └── smartbugs-curated/
│           └── dataset/              # ← Training data (143 labeled contracts)
│               ├── reentrancy/
│               ├── arithmetic/
│               ├── access_control/
│               └── ...
│
├── results/                          # ← Test results saved here
│   ├── triton_test_results_*.json
│   ├── triton_test_summary_*.txt
│   └── triton_results_table_*.md
│
├── logs/                             # ← Save logs here (optional)
│
└── HOW_TO_USE.md                     # ← This guide
```

---

## 🎯 Complete Workflow

### 1. Train the Model

```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs-curated/dataset \
    --output-dir models/checkpoints \
    --batch-size 4 \
    --num-epochs 10 \
    --train-mode all
```

**Output:**
- 4 trained model files in `models/checkpoints/`
- Time: 1-2 hours

---

### 2. Test the Model

```bash
python scripts/test_dataset_performance.py \
    --dataset smartbugs \
    --output-dir results
```

**Output:**
- 3 result files in `results/`
- Time: 5-10 minutes

---

### 3. Analyze Results

```bash
# View summary
cat results/triton_test_summary_*.txt

# View table
cat results/triton_results_table_*.md

# View detailed JSON
cat results/triton_test_results_*.json | jq '.overall_metrics'
```

---

### 4. Iterate (Improve Performance)

**Add More Data:**
```bash
# Use a larger dataset or combine multiple datasets
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/combined/ \
    --num-epochs 20
```

**Tune Hyperparameters:**
```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs-curated/dataset \
    --batch-size 8 \
    --num-epochs 20 \
    --learning-rate 0.0001
```

**Train Longer:**
```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs-curated/dataset \
    --num-epochs 50
```

---

## 💡 Tips for Better Results

### 1. Expand Training Data

Currently using **143 contracts**. Target: **500-1000+**

**Options:**
- Add SmartBugs Samples (+50 contracts)
- Add Not So Smart Contracts (+25 contracts)
- Manually label more contracts
- Use data augmentation (5-10× multiplier)

---

### 2. Handle Class Imbalance

The script already uses **class weights** automatically.

You can verify this in the training output:
```
Class weights calculated:
  reentrancy: count=31, weight=0.516
  arithmetic: count=15, weight=1.067
  access_control: count=18, weight=0.889
  ...
```

---

### 3. Fix Tool Errors

**Problem:** Slither/Mythril failures reduce training quality

**Check logs for:**
```bash
grep "ERROR" logs/*.log
grep "Slither analysis failed" logs/*.log
```

**Common causes:**
- Incompatible Solidity version
- Missing dependencies
- Timeout issues

---

### 4. Monitor Training

**Save logs:**
```bash
mkdir -p logs
python -u scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs-curated/dataset \
    --num-epochs 10 \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log
```

**Watch in real-time:**
```bash
# In another terminal
tail -f logs/training_*.log
```

---

### 5. Test on Different Datasets

**SmartBugs Curated (labeled):**
```bash
python scripts/test_dataset_performance.py --dataset smartbugs
```

**SolidiFI (buggy/patched pairs):**
```bash
python scripts/test_dataset_performance.py --dataset solidifi
```

**Your own contracts:**
```bash
python scripts/test_dataset_performance.py \
    --dataset custom \
    --custom-dir /path/to/contracts
```

---

## ❓ FAQ

**Q: How long does training take?**
A: 1-2 hours for 143 contracts, 10 epochs on CPU. Faster on GPU.

**Q: Can I stop and resume training?**
A: No, training must complete. Use fewer epochs for testing.

**Q: Why is detection rate only 10%?**
A: Small dataset (143 contracts) + class imbalance. Need more data!

**Q: Can I train on my own dataset?**
A: Yes! Organize your contracts by vulnerability type like SmartBugs Curated structure.

**Q: Do I need a GPU?**
A: No, but recommended. CPU works but is slower.

**Q: How do I improve performance?**
A: 1) Add more labeled data, 2) Train longer, 3) Fix Slither errors, 4) Tune hyperparameters

**Q: Where are logs saved?**
A: By default, printed to console. Use `tee` to save to file (see examples above).

**Q: Can I train only specific components?**
A: Yes! Use `--train-mode` option: `static`, `dynamic`, `semantic`, `fusion`, or `all`

---

## 📞 Getting Help

If you encounter issues:

1. **Check this guide's troubleshooting section**
2. **Read error messages carefully**
3. **Check logs** (if you saved them)
4. **Verify dataset structure** matches SmartBugs Curated format
5. **Check Python dependencies** are installed

---

## 🚀 Quick Reference

### Training
```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs-curated/dataset \
    --num-epochs 10
```

### Testing
```bash
python scripts/test_dataset_performance.py --dataset smartbugs
```

### View Results
```bash
cat results/triton_test_summary_*.txt
```

---

**Last Updated:** November 5, 2025
