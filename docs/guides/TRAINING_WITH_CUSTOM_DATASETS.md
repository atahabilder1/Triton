# Training with Custom Datasets - Complete Guide

## ✅ Quick Start

Your training script now supports **ANY dataset** with separate train/val/test folders!

### **Use FORGE Dataset (Recommended)**

```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --test-dir data/datasets/forge_balanced_accurate/test \
    --num-epochs 20 \
    --batch-size 8
```

### **Use Your Old Dataset (combined_labeled)**

```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/combined_labeled/train \
    --val-dir data/datasets/combined_labeled/val \
    --test-dir data/datasets/combined_labeled/test \
    --num-epochs 20 \
    --batch-size 8
```

### **Use SmartBugs Dataset**

```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs/train \
    --val-dir data/datasets/smartbugs/val \
    --test-dir data/datasets/smartbugs/test \
    --num-epochs 20 \
    --batch-size 8
```

### **Use ANY Custom Dataset**

```bash
python scripts/train_complete_pipeline.py \
    --train-dir /path/to/your/dataset/train \
    --val-dir /path/to/your/dataset/val \
    --test-dir /path/to/your/dataset/test \
    --num-epochs 20 \
    --batch-size 8
```

---

## 📁 Dataset Structure Requirements

Your dataset must follow this folder structure:

```
your_dataset/
├── train/
│   ├── safe/              ← One folder per class
│   │   ├── contract1.sol
│   │   └── contract2.sol
│   ├── access_control/
│   │   ├── contract3.sol
│   │   └── contract4.sol
│   ├── arithmetic/
│   ├── reentrancy/
│   └── ... (other classes)
│
├── val/                   ← Same structure as train
│   ├── safe/
│   ├── access_control/
│   └── ...
│
└── test/                  ← Same structure as train
    ├── safe/
    ├── access_control/
    └── ...
```

### **Supported Classes**

The script automatically recognizes these 10 vulnerability classes:

1. `safe` - No vulnerabilities
2. `access_control` - Authorization issues
3. `arithmetic` - Integer overflow/underflow
4. `bad_randomness` - Weak randomness
5. `denial_of_service` - DoS vulnerabilities
6. `front_running` - Transaction ordering issues
7. `reentrancy` - Reentrancy attacks
8. `short_addresses` - Short address attacks
9. `time_manipulation` - Timestamp dependence
10. `unchecked_low_level_calls` - Unchecked calls
11. `other` - Other vulnerabilities

**Note**: Your dataset doesn't need ALL classes - just create folders for the classes you have!

---

## 🎯 Complete Training Flow

### **What Happens When You Run Training?**

```
1. LOAD DATASETS
   ├── Load training data from --train-dir
   ├── Load validation data from --val-dir
   └── (Optional) Load test data from --test-dir

2. TRAIN STATIC ENCODER (Phase 1)
   ├── Train for 20 epochs
   ├── Validate after EACH epoch
   └── Save best model (lowest val loss)

3. TRAIN DYNAMIC ENCODER (Phase 2)
   ├── Train for 20 epochs
   ├── Validate after EACH epoch
   └── Save best model

4. TRAIN SEMANTIC ENCODER (Phase 3)
   ├── Train for 20 epochs
   ├── Validate after EACH epoch
   └── Save best model

5. TRAIN FUSION MODULE (Phase 4)
   ├── Train for 20 epochs (all encoders + fusion)
   ├── Validate after EACH epoch
   └── Save best fusion model

6. FINAL TEST EVALUATION (if --test-dir provided)
   ├── Evaluate Static Encoder on test set
   ├── Evaluate Dynamic Encoder on test set
   ├── Evaluate Semantic Encoder on test set
   └── Evaluate Fusion Module on test set
```

---

## 📊 Example Output

```bash
$ python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --test-dir data/datasets/forge_balanced_accurate/test \
    --num-epochs 20 \
    --batch-size 8
```

### **Output:**

```
================================================================================
COMPLETE TRITON TRAINING PIPELINE
================================================================================
Training directory: data/datasets/forge_balanced_accurate/train
Output directory: models/checkpoints
Batch size: 8
Epochs: 20
Learning rate: 0.001
Training mode: all

Loading dataset...
Loading training data from: data/datasets/forge_balanced_accurate/train
Found 606 safe contracts
Found 629 access_control contracts
Found 663 arithmetic contracts
...
Loaded 4540 contracts total

Loading validation data from: data/datasets/forge_balanced_accurate/val
Found 134 safe contracts
Found 134 access_control contracts
...
Loaded 1011 contracts total

Training samples: 4540
Validation samples: 1011

================================================================================
PHASE 1: Training Static Encoder
================================================================================

Epoch 1/20
Training Static Encoder: 100%|████████████| 568/568 [05:23<00:00]
Train Loss: 2.5678, Train Acc: 25.34%
Val Loss: 2.6123, Val Acc: 23.45%
✓ Saved best static encoder (val_loss: 2.6123)

Epoch 2/20
Training Static Encoder: 100%|████████████| 568/568 [05:21<00:00]
Train Loss: 2.3456, Train Acc: 28.91%
Val Loss: 2.4123, Val Acc: 27.12%
✓ Saved best static encoder (val_loss: 2.4123)

...

Static Encoder training complete! Best val_loss: 2.1234

================================================================================
PHASE 2: Training Dynamic Encoder
================================================================================
... (similar output)

================================================================================
PHASE 3: Training Semantic Encoder
================================================================================
... (similar output)

================================================================================
PHASE 4: Training Fusion Module End-to-End
================================================================================
... (similar output)

================================================================================
TRAINING COMPLETE!
================================================================================
Model checkpoints saved to: models/checkpoints

================================================================================
FINAL TEST SET EVALUATION
================================================================================
Loading test data from: data/datasets/forge_balanced_accurate/test
Test samples: 1024

Evaluating trained models on test set...
Static Encoder  - Test Loss: 2.2345, Test Acc: 32.45%
Dynamic Encoder - Test Loss: 2.1234, Test Acc: 38.67%
Semantic Encoder - Test Loss: 1.8901, Test Acc: 62.34%
Fusion Module   - Test Loss: 1.7654, Test Acc: 65.89%

================================================================================
FINAL TEST RESULTS SUMMARY
================================================================================
Static:  32.45%
Dynamic: 38.67%
Semantic: 62.34%
Fusion:  65.89%
```

---

## 🎛️ Command Line Options

### **Required Parameters**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--train-dir` | Training data directory | `data/datasets/forge_balanced_accurate/train` |

### **Optional Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--val-dir` | None | Validation directory. If not provided, splits train 80/20 |
| `--test-dir` | None | Test directory. If not provided, skips final test evaluation |
| `--output-dir` | `models/checkpoints` | Where to save trained models |
| `--batch-size` | 4 | Batch size for training |
| `--num-epochs` | 10 | Number of training epochs per component |
| `--learning-rate` | 0.001 | Learning rate for optimizer |
| `--max-samples` | 10000 | Max training samples (None = use all) |
| `--device` | auto | Device to use (cuda/cpu) |
| `--skip-tests` | False | Skip encoder functionality tests |
| `--train-mode` | all | Which components to train (all/static/dynamic/semantic/fusion) |

---

## 🔍 Advanced Usage

### **Train Only Specific Components**

```bash
# Train only static encoder
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --train-mode static \
    --num-epochs 30

# Train only fusion (requires pre-trained encoders)
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --train-mode fusion \
    --num-epochs 30
```

### **Use Limited Training Data (for quick testing)**

```bash
# Train on first 100 samples only
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --max-samples 100 \
    --num-epochs 5
```

### **Train Without Validation Directory (Auto-Split)**

```bash
# Script will automatically split train_dir 80/20
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --num-epochs 20
```

### **Save to Custom Output Directory**

```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --output-dir models/forge_experiment_1 \
    --num-epochs 20
```

### **Adjust for GPU Memory**

```bash
# If you get OOM (Out of Memory) errors
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --batch-size 2 \
    --num-epochs 20
```

---

## 📈 Comparing Different Datasets

### **Experiment 1: Train on FORGE**

```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --test-dir data/datasets/forge_balanced_accurate/test \
    --output-dir models/forge_experiment \
    --num-epochs 20 \
    --batch-size 8
```

### **Experiment 2: Train on Combined Labeled**

```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/combined_labeled/train \
    --val-dir data/datasets/combined_labeled/val \
    --test-dir data/datasets/combined_labeled/test \
    --output-dir models/combined_experiment \
    --num-epochs 20 \
    --batch-size 8
```

### **Experiment 3: Train on SmartBugs**

```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs/train \
    --val-dir data/datasets/smartbugs/val \
    --test-dir data/datasets/smartbugs/test \
    --output-dir models/smartbugs_experiment \
    --num-epochs 20 \
    --batch-size 8
```

### **Compare Results:**

| Dataset | Train Size | Static | Dynamic | Semantic | Fusion |
|---------|-----------|--------|---------|----------|--------|
| FORGE | 4,540 | 32% | 38% | 62% | 66% |
| Combined | 155 | 12% | 20% | 50% | 0% |
| SmartBugs | ~200 | 15% | 25% | 55% | 10% |

---

## ⚠️ Troubleshooting

### **Problem: "No such directory"**

```
Error: [Errno 2] No such file or directory: 'data/datasets/forge_balanced_accurate/train'
```

**Solution:** Make sure the path is correct and directories exist

```bash
# Check if directory exists
ls data/datasets/forge_balanced_accurate/train

# If not, check available datasets
ls data/datasets/
```

### **Problem: "Out of Memory" (OOM)**

```
RuntimeError: CUDA out of memory.
```

**Solution:** Reduce batch size

```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --batch-size 2  # ← Reduced from 8
```

### **Problem: Training is very slow**

**Solution 1:** Check if GPU is being used

```bash
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

**Solution 2:** Reduce number of samples for testing

```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --max-samples 500 \
    --num-epochs 5
```

### **Problem: Validation accuracy not improving**

**Possible causes:**
- ✅ Learning rate too high → Reduce to 0.0001
- ✅ Too few epochs → Increase to 30
- ✅ Too little data → Use larger dataset (FORGE recommended)

---

## ✅ Summary

### **Key Points**

1. **Required parameter:** `--train-dir` (path to training data)
2. **Recommended:** Use `--val-dir` and `--test-dir` for proper evaluation
3. **Validation happens:** After EVERY epoch automatically
4. **Test evaluation:** Only runs if `--test-dir` is provided
5. **Works with ANY dataset:** Just follow the folder structure

### **Recommended Command (FORGE Dataset)**

```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --test-dir data/datasets/forge_balanced_accurate/test \
    --num-epochs 20 \
    --batch-size 8
```

**Training Time:** 8-12 hours with GPU

**Expected Results:**
- Static: 30-40% accuracy
- Dynamic: 35-45% accuracy
- Semantic: 60-70% accuracy
- Fusion: 55-70% accuracy

**Good luck! 🚀**
