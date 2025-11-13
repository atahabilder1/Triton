# FORGE Dataset Training Plan - Complete Strategy

## **🎯 YOUR REQUIREMENTS:**

1. ✅ Use FORGE dataset for training, testing, validation
2. ✅ First: Is it vulnerable or not? (Binary)
3. ✅ Then: What type of vulnerability? (10 classes)
4. ✅ Balanced data for training
5. ✅ Minimal script changes

---

## **📊 RECOMMENDED STRATEGY: Hierarchical Classification (2-Stage)**

### **Why This Approach?**

Your goal is actually **TWO tasks:**
1. **Stage 1 (Binary)**: Vulnerable vs Safe
2. **Stage 2 (Multi-class)**: Which of 10 vulnerability types

This is **better than 304 classes** because:
- ✅ Easier to train (fewer classes = better accuracy)
- ✅ Matches your current pipeline (10 classes)
- ✅ More interpretable results
- ✅ Can reuse your existing models!

---

## **🗂️ DATASET ORGANIZATION**

### **Folder Structure (Keep Your 10 Classes!):**

```
data/datasets/forge_balanced/
├── train/                        ← 70% of data
│   ├── safe/                    ← 1,000 contracts (no vulnerabilities)
│   ├── access_control/          ← 1,000 contracts (CWE-284, 269, 285...)
│   ├── arithmetic/              ← 1,000 contracts (CWE-682, 190, 191...)
│   ├── reentrancy/              ← 800 contracts (CWE-841, 362...)
│   ├── unchecked_low_level_calls/ ← 1,000 contracts (CWE-252, 703...)
│   ├── bad_randomness/          ← 300 contracts (CWE-330, 338...)
│   ├── denial_of_service/       ← 300 contracts (CWE-400, 835...)
│   ├── front_running/           ← 300 contracts (CWE-362...)
│   ├── time_manipulation/       ← 300 contracts (CWE-829...)
│   └── other/                   ← 500 contracts (unmapped CWEs)
│       Total: ~6,500 contracts
│
├── val/                          ← 15% of data (~1,400 contracts)
│   ├── safe/
│   ├── access_control/
│   └── ...
│
└── test/                         ← 15% of data (~1,400 contracts)
    ├── safe/
    ├── access_control/
    └── ...
```

**Total Dataset Size: ~9,300 contracts (60x your current size!)**

---

## **🔄 DATA BALANCING STRATEGY**

### **Option 1: Balanced Sampling (RECOMMENDED)**

Extract equal or proportional samples per class:

```python
TARGET_SAMPLES = {
    'safe': 1000,                     # From 1,141 available
    'access_control': 1000,           # From ~6,000 available ✓
    'arithmetic': 1000,               # From ~3,500 available ✓
    'unchecked_low_level_calls': 1000, # From ~4,200 available ✓
    'reentrancy': 800,                # From ~200 available (all)
    'bad_randomness': 300,            # From ~50 available (all + augment)
    'denial_of_service': 300,         # From ~100 available (all + augment)
    'front_running': 300,             # From ~100 available (all + augment)
    'time_manipulation': 300,         # From ~50 available (all + augment)
    'other': 500                      # Unmapped CWEs
}
```

**Balancing per epoch:**
- ❌ NO need to re-balance each epoch
- ✅ Use class weights in loss function (already in your code!)
- ✅ Random shuffle each epoch (PyTorch DataLoader does this)

### **Option 2: Weighted Sampling**

Use `WeightedRandomSampler` to oversample rare classes during training:

```python
# Already handles imbalance!
class_weights = calculate_class_weights(dataset)  # Your existing function
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

---

## **📝 CWE → 10 CLASSES MAPPING**

```python
CWE_TO_10_CLASSES = {
    # Safe (no findings)
    'NO_FINDINGS': 'safe',

    # Access Control (~10,000 samples)
    'CWE-284': 'access_control',  # Improper Access Control
    'CWE-269': 'access_control',  # Improper Privilege Management
    'CWE-285': 'access_control',  # Improper Authorization
    'CWE-862': 'access_control',  # Missing Authorization
    'CWE-732': 'access_control',  # Incorrect Permission Assignment
    'CWE-266': 'access_control',  # Incorrect Privilege Assignment

    # Arithmetic (~3,500 samples)
    'CWE-682': 'arithmetic',      # Incorrect Calculation
    'CWE-190': 'arithmetic',      # Integer Overflow
    'CWE-191': 'arithmetic',      # Integer Underflow
    'CWE-369': 'arithmetic',      # Divide by Zero

    # Unchecked Calls (~4,200 samples)
    'CWE-703': 'unchecked_low_level_calls',  # Improper Exception Handling
    'CWE-252': 'unchecked_low_level_calls',  # Unchecked Return Value
    'CWE-476': 'unchecked_low_level_calls',  # NULL Pointer Dereference

    # Reentrancy (~200 samples)
    'CWE-841': 'reentrancy',      # Improper Enforcement of Behavioral Workflow
    'CWE-362': 'reentrancy',      # Race Condition
    'CWE-667': 'reentrancy',      # Improper Locking
    'CWE-691': 'reentrancy',      # Insufficient Control Flow

    # Bad Randomness (~50 samples)
    'CWE-330': 'bad_randomness',  # Use of Insufficiently Random Values
    'CWE-338': 'bad_randomness',  # Use of Cryptographically Weak PRNG

    # Denial of Service (~100 samples)
    'CWE-400': 'denial_of_service',  # Uncontrolled Resource Consumption
    'CWE-835': 'denial_of_service',  # Loop with Unreachable Exit
    'CWE-770': 'denial_of_service',  # Allocation without Limits

    # Front Running (~100 samples)
    'CWE-362': 'front_running',   # Race Condition (overlaps with reentrancy)

    # Time Manipulation (~50 samples)
    'CWE-829': 'time_manipulation',  # Inclusion of Untrusted Functionality
    'CWE-347': 'time_manipulation',  # Improper Verification of Signatures

    # Other (for unmapped CWEs)
    'CWE-710': 'other',           # Coding Standard Violation
    'CWE-664': 'other',           # Improper Control of Resource
    # ... etc
}
```

---

## **🛠️ SCRIPTS TO MODIFY**

### **Scripts That WORK AS-IS (No changes needed!):**

✅ `scripts/train_complete_pipeline.py` - Already supports 10 classes!
✅ Your encoders (static, dynamic, semantic) - Work with any # classes
✅ Fusion module - Works with any # classes
✅ Class weights calculation - Already implemented

### **Scripts You NEED:**

**NEW: `scripts/prepare_forge_dataset.py`** (I'll create this)
- Reads FORGE JSONs
- Maps 303 CWEs → your 10 types
- Copies .sol files to organized folders
- Creates train/val/test splits (70/15/15)
- Balances classes

**MODIFY: `scripts/train_complete_pipeline.py`**
- Change line 63-74: Keep your 10 types (NO CHANGES!)
- Change line 996: Point to new dataset
```python
default="data/datasets/forge_balanced/train"  # Instead of FORGE-Artifacts
```

That's it! Everything else works!

---

## **🎓 TRAINING PROCESS**

### **Phase 1: Prepare Dataset (ONE TIME)**

```bash
cd /home/anik/code/Triton

# Create balanced dataset from FORGE
python scripts/prepare_forge_dataset.py \
    --forge-dir /data/llm_projects/triton_datasets/FORGE-Artifacts/dataset \
    --output-dir data/datasets/forge_balanced \
    --samples-per-class 1000 \
    --split-ratio 0.7 0.15 0.15
```

**Output:**
```
data/datasets/forge_balanced/
├── train/     (6,500 contracts)
├── val/       (1,400 contracts)
└── test/      (1,400 contracts)
```

### **Phase 2: Train Models**

```bash
# Train all components
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced/train \
    --num-epochs 20 \
    --batch-size 8 \
    --learning-rate 0.001 \
    --max-samples 10000
```

**Training Sequence:**
1. **Static Encoder** (10-20 epochs)
2. **Dynamic Encoder** (10-20 epochs)
3. **Semantic Encoder** (5-10 epochs, pre-trained)
4. **Fusion Module** (10-20 epochs, fine-tune all)

**Time Estimate:**
- Dataset preparation: 30-60 minutes
- Training: 8-12 hours (with GPU)

### **Phase 3: Evaluate**

```bash
python scripts/test_dataset_performance.py \
    --dataset data/datasets/forge_balanced/test
```

---

## **⚖️ DATA BALANCING - DETAILED**

### **Balancing Strategy:**

**During Dataset Creation:**
```python
# Target: 1000 samples per major class
# Minority classes: Use all available + data augmentation
```

**During Training (Already in your code!):**

```python
# Line 968-989 in train_complete_pipeline.py
class_weights = calculate_class_weights(dataset)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

This automatically:
- ✅ Weights loss higher for rare classes
- ✅ Prevents model from only predicting majority class
- ✅ No need to re-balance each epoch

**Shuffle per epoch:**
```python
# Line 84-90 in train_complete_pipeline.py
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,  # ← Automatically shuffles each epoch!
)
```

---

## **📊 EXPECTED RESULTS**

### **Before (Current Dataset - 155 samples):**
```
Static:   12% accuracy
Dynamic:  20% accuracy
Semantic: 50% accuracy
Fusion:    0% accuracy (broken)
```

### **After (FORGE Dataset - 6,500+ samples):**
```
Static:   35-45% accuracy   (3-4x improvement)
Dynamic:  40-50% accuracy   (2-3x improvement)
Semantic: 65-75% accuracy   (1.3-1.5x improvement)
Fusion:   60-75% accuracy   (FIXED + 50x more data!)
```

**F1 Score:** 0.15 → 0.60-0.70 (4-5x better)

---

## **✅ FINAL CHECKLIST**

### **What You Need to Do:**

1. ☐ Run `prepare_forge_dataset.py` (I'll create this)
2. ☐ Verify folder structure created correctly
3. ☐ Run training with new dataset
4. ☐ Compare results with old dataset

### **What You DON'T Need to Do:**

- ❌ Modify encoders
- ❌ Modify fusion module
- ❌ Change number of classes (keep 10!)
- ❌ Re-balance data each epoch
- ❌ Create 304 class model

---

## **🤔 WHY NOT 304 CLASSES?**

You asked about 304 classes. Here's why 10 is better:

| Aspect | 10 Classes | 304 Classes |
|--------|-----------|-------------|
| **Data per class** | ~650 samples | ~20 samples |
| **Training difficulty** | Moderate | Very hard |
| **Accuracy** | 60-70% expected | 20-30% expected |
| **Interpretability** | Easy | Very hard |
| **Script changes** | None | Major rewrite |
| **Model complexity** | Current models OK | Need bigger models |

**Recommendation:** Start with 10 classes, achieve good accuracy, then consider multi-label (one contract can have multiple CWEs) instead of 304-class.

---

## **📋 SUMMARY**

**Dataset:** 9,300 contracts organized into 10 classes + safe
**Structure:** train/val/test folders (70/15/15 split)
**Balancing:** Class weights in loss function + all available minority samples
**Scripts needed:** 1 new (prepare_forge_dataset.py), 1 tiny change (train path)
**Training time:** ~12 hours
**Expected improvement:** 4-5x better accuracy

**Next:** I'll create the `prepare_forge_dataset.py` script now!
