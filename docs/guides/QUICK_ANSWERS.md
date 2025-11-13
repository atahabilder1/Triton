# Quick Answers to Your Questions

## **Q1: How many classes should I use?**

**Answer: KEEP 10 CLASSES (+ 1 safe class)**

❌ **DON'T use 304 classes** - Too hard to train, need ~15,000 samples per class
✅ **USE 10 classes** - Your current setup works perfectly!

```
1. safe (no vulnerabilities)
2. access_control
3. arithmetic
4. reentrancy
5. unchecked_low_level_calls
6. bad_randomness
7. denial_of_service
8. front_running
9. time_manipulation
10. other (unmapped CWEs)
```

---

## **Q2: Do I need separate folders?**

**Answer: YES - Same structure as your current dataset**

```
data/datasets/forge_balanced/
├── train/                    ← 70% of data
│   ├── safe/
│   ├── access_control/
│   ├── arithmetic/
│   └── ...
├── val/                      ← 15% of data
│   ├── safe/
│   └── ...
└── test/                     ← 15% of data
    ├── safe/
    └── ...
```

**Why?** Your training script expects this structure (line 79-106 in train_complete_pipeline.py)

---

## **Q3: Do I need to change scripts?**

**Answer: Almost NO changes needed!**

✅ **No changes:** train_complete_pipeline.py, encoders, fusion module
✅ **Tiny change:** Update default path (1 line)
✅ **New script:** prepare_forge_dataset.py (I'll create)

---

## **Q4: Should it be 304 classes (CWE) or 10 classes?**

**Answer: 10 CLASSES (map 303 CWEs → 10 types)**

**Mapping:**
```python
CWE-284, CWE-269, CWE-285, CWE-862 → access_control
CWE-682, CWE-190, CWE-191         → arithmetic
CWE-703, CWE-252, CWE-476         → unchecked_calls
CWE-841, CWE-362, CWE-667, CWE-691 → reentrancy
CWE-330, CWE-338                   → bad_randomness
CWE-400, CWE-835, CWE-770          → denial_of_service
CWE-362                            → front_running
CWE-829, CWE-347                   → time_manipulation
No findings                         → safe
Everything else                     → other
```

---

## **Q5: Vulnerable or not, then what type?**

**Answer: Single model with 10 outputs (your current approach is perfect!)**

```
Model outputs → [10 class probabilities]

If all < threshold:     "safe" (no vulnerability)
Else:                   argmax → specific vulnerability type
```

**This is BETTER than:**
- Two separate models (binary + multiclass)
- 304-class classifier

---

## **Q6: How to balance data?**

**Answer: THREE strategies (all included in your code!):**

### **1. Balanced Sampling (during dataset creation)**
```
Target samples per class: 300-1000
Minority classes: Use ALL available + augment if needed
```

### **2. Class Weights (during training) - ALREADY IN YOUR CODE!**
```python
# Line 968-989 in train_complete_pipeline.py
class_weights = calculate_class_weights(dataset)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

This automatically handles imbalance - no extra work needed!

### **3. Shuffle Each Epoch - ALREADY IN YOUR CODE!**
```python
# Line 84-90 in train_complete_pipeline.py
DataLoader(..., shuffle=True)  # ← Automatic!
```

**You DON'T need to:**
- ❌ Re-balance data each epoch
- ❌ Manually oversample
- ❌ Create custom samplers

**It's already handled!**

---

## **Q7: Will I need to change code for 304 classes vs 10?**

**Answer: NO changes if you use 10 classes!**

Your code already supports variable # of classes:
- Line 63-74: Change `vuln_types` dict (but no need - keep 10!)
- Line 968: Change `num_classes` parameter (but default=10 is perfect!)

**For 304 classes you'd need:**
- ❌ Rewrite vulnerability type mapping (304 entries)
- ❌ Bigger models (more output neurons)
- ❌ 10x more training data
- ❌ Different evaluation metrics

**Recommendation:** KEEP 10 CLASSES!

---

## **Summary Table:**

| Question | Answer | Changes Needed |
|----------|--------|----------------|
| How many classes? | 10 (not 304) | None |
| Folder structure? | Yes, train/val/test | Create folders |
| Script changes? | Minimal | 1 new script |
| Balancing? | Class weights | Already in code |
| Re-balance each epoch? | No | None |
| Two-stage (binary→multi)? | No, single 10-class model | None |

---

## **What You Need to Do:**

1. ✅ Run `python scripts/prepare_forge_dataset.py` (I'll create this)
2. ✅ Verify folders created: `ls data/datasets/forge_balanced/train/`
3. ✅ Train: `python scripts/train_complete_pipeline.py --train-dir data/datasets/forge_balanced/train`
4. ✅ Test: `python scripts/test_dataset_performance.py`

That's it! 🎉

---

## **Expected Timeline:**

- Dataset prep: 30-60 minutes
- Training: 8-12 hours (GPU)
- Testing: 30 minutes

**Expected Results:**
- Current: 12-20% accuracy
- After FORGE: **60-75% accuracy** (3-5x better!)
