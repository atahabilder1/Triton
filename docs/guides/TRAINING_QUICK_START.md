# Quick Start: Static Vulnerability Detection Training

## 🚀 Start Training (GPU Optimized)

```bash
./start_training_gpu.sh
```

This uses **full GPU power** with optimized settings!

---

## 📊 Real-Time Training Monitoring

### What You'll See During Training

```
================================================================================
EPOCH 1/50 - TRAINING
================================================================================

  Batch [  10/568] | Loss: 1.2345 | Acc: 45.67% | Speed: 1.23 batch/s | ETA: 7m 32s
  Batch [  20/568] | Loss: 1.1234 | Acc: 48.91% | Speed: 1.25 batch/s | ETA: 7m 15s
  Batch [  30/568] | Loss: 1.0987 | Acc: 51.23% | Speed: 1.27 batch/s | ETA: 7m 02s
  ...

================================================================================
EPOCH 1/50 SUMMARY
================================================================================
⏱️  Time: 8m 45s
📉 Train Loss: 1.0234 | Train Acc: 52.34%
📊 Val Loss:   0.9876 | Val Acc:   55.67% | Val F1: 0.5234
✅ NEW BEST MODEL SAVED!
================================================================================
```

### Key Indicators of Good Training

✅ **Loss is decreasing**:
   - Epoch 1: Loss 1.23 → Epoch 5: Loss 0.85 → Epoch 10: Loss 0.67

✅ **Accuracy is increasing**:
   - Epoch 1: 45% → Epoch 5: 58% → Epoch 10: 65%

✅ **Validation F1 improving**:
   - Epoch 1: F1 0.42 → Epoch 5: F1 0.56 → Epoch 10: F1 0.64

❌ **Warning Signs (should cancel training)**:
   - Loss stays constant or increases for 5+ epochs
   - Accuracy stuck at low values (< 30%)
   - See message: "⚠️ No improvement for 5/5 epochs"

---

## 📈 Per-Vulnerability Detection (Every 5 Epochs)

```
================================================================================
EPOCH 5 VALIDATION - DETAILED METRICS
================================================================================

🎯 OVERALL ACCURACY: 58.34%

Vulnerability Type              Precision     Recall   F1-Score  Support  Detected
------------------------------------------------------------------------------------
✅ reentrancy                      0.7234     0.6891     0.7058       119    82/119  (68.9%)
✅ arithmetic                      0.6912     0.7397     0.7146       146   108/146  (74.0%)
✅ unchecked_low_level_calls       0.7260     0.7260     0.7260       146   106/146  (72.6%)
⚠️  access_control                 0.5234     0.5891     0.5546       148    87/148  (58.9%)
⚠️  denial_of_service              0.4986     0.5081     0.5033        74    38/74   (51.4%)
❌ short_addresses                 0.2286     0.2857     0.2545         7     2/7    (28.6%)
------------------------------------------------------------------------------------
📊 MACRO F1                                                        0.5890
📊 WEIGHTED F1                                                     0.6234
📊 TOTAL DETECTED                                                  623/1024  (60.8%)
================================================================================
```

### Understanding the Symbols

- **✅** (Green): Recall ≥ 70% → Model detecting well!
- **⚠️** (Yellow): Recall 50-70% → Moderate detection
- **❌** (Red): Recall < 50% → Poor detection

---

## 🛑 When to Stop Training

### Auto-Stop (Early Stopping)

Training automatically stops if:
- No improvement for **5 consecutive epochs**
- Message: `🛑 EARLY STOPPING: No improvement for 5 epochs`

### Manual Stop (Ctrl+C)

Stop training if you see:

1. **Loss not decreasing**:
   ```
   Epoch 10: Loss 0.89
   Epoch 11: Loss 0.91  ← Getting worse!
   Epoch 12: Loss 0.92
   ```

2. **Accuracy stuck**:
   ```
   Epoch 8:  Acc 45.2%
   Epoch 9:  Acc 45.8%
   Epoch 10: Acc 45.3%  ← No progress
   Epoch 11: Acc 45.9%
   ```

3. **Warning messages**:
   ```
   ⚠️ No improvement for 3/5 epochs
   ⚠️ No improvement for 4/5 epochs  ← Close to auto-stop
   ```

---

## 💪 GPU Utilization Check

### Check GPU Usage During Training

```bash
# In another terminal:
watch -n 1 nvidia-smi
```

Look for:
- **GPU Memory**: Should use 8-12 GB (out of 46 GB on A6000)
- **GPU Utilization**: Should be 80-100%
- **Power Usage**: Should be near max (e.g., 250W+ on A6000)

### If GPU not fully utilized:

Increase batch size:
```bash
python train_static_optimized.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --test-dir data/datasets/forge_balanced_accurate/test \
    --batch-size 24  # Increase from 16 to 24
```

---

## 📁 Dataset Summary

```
FORGE Balanced Dataset
├── Train:      4,540 contracts (69%)
├── Validation: 1,011 contracts (15%)
└── Test:       1,024 contracts (16%)
Total:          6,575 contracts

Per Vulnerability Type:
- arithmetic:                663 train | 148 val | 146 test
- unchecked_low_level_calls: 666 train | 147 val | 146 test
- access_control:            629 train | 137 val | 148 test
- safe:                      606 train | 143 val | 140 test
- reentrancy:                553 train | 117 val | 119 test
- other:                     620 train | 141 val | 143 test
- denial_of_service:         317 train |  73 val |  74 test
- time_manipulation:         206 train |  45 val |  45 test
- front_running:             138 train |  30 val |  32 test
- bad_randomness:            112 train |  24 val |  24 test
- short_addresses:            30 train |   6 val |   7 test (⚠️ very imbalanced!)
```

---

## ⏱️ Expected Training Time

**GPU (A6000, 46GB)**:
- Epoch 1 (with PDG extraction): ~25-30 minutes
- Epochs 2-50 (cached PDGs): ~8-10 minutes each
- **Total**: 2-3 hours for 20 epochs

**CPU Only**:
- Epoch 1: ~2-3 hours
- Epochs 2-50: ~45-60 minutes each
- **Total**: 15-20 hours for 20 epochs

---

## 🎯 Expected Final Results

Based on FORGE dataset, you should see:

**Overall Accuracy**: 60-72%

**Best Detected Vulnerabilities** (Recall > 70%):
- ✅ Reentrancy: 75-82%
- ✅ Arithmetic: 70-77%
- ✅ Unchecked Calls: 70-76%
- ✅ Safe Contracts: 68-75%

**Moderate Detection** (Recall 50-70%):
- ⚠️ Access Control: 60-70%
- ⚠️ Denial of Service: 55-65%
- ⚠️ Time Manipulation: 55-65%

**Challenging** (Recall < 50%):
- ❌ Short Addresses: 30-45% (very few training samples)
- ❌ Bad Randomness: 45-55% (semantic issue, not structural)
- ❌ Front Running: 50-60% (transaction ordering, not visible in PDG)

---

## 📊 View Training Progress (TensorBoard)

```bash
tensorboard --logdir runs/
```

Open browser: http://localhost:6006

You'll see:
- **Loss curves**: Should decrease over time
- **Accuracy curves**: Should increase over time
- **F1 score**: Should improve
- **Learning rate**: Auto-adjusted if model plateaus

---

## 🔧 Troubleshooting

### Out of Memory Error

Reduce batch size:
```bash
--batch-size 8  # Instead of 16
```

### Training Too Slow

Increase batch size (if GPU has memory):
```bash
--batch-size 24  # Instead of 16
```

Increase workers (if CPU has cores):
```bash
--num-workers 16  # Instead of 8
```

### Poor Performance

Try different learning rate:
```bash
--learning-rate 0.0001  # Lower (more stable)
# or
--learning-rate 0.01    # Higher (faster but less stable)
```

---

## 📂 Output Files

After training:

1. **Model checkpoint**: `models/checkpoints/static_encoder_best.pt`
   - Best model based on validation F1
   - Can be loaded for inference

2. **Test results**: `models/checkpoints/test_results_YYYYMMDD_HHMMSS.txt`
   - Detailed per-vulnerability metrics
   - Precision, recall, F1 for each type

3. **Training log**: `logs/training_gpu_YYYYMMDD_HHMMSS.log`
   - Full console output
   - All batch/epoch stats

4. **TensorBoard logs**: `runs/static_optimized_YYYYMMDD_HHMMSS/`
   - Training curves
   - Validation metrics

---

## 🚀 Quick Commands

**Start training**:
```bash
./start_training_gpu.sh
```

**Monitor GPU**:
```bash
watch -n 1 nvidia-smi
```

**View TensorBoard**:
```bash
tensorboard --logdir runs/
```

**Stop training gracefully**:
```
Press Ctrl+C once (will finish current batch)
```

**Force stop**:
```
Press Ctrl+C twice
```

---

## ❓ Is Training Working?

### Good Signs ✅

- Loss decreasing: 1.2 → 0.9 → 0.7 → 0.5
- Accuracy increasing: 45% → 55% → 62% → 68%
- F1 score improving: 0.42 → 0.55 → 0.63 → 0.68
- Model saved every few epochs: "✅ NEW BEST MODEL SAVED!"
- GPU utilization: 80-100%

### Bad Signs ❌

- Loss stuck or increasing: 0.9 → 0.91 → 0.92
- Accuracy not improving: 45% → 46% → 45%
- Multiple warnings: "⚠️ No improvement for X epochs"
- GPU idle: < 20% utilization

If you see bad signs for 5+ epochs: **Stop and debug!**

---

## 🎓 Understanding Loss Function

**What is it?**
- Measures how "wrong" the model's predictions are
- Lower loss = better predictions

**How it works:**
```
Model predicts: "reentrancy" (confidence: 0.85)
Ground truth:   "reentrancy"
→ Loss: 0.05 (LOW - correct prediction)

Model predicts: "safe" (confidence: 0.90)
Ground truth:   "reentrancy"
→ Loss: 2.30 (HIGH - wrong prediction)
```

**Training goal**: Minimize loss by updating model weights

---

## 📝 Notes

- First epoch is slow (extracting PDGs from contracts)
- Later epochs are fast (PDGs cached in `data/cache/`)
- Early stopping prevents overfitting
- Class weights handle imbalanced dataset
- Model auto-saves best version based on validation F1
