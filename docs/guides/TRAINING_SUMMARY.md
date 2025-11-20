# Static Vulnerability Detection - Complete Summary

## 📁 Files Created

### Training Scripts
1. **`train_static_optimized.py`** - Optimized training script
   - Uses full GPU power
   - Real-time progress monitoring
   - Early stopping if not improving
   - Detailed per-vulnerability metrics

2. **`start_training_gpu.sh`** - One-command launcher
   - Checks GPU availability
   - Uses optimal settings for A6000
   - Logs everything to file

### Documentation
1. **`TRAINING_QUICK_START.md`** - Quick reference guide
2. **`STATIC_TRAINING_GUIDE.md`** - Detailed explanation

---

## 🚀 How to Start Training

```bash
./start_training_gpu.sh
```

That's it! This will:
- ✅ Use full GPU power (16 batch size, 8 workers)
- ✅ Train for up to 50 epochs (auto-stops if not improving)
- ✅ Show real-time progress every 10 batches
- ✅ Print detailed metrics every 5 epochs
- ✅ Save best model automatically

---

## 📊 What You'll See (Real-Time)

### Every 10 Batches:
```
Batch [  10/568] | Loss: 1.2345 | Acc: 45.67% | Speed: 1.23 batch/s | ETA: 7m 32s
```

**How to know if training is working**:
- ✅ Loss decreasing (1.23 → 1.15 → 1.08 → ...)
- ✅ Accuracy increasing (45% → 48% → 51% → ...)
- ✅ Speed consistent (1.2-1.5 batch/s)

### Every Epoch:
```
================================================================================
EPOCH 5/50 SUMMARY
================================================================================
⏱️  Time: 8m 45s
📉 Train Loss: 1.0234 | Train Acc: 52.34%
📊 Val Loss:   0.9876 | Val Acc:   55.67% | Val F1: 0.5234
✅ NEW BEST MODEL SAVED!
================================================================================
```

### Every 5 Epochs (Detailed Metrics):
```
✅ reentrancy                      0.7234     0.6891     0.7058    94/119  (79.0%)
✅ arithmetic                      0.6912     0.7397     0.7146   108/146  (74.0%)
⚠️  access_control                 0.5234     0.5891     0.5546    87/148  (58.9%)
❌ short_addresses                 0.2286     0.2857     0.2545     2/7    (28.6%)
```

**Symbols**:
- ✅ = Detecting well (>70% recall)
- ⚠️ = Moderate (50-70% recall)
- ❌ = Poor (<50% recall)

---

## 🛑 When to Cancel Training

### Auto-Stop
Training stops automatically if:
```
⚠️ No improvement for 5/5 epochs
🛑 EARLY STOPPING: No improvement for 5 epochs
```

### Manual Stop (Ctrl+C)
Stop if you see for 5+ epochs:
- Loss not decreasing
- Accuracy stuck
- F1 score not improving

---

## 💪 GPU Optimization

### Check GPU Usage:
```bash
watch -n 1 nvidia-smi
```

**Should see**:
- GPU Memory: 8-12 GB used
- GPU Utilization: 80-100%
- Power: 250W+ (near max)

### If GPU underutilized:
Edit `start_training_gpu.sh`, increase batch size:
```bash
--batch-size 24  # Instead of 16
```

---

## 📊 Dataset Overview

```
FORGE Balanced Dataset: 6,575 contracts
├── Train: 4,540 (69%)
├── Val:   1,011 (15%)
└── Test:  1,024 (16%)

Vulnerability Types (11):
✅ Well-represented (600+ samples):
   - arithmetic: 663
   - unchecked_low_level_calls: 666
   - access_control: 629
   - safe: 606
   - reentrancy: 553
   - other: 620

⚠️ Moderate (100-400 samples):
   - denial_of_service: 317
   - time_manipulation: 206
   - front_running: 138
   - bad_randomness: 112

❌ Very imbalanced (<100 samples):
   - short_addresses: 30 (will be hard to detect!)
```

---

## 🎯 Expected Results

**Overall Accuracy**: 60-72%

**Best Detected** (using PDG structure):
- Reentrancy: 75-82%
- Arithmetic: 70-77%
- Unchecked Calls: 70-76%

**Challenging** (not visible in PDG):
- Short Addresses: 30-45%
- Bad Randomness: 45-55%
- Front Running: 50-60%

---

## ⏱️ Training Time

**GPU (A6000)**:
- First epoch: ~25-30 min (extracts PDGs)
- Later epochs: ~8-10 min (uses cache)
- **Total**: 2-3 hours for 20 epochs

**CPU**:
- First epoch: ~2-3 hours
- Later epochs: ~45-60 min
- **Total**: 15-20 hours for 20 epochs

---

## 📈 Monitoring Tools

### TensorBoard (Real-time graphs):
```bash
tensorboard --logdir runs/
```
Open: http://localhost:6006

### Training Log (File):
```bash
tail -f logs/training_gpu_YYYYMMDD_HHMMSS.log
```

### GPU Monitor:
```bash
watch -n 1 nvidia-smi
```

---

## 📂 Output Files

After training completes:

1. **`models/checkpoints/static_encoder_best.pt`**
   - Best model (highest validation F1)

2. **`models/checkpoints/test_results_*.txt`**
   - Final test metrics (per-vulnerability)

3. **`logs/training_gpu_*.log`**
   - Complete training log

4. **`runs/static_optimized_*/`**
   - TensorBoard logs

---

## 🔬 Understanding the Model

### Input (Intermediate Representation):
```
Smart Contract (.sol)
    ↓
Slither Analysis
    ↓
Program Dependence Graph (PDG)
    ├── Nodes: functions, variables, modifiers
    └── Edges: calls, reads, writes, uses_modifier
```

### Model Architecture:
```
PDG → Node Encoding (5→128 dim)
    → GAT Layer 1 (128→256 dim)
    → GAT Layer 2 (256→256 dim)
    → GAT Layer 3 (256→256 dim)
    → Global Pooling
    → Projection (256→768 dim)
    → 11 Classification Heads
    → Vulnerability Prediction
```

### Loss Function:
```python
CrossEntropyLoss(predictions, ground_truth, class_weights)
```

**Purpose**: Measures prediction error
- Low loss (0.2) = good predictions
- High loss (2.5) = poor predictions
- Training minimizes loss via backpropagation

### Class Weights:
Handles imbalanced dataset:
- short_addresses (30 samples): weight = 11.0
- arithmetic (663 samples): weight = 0.5

---

## 🐛 Troubleshooting

### Out of Memory
```bash
--batch-size 8  # Reduce from 16
```

### Training Too Slow
```bash
--batch-size 24  # Increase if GPU has memory
--num-workers 16  # Use more CPU cores
```

### Poor Accuracy
Try different learning rate:
```bash
--learning-rate 0.0001  # More stable
# or
--learning-rate 0.01    # Faster convergence
```

### Slither Errors
Check Solidity compiler:
```bash
solc-select use 0.8.0
```

---

## ✅ Quick Checklist

Before training:
- [ ] GPU available (`nvidia-smi` works)
- [ ] Dataset exists (`data/datasets/forge_balanced_accurate/`)
- [ ] Virtual env activated (`source triton_env/bin/activate`)

During training:
- [ ] GPU usage 80-100% (`watch -n 1 nvidia-smi`)
- [ ] Loss decreasing over epochs
- [ ] Accuracy increasing over epochs
- [ ] No warning messages for 5+ epochs

After training:
- [ ] Best model saved (`models/checkpoints/static_encoder_best.pt`)
- [ ] Test results generated (`test_results_*.txt`)
- [ ] Overall accuracy > 60%

---

## 📞 Quick Commands Reference

```bash
# Start training
./start_training_gpu.sh

# Monitor GPU
watch -n 1 nvidia-smi

# View TensorBoard
tensorboard --logdir runs/

# Check training log
tail -f logs/training_gpu_*.log

# Stop training
Ctrl+C

# Test trained model
python test_static_model.py  # (create if needed)
```

---

## 🎓 Key Concepts

**Program Dependence Graph (PDG)**:
- Graph showing relationships in smart contract
- Nodes = functions, variables, modifiers
- Edges = calls, reads, writes
- Captures control/data flow

**Graph Attention Network (GAT)**:
- Neural network for graph data
- Learns important relationships via attention
- 3 layers, 8 attention heads each

**Loss Function**:
- Metric of prediction error
- Backpropagation minimizes loss
- Model learns by adjusting weights

**Class Weights**:
- Handle imbalanced dataset
- Rare classes get higher weights
- Prevents model ignoring minority classes

**Early Stopping**:
- Stops if no improvement for 5 epochs
- Prevents overfitting
- Saves time

---

## 📚 Related Documentation

- **TRAINING_QUICK_START.md** - Quick reference
- **STATIC_TRAINING_GUIDE.md** - Detailed guide
- **README.md** - Project overview
- **PROJECT_ORGANIZATION.md** - File structure

---

**Last Updated**: 2025-11-19
