# How to Train Triton with FORGE Dataset

## ✅ Quick Start

Train all components with validation:

```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --num-epochs 20 \
    --batch-size 8
```

---

## 📊 When Does Validation Happen?

### During Training

**After EVERY epoch**, for each component:

1. **Static Encoder Training** (Phase 1):
   - Train on train set → Calculate loss and accuracy
   - **Validate on val set** → Calculate val_loss and val_acc
   - **Save model if val_loss improves** ✅
   - Repeat for 20 epochs

2. **Dynamic Encoder Training** (Phase 2):
   - Train on train set → Calculate loss and accuracy
   - **Validate on val set** → Calculate val_loss and val_acc
   - **Save model if val_loss improves** ✅
   - Repeat for 20 epochs

3. **Semantic Encoder Training** (Phase 3):
   - Train on train set → Calculate loss and accuracy
   - **Validate on val set** → Calculate val_loss and val_acc
   - **Save model if val_loss improves** ✅
   - Repeat for 20 epochs

4. **Fusion Module Training** (Phase 4):
   - Train on train set → Calculate loss and accuracy
   - **Validate on val set** → Calculate val_loss and val_acc
   - **Save best fusion model** ✅
   - Repeat for 20 epochs

### Example Output Per Epoch

```
Epoch 5/20
Train Loss: 2.3456, Train Acc: 35.67%
Val Loss: 2.4123, Val Acc: 33.21%
✓ Saved best static encoder (val_loss: 2.4123)
```

---

## 🎯 What Gets Saved?

### Best Models (Based on Validation Performance)

After training, you'll have:

```
models/checkpoints/
├── static_encoder_best.pt          ← Best static model (lowest val_loss)
├── dynamic_encoder_best.pt         ← Best dynamic model
├── semantic_encoder_best.pt        ← Best semantic model
├── fusion_module_best.pt           ← Best fusion model
├── static_encoder_fusion_best.pt   ← Static after fusion training
├── dynamic_encoder_fusion_best.pt  ← Dynamic after fusion training
└── semantic_encoder_fusion_best.pt ← Semantic after fusion training
```

**Key Point**: Only the epoch with the **lowest validation loss** is saved!

---

## 📈 Training Process Timeline

### Total Training Time: ~8-12 hours (with GPU)

```
Phase 1: Static Encoder   (20 epochs × ~5 min)   = ~1.5 hours
Phase 2: Dynamic Encoder  (20 epochs × ~8 min)   = ~2.5 hours
Phase 3: Semantic Encoder (20 epochs × ~15 min)  = ~5 hours
Phase 4: Fusion Module    (20 epochs × ~10 min)  = ~3 hours
```

### What You'll See During Training

```
================================================================================
COMPLETE TRITON TRAINING PIPELINE
================================================================================
Training directory: data/datasets/forge_balanced_accurate/train
Validation directory: data/datasets/forge_balanced_accurate/val
Batch size: 8
Epochs: 20
Learning rate: 0.001

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
Training Static Encoder: 100%|████████| 568/568 [05:23<00:00]
Train Loss: 2.5678, Train Acc: 25.34%
Val Loss: 2.6123, Val Acc: 23.45%
✓ Saved best static encoder (val_loss: 2.6123)

Epoch 2/20
Training Static Encoder: 100%|████████| 568/568 [05:21<00:00]
Train Loss: 2.3456, Train Acc: 28.91%
Val Loss: 2.4123, Val Acc: 27.12%
✓ Saved best static encoder (val_loss: 2.4123)

...

Static Encoder training complete! Best val_loss: 2.1234

================================================================================
PHASE 2: Training Dynamic Encoder
================================================================================
...
```

---

## 🔍 Command Line Options

### Basic Training

```bash
# Train all components (recommended)
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --num-epochs 20 \
    --batch-size 8
```

### Train Individual Components

```bash
# Train only static encoder
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --train-mode static \
    --num-epochs 20

# Train only fusion (requires pre-trained encoders)
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --train-mode fusion \
    --num-epochs 20
```

### Adjust Hyperparameters

```bash
# Reduce batch size if GPU runs out of memory
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --num-epochs 20 \
    --batch-size 4 \
    --learning-rate 0.0005
```

### Skip Encoder Tests (Faster Start)

```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --num-epochs 20 \
    --batch-size 8 \
    --skip-tests
```

---

## 📊 Monitoring Training

### Watch Training Progress

```bash
# Run training in background and save logs
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --num-epochs 20 \
    --batch-size 8 \
    2>&1 | tee training_log_$(date +%Y%m%d_%H%M%S).txt
```

### Check GPU Usage

```bash
# In another terminal
watch -n 1 nvidia-smi
```

### Expected Metrics

**Good Training Progress:**
- ✅ Train loss decreasing
- ✅ Val loss decreasing
- ✅ Train accuracy increasing
- ✅ Val accuracy increasing
- ✅ Val loss < Train loss (small gap = good)

**Warning Signs:**
- ⚠️ Val loss increasing while train loss decreasing = **Overfitting**
- ⚠️ Both losses flat = **Learning rate too low**
- ⚠️ Loss = NaN = **Learning rate too high** or **numerical instability**

---

## 🎯 After Training

### Test Your Models

```bash
# Test on held-out test set
python scripts/test_dataset_performance.py \
    --dataset data/datasets/forge_balanced_accurate/test
```

### Expected Results

**Before (Current Dataset - 155 samples):**
- Static: 12% accuracy
- Dynamic: 20% accuracy
- Semantic: 50% accuracy
- Fusion: 0% (broken)

**After (FORGE Dataset - 4,540 samples):**
- Static: 30-40% accuracy ✅
- Dynamic: 35-45% accuracy ✅
- Semantic: 60-70% accuracy ✅
- Fusion: 55-70% accuracy ✅

---

## ⚠️ Troubleshooting

### Out of Memory Error

```bash
# Reduce batch size
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --batch-size 2  # ← Reduced from 8
```

### Training Too Slow

```bash
# Check if GPU is being used
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# If False, install CUDA-enabled PyTorch:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Validation Accuracy Not Improving

This is normal early in training. Give it at least 5-10 epochs.

If still not improving after 10 epochs:
- ✅ Reduce learning rate: `--learning-rate 0.0001`
- ✅ Increase epochs: `--num-epochs 30`
- ✅ Check class weights are being used (should see in logs)

---

## 🎉 Summary

### Key Points

1. **Validation happens after EVERY epoch** ✅
2. **Only best models (lowest val_loss) are saved** ✅
3. **Use separate val folder with `--val-dir`** ✅
4. **Training takes 8-12 hours with GPU** ⏰
5. **Expected 3-5x accuracy improvement** 📈

### Start Training NOW

```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --num-epochs 20 \
    --batch-size 8
```

Good luck! 🚀
