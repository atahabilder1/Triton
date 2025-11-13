# Triton Vulnerability Detection - Quick Reference

## 🎯 Two Commands You Need

### 1️⃣ Train
```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs-curated/dataset \
    --num-epochs 10
```

### 2️⃣ Test
```bash
python scripts/test_dataset_performance.py --dataset smartbugs
```

## 📖 Full Documentation
See [HOW_TO_USE.md](HOW_TO_USE.md) for complete guide.

## 📊 Current Performance
- Detection Rate: 10.49%
- Best: Reentrancy (32.26%)
- Dataset: 143 labeled contracts

## 🔧 Quick Options

### Fast Test (3 epochs, 50 samples)
```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs-curated/dataset \
    --num-epochs 3 \
    --max-samples 50
```

### View Results
```bash
cat results/triton_test_summary_*.txt
```
