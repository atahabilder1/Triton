# Triton: Multi-Modal Smart Contract Vulnerability Detection

Triton is an AI-powered system that detects vulnerabilities in Ethereum smart contracts using:
- **Static Analysis** (Slither): Control Flow Graphs
- **Dynamic Analysis** (Mythril): Execution Traces
- **Semantic Analysis** (CodeBERT): Code Understanding
- **Cross-Modal Fusion**: Combines all three for better accuracy

---

## 🚀 Quick Start

### 1. Train the Model
```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/smartbugs-curated/dataset \
    --num-epochs 10
```

### 2. Test the Model
```bash
python scripts/test_dataset_performance.py --dataset smartbugs
```

### 3. View Results
```bash
cat results/triton_test_summary_*.txt
```

---

## 📖 Documentation

- **[HOW_TO_USE.md](HOW_TO_USE.md)** - Complete training and testing guide
- **[README_SIMPLE.md](README_SIMPLE.md)** - Quick reference card
- **[docs/](docs/)** - Detailed guides

---

## 📊 Current Performance

**Dataset:** 143 labeled contracts (SmartBugs Curated)

**Detection Rates:**
- Overall: 10.49%
- Reentrancy: 32.26% (best)
- Arithmetic: 20.00%
- Access Control: 11.11%

**Why low?** Small dataset (218 contracts) + class imbalance. Needs data augmentation.

---

## 📁 Project Structure

```
Triton/
├── scripts/
│   ├── train_complete_pipeline.py       # Main training script
│   └── test_dataset_performance.py      # Main testing script
│
├── models/checkpoints/               # Trained models
│
├── data/datasets/                    # Training data
│   └── smartbugs-curated/dataset/    # 143 labeled contracts
│
├── results/                          # Test results
│
└── HOW_TO_USE.md                     # Read this!
```

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- Slither
- Mythril
- PyTorch Geometric

Install:
```bash
pip install -r requirements.txt
```

---

## 🎯 What Triton Does

1. **Loads** smart contracts from dataset
2. **Extracts** features using 3 analysis tools
3. **Trains** 4 neural network components
4. **Detects** 10 vulnerability types:
   - Reentrancy
   - Arithmetic Overflow/Underflow
   - Access Control
   - Unchecked External Calls
   - Bad Randomness
   - Denial of Service
   - Front Running
   - Time Manipulation
   - Short Address Attack
   - Others

---

## 📚 Training Process (4 Phases)

**Phase 1:** Static Encoder (learns from CFGs)
**Phase 2:** Dynamic Encoder (learns from execution traces)
**Phase 3:** Semantic Encoder (fine-tunes CodeBERT)
**Phase 4:** Fusion Module (combines all three)

**Time:** 1-2 hours for 143 contracts

---

## 🔬 Testing Process

1. Load trained models
2. Analyze test contracts
3. Generate detection reports
4. Compare with ground truth

**Time:** 5-10 minutes for 143 contracts

---

## 💡 Improving Performance

Current detection is low due to small dataset. To improve:

1. **Add more labeled data** (target: 500-1000 contracts)
2. **Data augmentation** (5-10× increase)
3. **Fix Slither/Mythril errors** (currently ~100% failure)
4. **Train longer** (more epochs)
5. **Tune hyperparameters** (batch size, learning rate)

---

## 📄 License

[Your License Here]

## 🤝 Contributing

[Contributing Guidelines]

---

**Last Updated:** November 5, 2025
