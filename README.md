# Triton: Multi-Modal Smart Contract Vulnerability Detection

Triton is an AI-powered system that detects vulnerabilities in Ethereum smart contracts using:
- **Static Analysis** (Slither): Program Dependence Graphs (PDG)
- **Dynamic Analysis** (Mythril): Execution Traces
- **Semantic Analysis** (CodeBERT): Code Understanding
- **Cross-Modal Fusion**: Combines all three for better accuracy

---

## 🚀 Quick Start

### Static Vulnerability Detection (Recommended)

**GPU-Optimized Training:**
```bash
./start_static_training_gpu.sh
```

**Standard Training:**
```bash
./start_static_training.sh
```

**Time:** 2-3 hours (GPU) | 12-15 hours (CPU)

See **[QUICK_START.md](QUICK_START.md)** for detailed guide.

### Full Multi-Modal Pipeline

```bash
./start_full_training.sh
```

**Time:** 8-12 hours (trains all 4 models: static, dynamic, semantic, fusion)

---

## 📖 Documentation

### Quick References
- **[QUICK_START.md](QUICK_START.md)** - Static training quick start
- **[docs/guides/TRAINING_QUICK_START.md](docs/guides/TRAINING_QUICK_START.md)** - Quick reference guide
- **[docs/guides/STATIC_TRAINING_GUIDE.md](docs/guides/STATIC_TRAINING_GUIDE.md)** - Detailed training guide
- **[docs/guides/TRAINING_SUMMARY.md](docs/guides/TRAINING_SUMMARY.md)** - Complete summary

### Detailed Guides
- **[docs/guides/HOW_TO_TRAIN.md](docs/guides/HOW_TO_TRAIN.md)** - Multi-modal training guide
- **[docs/guides/DATASET_AND_TRAINING_SUMMARY.md](docs/guides/DATASET_AND_TRAINING_SUMMARY.md)** - Dataset information
- **[PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md)** - Project structure

---

## 📊 Expected Performance

**Dataset:** FORGE (6,575 contracts)
- Train: 4,540 contracts (69%)
- Validation: 1,011 contracts (15%)
- Test: 1,024 contracts (16%)

**Static Encoder Detection Rates:**
- Overall Accuracy: 60-72%
- Reentrancy: 75-82% ✅
- Arithmetic: 70-77% ✅
- Unchecked Calls: 70-76% ✅
- Access Control: 60-70% ⚠️
- Short Addresses: 30-45% ❌ (very imbalanced class)

---

## 📁 Project Structure

```
Triton/
├── scripts/                          # Training & testing scripts
│   ├── train_static_optimized.py       # GPU-optimized static training ⭐
│   ├── train_static_only.py            # Standard static training
│   ├── train_complete_pipeline.py      # Multi-modal pipeline
│   ├── test_dataset_performance.py     # Testing script
│   ├── monitor_training_detailed.sh    # Training monitoring
│   └── quick_status.sh                 # Quick training status
│
├── encoders/                         # Model architectures
│   ├── static_encoder.py               # PDG + GAT model
│   ├── dynamic_encoder.py              # Execution trace + LSTM
│   ├── semantic_encoder.py             # CodeBERT fine-tuning
│   └── ...
│
├── fusion/                           # Cross-modal fusion
│   └── cross_modal_fusion.py           # Fusion module
│
├── tools/                            # Analysis tools
│   └── slither_wrapper.py              # PDG extraction (Slither)
│
├── models/checkpoints/               # Trained models
│   ├── static_encoder_best.pt          # Best static model
│   └── ...
│
├── data/datasets/                    # Training data
│   └── forge_balanced_accurate/        # 6,575 contracts
│       ├── train/                      # 4,540 contracts
│       ├── val/                        # 1,011 contracts
│       └── test/                       # 1,024 contracts
│
├── docs/guides/                      # Documentation
│   ├── TRAINING_QUICK_START.md         # Quick reference
│   ├── STATIC_TRAINING_GUIDE.md        # Detailed guide
│   ├── TRAINING_SUMMARY.md             # Complete summary
│   ├── DATASET_AND_TRAINING_SUMMARY.md # Dataset info
│   └── HOW_TO_TRAIN.md                 # Multi-modal guide
│
├── logs/                             # Training logs
├── runs/                             # TensorBoard logs
├── results/                          # Test results
│
├── start_static_training_gpu.sh      # Static GPU-optimized launcher ⭐
├── start_static_training.sh          # Static standard launcher
├── start_full_training.sh            # Full pipeline launcher (all 4 models)
├── QUICK_START.md                    # Quick start guide
└── README.md                         # This file
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
2. **Extracts** features using 3 analysis methods:
   - **Static**: Program Dependence Graphs (Slither)
   - **Dynamic**: Execution Traces (Mythril)
   - **Semantic**: Code Embeddings (CodeBERT)
3. **Trains** neural network components:
   - Static: Graph Attention Network (GAT)
   - Dynamic: LSTM
   - Semantic: Fine-tuned Transformer
   - Fusion: Cross-modal attention
4. **Detects** 11 vulnerability types:
   - Reentrancy
   - Arithmetic Overflow/Underflow
   - Access Control
   - Unchecked Low-Level Calls
   - Bad Randomness
   - Denial of Service
   - Front Running
   - Time Manipulation
   - Short Address Attack
   - Other
   - Safe (no vulnerabilities)

---

## 📚 Training Modes

### Static-Only (Recommended)
- **Time:** 2-3 hours (GPU)
- **Model:** Graph Attention Network on PDGs
- **Accuracy:** 60-72%
- **Best for:** Control flow vulnerabilities (reentrancy, arithmetic, etc.)

### Full Multi-Modal Pipeline
- **Time:** 8-12 hours (GPU)
- **Models:** All 4 components (static, dynamic, semantic, fusion)
- **Accuracy:** 55-70% (fusion)
- **Best for:** Comprehensive detection

---

## 🔬 Real-Time Training Monitoring

During training, you'll see:

**Every 10 batches:**
```
Batch [  10/568] | Loss: 1.2345 | Acc: 45.67% | Speed: 1.23 batch/s | ETA: 7m 32s
```

**Every epoch:**
```
EPOCH 5/50 SUMMARY
Train Loss: 1.0234 | Train Acc: 52.34%
Val Loss:   0.9876 | Val Acc:   55.67% | Val F1: 0.5234
✅ NEW BEST MODEL SAVED!
```

**Every 5 epochs (detailed metrics):**
```
✅ reentrancy           0.7234  0.6891  0.7058   94/119 (79.0%)
⚠️  access_control      0.5234  0.5891  0.5546   87/148 (58.9%)
❌ short_addresses      0.2286  0.2857  0.2545    2/7   (28.6%)
```

See `docs/guides/TRAINING_QUICK_START.md` for interpretation.

---

## 💪 GPU Optimization

The training scripts are optimized for RTX A6000 (46GB VRAM):
- Batch size: 16 (adjust for your GPU)
- Parallel data loading: 8 workers
- Mixed precision training
- Automatic early stopping

**Monitor GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Expected:** 80-100% GPU utilization, 8-12 GB memory usage

---

## 📄 License

MIT License

---

**Last Updated:** November 19, 2025
