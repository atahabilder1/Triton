# Triton Project Organization

**Last Updated:** November 6, 2025
**Status:** Cleaned and Organized

---

## 📁 Project Structure

```
Triton/
├── 📄 README.md                    # Main project documentation
├── 📄 LICENSE                      # MIT License
├── 📄 requirements.txt             # Python dependencies
│
├── 🔧 test_modality.py            # Main testing script (unified)
├── 🔧 test.sh                     # Run all 4 tests automatically
├── 🔧 start_training.sh           # Main training script
│
├── 📂 encoders/                   # Encoder implementations
│   ├── static_encoder.py          # PDG-based graph encoder (GAT)
│   ├── dynamic_encoder.py         # Execution trace encoder (LSTM)
│   └── semantic_encoder.py        # CodeBERT semantic encoder
│
├── 📂 fusion/                     # Cross-modal fusion
│   └── cross_modal_fusion.py     # Attention-based fusion module
│
├── 📂 orchestrator/              # Agentic workflow
│   └── agentic_workflow.py       # Multi-agent orchestration
│
├── 📂 tools/                     # Analysis tools
│   ├── slither_wrapper.py        # Slither integration (PDG extraction)
│   └── mythril_wrapper.py        # Mythril integration (trace analysis)
│
├── 📂 utils/                     # Utilities
│   └── helpers.py                # Helper functions
│
├── 📂 scripts/                   # Training & utility scripts
│   ├── train_complete_pipeline.py    # Main training pipeline
│   ├── testing/                      # Old test scripts (archived)
│   │   ├── test_all_models.py
│   │   ├── test_each_modality.py
│   │   ├── test_comprehensive_report.py
│   │   ├── test_models_detailed.py
│   │   └── test_with_safe_detection.py
│   ├── check_errors.sh               # Error checking utility
│   ├── monitor_training.sh           # Training monitor
│   ├── run_full_test.sh              # Comprehensive test runner
│   ├── run_test.sh                   # Quick test runner
│   └── train_fusion_nocache.sh       # Fusion training (no cache)
│
├── 📂 docs/                      # Documentation
│   ├── README.md                     # Docs index
│   ├── LABELED_DATASET_SUMMARY.md    # Dataset documentation
│   ├── TESTING_GUIDE.md              # How to test models
│   ├── TEST_USAGE_EXAMPLES.md        # Test script examples
│   ├── guides/                       # User guides
│   │   ├── DATASET_GUIDE.md
│   │   ├── HOW_TO_USE.md
│   │   └── README_SIMPLE.md
│   ├── reports/                      # Weekly/final reports
│   │   ├── WEEKLY_PROGRESS_REPORT_NOV_5_2025.md
│   │   ├── SESSION_SUMMARY_NOV_5-6_2025.md
│   │   ├── FINAL_TEST_RESULTS_NOV_6_2025.md
│   │   └── TECHNICAL_QA_ANSWERS.md
│   └── archive/                      # Archived docs
│       ├── FINAL_PDG_FIX_STATUS.md
│       ├── FIXES_COMPLETE.md
│       ├── IMPROVEMENTS_IMPLEMENTED.md
│       ├── PERFORMANCE_SUMMARY.md
│       ├── TEST_RESULTS_ANALYSIS.md
│       ├── TRAINING_PROCESS_EXPLAINED.md
│       └── TRAINING_STATUS.md
│
├── 📂 models/                    # Model checkpoints
│   └── checkpoints/
│       ├── static_encoder_best.pt
│       ├── dynamic_encoder_best.pt
│       ├── semantic_encoder_best.pt
│       ├── fusion_module_best.pt
│       ├── static_encoder_fusion_best.pt
│       ├── dynamic_encoder_fusion_best.pt
│       └── semantic_encoder_fusion_best.pt
│
├── 📂 data/                      # Datasets
│   └── datasets/
│       ├── combined_labeled/         # Main dataset (228 contracts)
│       │   ├── train/               # 155 contracts
│       │   ├── val/                 # 29 contracts
│       │   ├── test/                # 44 contracts
│       │   ├── dataset_summary.json
│       │   └── train_val_test_splits.json
│       └── smartbugs/               # Original SmartBugs dataset
│
├── 📂 logs/                      # Training & test logs (29 files)
│   ├── all_modalities_test.log
│   ├── detailed_test_output.log
│   ├── final_retrain_*.log
│   ├── fusion_*.log
│   ├── retrain_output_*.log
│   ├── test_*.log
│   ├── training_*.log
│   └── ...
│
├── 📂 tests/                     # Unit tests
│   └── test_unit_components.py
│
└── 📂 results/                   # Test results (if any)

```

---

## 🚀 Quick Start Commands

### Training:
```bash
./start_training.sh
```

### Testing:
```bash
# Test all 4 modalities
./test.sh

# Test individual modality
python3 test_modality.py --modality semantic
python3 test_modality.py --modality static
python3 test_modality.py --modality dynamic
python3 test_modality.py --modality fusion
```

---

## 📚 Key Documentation

| File | Purpose | Location |
|------|---------|----------|
| **README.md** | Main project overview | Root |
| **LABELED_DATASET_SUMMARY.md** | Dataset documentation | `docs/` |
| **TESTING_GUIDE.md** | How to test models | `docs/` |
| **WEEKLY_PROGRESS_REPORT_NOV_5_2025.md** | Latest progress | `docs/reports/` |
| **SESSION_SUMMARY_NOV_5-6_2025.md** | Complete session log | `docs/reports/` |
| **FINAL_TEST_RESULTS_NOV_6_2025.md** | Test results | `docs/reports/` |

---

## 🧹 What Was Cleaned

### Moved to `logs/` (29 files):
- All `.log` files from root directory
- Training logs, test logs, retrain logs

### Moved to `scripts/`:
- All utility shell scripts (`.sh`)
- Training helper scripts
- Old test scripts → `scripts/testing/`

### Moved to `docs/`:
- User guides → `docs/guides/`
- Final reports → `docs/reports/`
- Old/duplicate docs → `docs/archive/`

### Removed:
- `detailed_test_results.json` - Duplicate data
- `__init__.py` - Empty file
- `main.py` - Unused entry point

---

## 📊 File Statistics

- **Root directory:** 10 files (clean!)
- **Documentation:** 15+ organized files
- **Scripts:** 10+ organized in `scripts/`
- **Logs:** 29 organized in `logs/`
- **Models:** 7 checkpoints in `models/checkpoints/`
- **Test data:** 228 contracts properly organized

---

## ✅ Current Root Directory (Clean)

```
Triton/
├── README.md              # Main documentation
├── LICENSE                # License file
├── requirements.txt       # Dependencies
├── test_modality.py       # Main test script
├── test.sh                # Run all tests
├── start_training.sh      # Main training script
├── data/                  # Datasets
├── docs/                  # All documentation
├── encoders/              # Encoder code
├── fusion/                # Fusion module
├── logs/                  # All logs
├── models/                # Checkpoints
├── orchestrator/          # Workflow
├── scripts/               # Utility scripts
├── tests/                 # Unit tests
├── tools/                 # Analysis tools
└── utils/                 # Helpers
```

**Everything is now organized and easy to find!** 🎯
