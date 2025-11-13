# Old Scripts Archive

This folder contains old/outdated scripts that have been superseded by newer versions.

## Archived Files

| File | Date | Status | Replacement |
|------|------|--------|-------------|
| `test_models.py` | Nov 6 | Outdated | Use `scripts/testing/test_*.py` |
| `test.sh` | Nov 6 | Outdated | Use `scripts/run_test.sh` |
| `test_train.sh` | Nov 13 | Outdated | Use `start_full_training.sh` |
| `training_log.txt` | Nov 13 | Old log | Check `logs/training_*.log` |

## Why Archived?

These scripts were from **early development** and have been replaced by:

1. **Better organized scripts** in `scripts/` folder:
   - `scripts/train_complete_pipeline.py` - Main training
   - `scripts/testing/` - Testing suite
   - `scripts/training/` - Training utilities

2. **Improved shell scripts** in root:
   - `start_full_training.sh` - FORGE dataset training
   - `monitor_training.sh` - Training monitoring

## Current Structure

```
/home/anik/code/Triton/
├── start_full_training.sh      # ← Use this for FORGE training
├── monitor_training.sh         # ← Use this for monitoring
│
├── scripts/
│   ├── train_complete_pipeline.py    # Main training script
│   ├── testing/                      # Test scripts
│   └── training/                     # Training utilities
│
└── logs/                             # Current training logs
```

---

**Note**: These files are kept for reference only. Use the current scripts for all new work.
