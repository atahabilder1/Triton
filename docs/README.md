# Triton Documentation Index

Welcome to the Triton vulnerability detection system documentation.

## 📚 Documentation Structure

### Core Documentation

- **[Dataset Analysis](DATASET_ANALYSIS.md)** ⭐ **NEW** - Critical analysis of training/test data issues
- **[Technical Documentation](TECHNICAL_DOCUMENTATION.md)** - System architecture and design

### Quick Start Guides

Navigate to [`docs/guides/`](guides/) for:

- **[Quick Start](guides/QUICK_START.md)** - Get up and running in 5 minutes
- **[Training Guide](guides/TRAINING_GUIDE.md)** - Complete training pipeline walkthrough
- **[Dataset Guide](guides/DATASET_GUIDE.md)** - Working with vulnerability datasets

### Reports & Analysis

Navigate to [`docs/reports/`](reports/) for:

- **[Training Summary](reports/TRAINING_SUMMARY.md)** - Latest training run results
- **[Performance Tracking](reports/PERFORMANCE_TRACKING.md)** - Performance metrics over time
- **[Weekly Progress](reports/WEEKLY_PROGRESS.md)** - Development progress log
- **[Training Issue Report](reports/TRAINING_ISSUE_REPORT.md)** - Known issues and fixes

### Archive

Old/duplicate documentation moved to [`docs/archive/`](archive/)

---

## 🚀 Quick Links

### For New Users
1. Start with [Quick Start](guides/QUICK_START.md)
2. Read [Dataset Analysis](DATASET_ANALYSIS.md) to understand current issues
3. Check [Training Guide](guides/TRAINING_GUIDE.md) before training

### For Developers
1. Review [Technical Documentation](TECHNICAL_DOCUMENTATION.md)
2. Check [Training Issue Report](reports/TRAINING_ISSUE_REPORT.md) for known bugs
3. See [Weekly Progress](reports/WEEKLY_PROGRESS.md) for recent changes

### For Troubleshooting
1. **Poor Performance?** → Read [Dataset Analysis](DATASET_ANALYSIS.md)
2. **Training Errors?** → Check [Training Issue Report](reports/TRAINING_ISSUE_REPORT.md)
3. **Setup Issues?** → Run `scripts/utils/verify_setup.sh`

---

## 📊 Current Status (Nov 2, 2025)

### Test Results
- **Overall Detection**: 12.59% (18/143 contracts)
- **Reentrancy**: 54.84% ✅
- **Other Types**: 0-33% ❌

### Known Issues
1. ❌ Model trained on unlabeled FORGE data (78,224 contracts)
2. ❌ Tested on labeled SmartBugs data (143 contracts)
3. ❌ No class weights in loss function
4. ❌ Severe class imbalance (52:1 ratio)

### Recommended Actions
See [Dataset Analysis](DATASET_ANALYSIS.md) for:
- Root cause explanation
- Step-by-step fixes
- Expected improvements (12% → 60-80%)

---

## 🛠️ Useful Scripts

Training scripts in `scripts/training/`:
- `start_training.sh` - Start training in background
- `stop_training.sh` - Stop running training
- `monitor_training.sh` - Watch training progress

Testing scripts in `scripts/testing/`:
- `quick_check.sh` - Quick model test
- `run_tests.sh` - Full test suite

Utility scripts in `scripts/utils/`:
- `verify_setup.sh` - Check environment
- `check_training_completion.sh` - Monitor training completion

---

**Last Updated**: November 2, 2025
