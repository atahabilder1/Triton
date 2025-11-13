# Triton Documentation Index

**Last Updated**: November 13, 2025

---

## 📖 Quick Navigation

### 🏠 Root Documentation
- [**Main README**](../README.md) - Project overview and quick start
- [**Weekly Progress Report**](../WEEKLY_PROGRESS_NOV13_2025.md) - Current week's progress (Nov 13, 2025)
- [**Project Organization**](../PROJECT_ORGANIZATION.md) - Project structure and file organization

---

## 📚 Guides (How-To Documentation)

Located in: `docs/guides/`

1. [**HOW_TO_TRAIN.md**](guides/HOW_TO_TRAIN.md)
   - Step-by-step training guide
   - Command examples and parameters
   - Training best practices

2. [**TRAINING_WITH_CUSTOM_DATASETS.md**](guides/TRAINING_WITH_CUSTOM_DATASETS.md)
   - Using custom datasets for training
   - Dataset preparation and formatting
   - Custom dataset configuration

3. [**QUICK_ANSWERS.md**](guides/QUICK_ANSWERS.md)
   - Frequently asked questions
   - Quick troubleshooting tips
   - Common use cases

4. [**TESTING_GUIDE.md**](TESTING_GUIDE.md)
   - Testing procedures
   - Test dataset usage
   - Evaluation metrics

5. [**TEST_USAGE_EXAMPLES.md**](TEST_USAGE_EXAMPLES.md)
   - Practical testing examples
   - Sample commands and outputs

---

## 📊 Reports (Analysis & Results)

Located in: `docs/reports/`

### FORGE Dataset Reports
1. [**FORGE_DATASET_ANALYSIS.md**](reports/FORGE_DATASET_ANALYSIS.md) - Initial FORGE dataset analysis
2. [**FORGE_CLASSES_SUMMARY.md**](reports/FORGE_CLASSES_SUMMARY.md) - Vulnerability class breakdown
3. [**FORGE_DATASET_GUIDE.md**](reports/FORGE_DATASET_GUIDE.md) - Guide to using FORGE dataset
4. [**FORGE_INTEGRATION_COMPLETE.md**](reports/FORGE_INTEGRATION_COMPLETE.md) - Integration completion report
5. [**FORGE_TRAINING_PLAN.md**](reports/FORGE_TRAINING_PLAN.md) - Training strategy for FORGE
6. [**ACCURATE_CWE_MAPPING_RESULTS.md**](reports/ACCURATE_CWE_MAPPING_RESULTS.md) - CWE to vulnerability class mapping

### Dataset & Training Reports
7. [**DATASET_READY.md**](reports/DATASET_READY.md) - Dataset preparation completion
8. [**LABELED_DATASET_SUMMARY.md**](LABELED_DATASET_SUMMARY.md) - Summary of labeled datasets
9. [**ENHANCED_TRAINING_FEATURES.md**](reports/ENHANCED_TRAINING_FEATURES.md) - New training features added
10. [**TEST_TRAINING_RESULTS.md**](reports/TEST_TRAINING_RESULTS.md) - Test training run results

### Comparison Reports
11. [**PREVIOUS_VS_NEW_SETUP.md**](reports/PREVIOUS_VS_NEW_SETUP.md) - Comparison of old vs new setup

---

## 🗂️ Archive

Located in: `docs/archive/`

Historical documentation and superseded files are moved here when no longer current.

---

## 📈 Current Status

**Training Status**: ✅ Running (Started: Nov 13, 2025 at 03:49 AM)
- **Dataset**: FORGE (6,575 contracts)
- **Training**: 4,540 contracts (11 classes)
- **Validation**: 1,011 contracts
- **Expected Duration**: 8-12 hours

**Latest Updates**:
- ✅ FORGE dataset integrated (6,575 contracts)
- ✅ Safe class added (889 contracts, 13.5%)
- ✅ Enhanced training features (TensorBoard, checkpoints, per-class metrics)
- ✅ Class weighting implemented (22.3:1 imbalance ratio)

---

## 🔍 Finding Information

### By Topic

**Getting Started**:
- Read: [Main README](../README.md)
- Then: [HOW_TO_TRAIN.md](guides/HOW_TO_TRAIN.md)

**Dataset Information**:
- Overview: [FORGE_DATASET_GUIDE.md](reports/FORGE_DATASET_GUIDE.md)
- Analysis: [FORGE_DATASET_ANALYSIS.md](reports/FORGE_DATASET_ANALYSIS.md)
- Classes: [FORGE_CLASSES_SUMMARY.md](reports/FORGE_CLASSES_SUMMARY.md)

**Training**:
- Guide: [HOW_TO_TRAIN.md](guides/HOW_TO_TRAIN.md)
- Custom datasets: [TRAINING_WITH_CUSTOM_DATASETS.md](guides/TRAINING_WITH_CUSTOM_DATASETS.md)
- Features: [ENHANCED_TRAINING_FEATURES.md](reports/ENHANCED_TRAINING_FEATURES.md)

**Testing**:
- Guide: [TESTING_GUIDE.md](TESTING_GUIDE.md)
- Examples: [TEST_USAGE_EXAMPLES.md](TEST_USAGE_EXAMPLES.md)
- Results: [TEST_TRAINING_RESULTS.md](reports/TEST_TRAINING_RESULTS.md)

**Current Progress**:
- Report: [WEEKLY_PROGRESS_NOV13_2025.md](../WEEKLY_PROGRESS_NOV13_2025.md)

---

## 📝 Documentation Standards

When creating new documentation:

1. **Location**:
   - Guides/How-tos → `docs/guides/`
   - Reports/Analysis → `docs/reports/`
   - Current status → Root directory
   - Archived → `docs/archive/`

2. **Naming Convention**:
   - Use UPPERCASE for important docs: `TRAINING_GUIDE.md`
   - Use descriptive names: `FORGE_DATASET_ANALYSIS.md`
   - Add dates for reports: `WEEKLY_PROGRESS_NOV13_2025.md`

3. **Format**:
   - Start with title and summary
   - Include table of contents for long docs
   - Use clear section headers
   - Add code examples where relevant
   - Include timestamps/dates

---

## 🔗 External Resources

- **FORGE Paper**: [arXiv:2506.18795](https://arxiv.org/abs/2506.18795) (ICSE 2026)
- **FORGE Artifacts**: `/data/llm_projects/triton_datasets/FORGE-Artifacts/`
- **Project Repository**: `/home/anik/code/Triton/`

---

**For the most current information**, always check:
1. [WEEKLY_PROGRESS_NOV13_2025.md](../WEEKLY_PROGRESS_NOV13_2025.md) - Current status
2. Training logs: `logs/training_*.log`
3. TensorBoard: `tensorboard --logdir runs/`
