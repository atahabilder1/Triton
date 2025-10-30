# Triton Testing Guide - Complete Results with Vulnerability Breakdown

This guide shows you exactly how to run Triton tests and get comprehensive results with category-wise vulnerability detection percentages.

## Step 1: Download Datasets

### Quick Start - Download All Datasets

```bash
python scripts/download_datasets.py --dataset all
```

This will download:
- **SmartBugs**: 143 vulnerable contracts (9 vulnerability types)
- **SolidiFI**: 9,369 contracts with injected vulnerabilities
- **SmartBugs Wild**: 47,398 real-world contracts
- **Audit Datasets**: Not-So-Smart-Contracts (Trail of Bits) and DeFi vulnerabilities
- **Securify**: Test contracts from Securify project

### Download Individual Datasets

```bash
# Download only SmartBugs (recommended for initial testing)
python scripts/download_datasets.py --dataset smartbugs

# Download only SolidiFI
python scripts/download_datasets.py --dataset solidifi

# Download SmartBugs Wild (large dataset)
python scripts/download_datasets.py --dataset wild

# Download audit datasets
python scripts/download_datasets.py --dataset audits

# Download Securify dataset
python scripts/download_datasets.py --dataset securify
```

### Check Downloaded Datasets

```bash
# View dataset summary
cat data/datasets/dataset_summary.json

# List all downloaded contracts
find data/datasets -name "*.sol" | wc -l
```

---

## Step 2: Test Triton

### Test on SmartBugs Curated (Recommended Start)

SmartBugs Curated is the best dataset to start with - it has 143 contracts with known vulnerabilities organized by type.

```bash
cd /home/anik/code/Triton
source triton_env/bin/activate
python scripts/test_triton.py --dataset smartbugs --output-dir results
```

This will:
- Analyze all 143 SmartBugs Curated contracts
- Compare results with ground truth
- Calculate precision, recall, F1-score for each vulnerability type
- Generate 3 output files with comprehensive breakdown

### Test on SolidiFI

SolidiFI has many more contracts (~9,000+). The script limits to first 100 for initial testing.

```bash
python scripts/test_triton.py --dataset solidifi --output-dir results
```

### Test on Custom Contracts

If you have your own contracts to test:

```bash
python scripts/test_triton.py --dataset custom --custom-dir /path/to/your/contracts --output-dir results
```

### Test on All Datasets

```bash
python scripts/test_triton.py --dataset all --output-dir results
```

---

## Step 3: Review Results

After testing, you'll find **3 output files** in the `results/` directory:

### 1. Detailed Results (JSON)

```bash
cat results/triton_test_results_YYYYMMDD_HHMMSS.json
```

This contains:
- Per-contract analysis results
- Vulnerabilities detected
- Confidence scores
- Modality contributions
- Analysis time
- Comparison with ground truth

### 2. Summary Report (Text) - WITH VULNERABILITY BREAKDOWN TABLE

```bash
cat results/triton_test_summary_YYYYMMDD_HHMMSS.txt
```

This contains:
- Overall metrics (precision, recall, F1)
- Average analysis time
- Throughput (contracts/second)
- **DETAILED TABLE: Total vulnerabilities and detection percentage for EACH category**

Example output:
```
====================================================================================================
TRITON VULNERABILITY DETECTION - COMPREHENSIVE SUMMARY
====================================================================================================

Timestamp: 2025-10-30T15:30:00
System: Triton v2.0

OVERALL METRICS
----------------------------------------------------------------------------------------------------
Average Precision: 0.9250
Average Recall: 0.9100
Average F1: 0.9175
Average Analysis Time: 2.3500
Total Contracts Tested: 143

SMARTBUGS DATASET
====================================================================================================
Total Contracts: 143
Successful Analyses: 143
Total Time: 335.50s

VULNERABILITY DETECTION BREAKDOWN
====================================================================================================

Vulnerability Type              | Total    | Detected   | Missed   | Detection %
----------------------------------------------------------------------------------------------------
Access Control                  | 18       | 17         | 1        |      94.44%
Arithmetic                      | 15       | 14         | 1        |      93.33%
Bad Randomness                  | 8        | 7          | 1        |      87.50%
Denial Of Service               | 6        | 6          | 0        |     100.00%
Front Running                   | 4        | 3          | 1        |      75.00%
Other                           | 3        | 2          | 1        |      66.67%
Reentrancy                      | 31       | 30         | 1        |      96.77%
Time Manipulation               | 5        | 5          | 0        |     100.00%
Unchecked Low Level Calls       | 52       | 50         | 2        |      96.15%
----------------------------------------------------------------------------------------------------
TOTAL                           | 143      | 134        | 9        |      93.71%
====================================================================================================
```

### 3. Markdown Table (for Reports/Presentations)

```bash
cat results/triton_results_table_YYYYMMDD_HHMMSS.md
```

This contains GitHub-flavored markdown table you can directly copy-paste into:
- README files
- Research papers
- Presentations
- Reports

Example markdown output:
```markdown
# Triton Vulnerability Detection Results

## SMARTBUGS Dataset

| Vulnerability Type | Total | Detected | Missed | Detection Rate |
|-------------------|-------|----------|--------|----------------|
| Access Control | 18 | 17 | 1 | 94.44% |
| Arithmetic | 15 | 14 | 1 | 93.33% |
| Bad Randomness | 8 | 7 | 1 | 87.50% |
| Denial Of Service | 6 | 6 | 0 | 100.00% |
| Front Running | 4 | 3 | 1 | 75.00% |
| Other | 3 | 2 | 1 | 66.67% |
| Reentrancy | 31 | 30 | 1 | 96.77% |
| Time Manipulation | 5 | 5 | 0 | 100.00% |
| Unchecked Low Level Calls | 52 | 50 | 2 | 96.15% |
| **TOTAL** | **143** | **134** | **9** | **93.71%** |
```

---

## Understanding the Metrics

### Accuracy Metrics

- **Precision**: Of all vulnerabilities Triton detected, what percentage were actually real?
  - Formula: TP / (TP + FP)
  - Higher is better (fewer false positives)

- **Recall**: Of all real vulnerabilities, what percentage did Triton detect?
  - Formula: TP / (TP + FN)
  - Higher is better (fewer missed vulnerabilities)

- **F1-Score**: Harmonic mean of precision and recall
  - Formula: 2 * (Precision * Recall) / (Precision + Recall)
  - Balanced metric (higher is better)

### Performance Metrics

- **Average Analysis Time**: Time to analyze one contract (seconds)
- **Throughput**: Contracts analyzed per second
- **Total Time**: Total time for entire dataset

### Vulnerability Detection Metrics

- **True Positives (TP)**: Correctly detected vulnerabilities
- **False Positives (FP)**: Incorrectly flagged as vulnerable
- **False Negatives (FN)**: Missed vulnerabilities
- **True Negatives (TN)**: Correctly identified as safe

---

## Example Workflow

### Quick Testing (5-10 minutes)

```bash
# 1. Download SmartBugs only
python scripts/download_datasets.py --dataset smartbugs

# 2. Test on SmartBugs
python scripts/test_triton.py --dataset smartbugs

# 3. View summary
cat results/triton_test_summary_*.txt
```

### Comprehensive Testing (30-60 minutes)

```bash
# 1. Download all datasets
python scripts/download_datasets.py --dataset all

# 2. Test on all datasets
python scripts/test_triton.py --dataset all

# 3. View detailed results
cat results/triton_test_results_*.json
cat results/triton_test_summary_*.txt
```

### Testing Specific Vulnerability Types

```bash
# Test only reentrancy contracts
find data/datasets/smartbugs/dataset/reentrancy -name "*.sol" > reentrancy_list.txt
python scripts/test_triton.py --dataset custom --custom-dir data/datasets/smartbugs/dataset/reentrancy

# Test only arithmetic vulnerabilities
python scripts/test_triton.py --dataset custom --custom-dir data/datasets/smartbugs/dataset/arithmetic
```

---

## Troubleshooting

### Dataset Download Fails

If git clone fails:
```bash
# Check git is installed
git --version

# Check internet connection
ping github.com

# Try manual clone
cd data/datasets
git clone https://github.com/smartbugs/smartbugs.git
```

### Analysis Errors

If Triton fails on some contracts:
```bash
# Check the error messages in the JSON results
cat results/triton_test_results_*.json | grep -A 5 '"success": false'

# Enable verbose logging
export TRITON_LOG_LEVEL=DEBUG
python scripts/test_triton.py --dataset smartbugs
```

### Memory Issues

If you run out of memory on large datasets:
```bash
# Reduce batch size in triton_system.py
# Or test smaller subsets
python scripts/test_triton.py --dataset custom --custom-dir data/datasets/smartbugs/dataset/reentrancy
```

---

## Advanced Testing

### Compare with Baselines

To compare Triton with other tools (Slither, Mythril, etc.):

```bash
# Run baseline tools on same dataset
# Then compare results with Triton's output

# Example: Compare with Slither
for contract in data/datasets/smartbugs/dataset/*/*.sol; do
    slither "$contract" > "results/slither_$(basename $contract).txt" 2>&1
done
```

### Ablation Study

To test individual contributions:

```bash
# Test without GraphCodeBERT fine-tuning
# (Modify triton_system.py to use base GraphCodeBERT)

# Test without intelligent fusion
# (Modify fusion module to use fixed weights)

# Test without RL orchestration
# (Modify to use fixed analysis strategy)
```

### Cross-Validation

```bash
# Split dataset into train/val/test
# Test on held-out test set only
```

---

## Expected Results (from Presentation)

Based on the presentation, Triton v2.0 expects:

- **Accuracy**: 92.5% F1-score
- **Speed**: 73% faster than v1.0
- **Throughput**: 3.8× higher than v1.0
- **False Positive Rate**: 40% reduction from v1.0

Your actual results may vary depending on:
- Training status (models may still be training)
- Hardware (GPU availability)
- Dataset characteristics

---

## Next Steps After Testing

1. **Analyze Results**: Review which vulnerabilities are detected well and which are missed
2. **Error Analysis**: Deep dive into false positives and false negatives
3. **Hyperparameter Tuning**: Adjust thresholds, weights, etc.
4. **Model Training**: Complete training of GraphCodeBERT-Solidity and RL agent
5. **Benchmark Comparison**: Compare with Slither, Mythril, Securify, etc.

---

## Citation

If you use these datasets in your research, please cite:

**SmartBugs**:
```
@inproceedings{smartbugs,
  title={SmartBugs: A Framework to Analyze Solidity Smart Contracts},
  author={Durieux, Thomas and others},
  year={2020}
}
```

**SolidiFI**:
```
@inproceedings{solidifi,
  title={SolidiFI: An Automated Tool for Smart Contract Fault Injection},
  author={Nguyen, Thao and others},
  year={2021}
}
```

---

## Support

For issues or questions:
- Check existing issues in the GitHub repository
- Create a new issue with test results and error logs
- Include dataset name, contract details, and environment info
