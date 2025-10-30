# Triton Quick Start Guide

Get started testing Triton in 5 minutes!

## Prerequisites

Make sure you have the virtual environment activated:

```bash
source triton_env/bin/activate
```

## Option 1: Quick Test (5 minutes)

### Step 1: Download SmartBugs Dataset

```bash
python scripts/download_datasets.py --dataset smartbugs
```

This downloads 143 vulnerable contracts organized by vulnerability type.

### Step 2: Run a Simple Test

Test on a few contracts first to make sure everything works:

```bash
# Test a single contract with the main script
python main.py data/datasets/smartbugs/dataset/reentrancy/simple_dao.sol --verbose

# Or test on the reentrancy category (small set)
python scripts/test_triton.py --dataset custom --custom-dir data/datasets/smartbugs/dataset/reentrancy
```

### Step 3: View Results

```bash
# View the latest test summary
cat results/triton_test_summary_*.txt

# View detailed JSON results
cat results/triton_test_results_*.json | python -m json.tool | less
```

## Option 2: Full SmartBugs Evaluation (15-30 minutes)

### Run Full SmartBugs Test

```bash
python scripts/test_triton.py --dataset smartbugs --output-dir results
```

This will:
- Test all 143 SmartBugs contracts
- Calculate precision, recall, F1-score for each vulnerability type
- Generate comprehensive reports

### View Results

```bash
# View summary
cat results/triton_test_summary_*.txt

# View specific vulnerability results
cat results/triton_test_results_*.json | python -m json.tool | grep -A 10 "reentrancy"
```

## Option 3: Test Your Own Contracts

```bash
# Create a test contracts directory
mkdir -p test_contracts

# Copy your .sol files there
cp /path/to/your/contract.sol test_contracts/

# Run analysis
python scripts/test_triton.py --dataset custom --custom-dir test_contracts
```

## Understanding the Output

### Console Output

During analysis, you'll see:
```
2025-10-30 01:30:00 - INFO - Testing on SmartBugs dataset...
2025-10-30 01:30:05 - INFO - Testing reentrancy contracts...
2025-10-30 01:30:10 - INFO - SmartBugs testing complete: 143/143 contracts analyzed
```

### Summary Report

The summary report shows:
```
OVERALL METRICS
--------------------------------------------------------------------------------
Average Precision: 0.8542
Average Recall: 0.9123
Average F1: 0.8821
Average Analysis Time: 2.3456s
Total Contracts Tested: 143

SMARTBUGS DATASET
--------------------------------------------------------------------------------
Total Contracts: 143
Successful Analyses: 143
Total Time: 335.42s

By Vulnerability Type:
  reentrancy:
    total: 25
    detected: 23
    missed: 2
    false_positives: 1
  ...
```

### Detailed JSON Results

The JSON file contains per-contract results:
```json
{
  "metadata": {
    "timestamp": "2025-10-30T01:30:00",
    "system": "Triton v2.0"
  },
  "datasets": {
    "smartbugs": {
      "contracts": [
        {
          "contract_path": "data/datasets/smartbugs/dataset/reentrancy/simple_dao.sol",
          "analysis_time": 2.45,
          "vulnerabilities_found": ["reentrancy"],
          "confidence_scores": {
            "reentrancy": 0.95
          },
          "modality_contributions": {
            "static": 0.3,
            "dynamic": 0.6,
            "semantic": 0.1
          },
          "ground_truth": {
            "vulnerabilities": ["reentrancy"]
          },
          "metrics": {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0
          }
        }
      ]
    }
  }
}
```

## Troubleshooting

### Error: Dataset not found

```bash
# Make sure you downloaded the dataset first
python scripts/download_datasets.py --dataset smartbugs

# Check if it exists
ls data/datasets/smartbugs/dataset/
```

### Error: Module not found

```bash
# Make sure virtual environment is activated
source triton_env/bin/activate

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Error: CUDA out of memory

```bash
# Run on CPU instead (default)
python scripts/test_triton.py --dataset smartbugs

# Or reduce batch size if using GPU
# (Edit scripts/test_triton.py and change device='cpu')
```

### Analysis is slow

```bash
# Test on smaller subset first
python scripts/test_triton.py --dataset custom --custom-dir data/datasets/smartbugs/dataset/reentrancy

# Enable GPU if available (edit test_triton.py, change device='cuda')
```

## Next Steps

Once you have initial results:

1. **Analyze Performance**: Look at which vulnerability types are detected well and which need improvement

2. **Error Analysis**: Examine false positives and false negatives
   ```bash
   # Extract contracts with low F1 scores
   cat results/triton_test_results_*.json | python -c "
   import json, sys
   data = json.load(sys.stdin)
   for contract in data['datasets']['smartbugs']['contracts']:
       if contract.get('metrics', {}).get('f1', 1.0) < 0.5:
           print(contract['contract_path'])
   "
   ```

3. **Compare Baselines**: Run other tools (Slither, Mythril) on same contracts

4. **Fine-tune**: Adjust confidence thresholds, modify fusion weights, etc.

5. **Train Models**: Complete training of GraphCodeBERT-Solidity and RL agent

## Common Commands Reference

```bash
# Download datasets
python scripts/download_datasets.py --dataset smartbugs
python scripts/download_datasets.py --dataset solidifi
python scripts/download_datasets.py --dataset all

# Run tests
python scripts/test_triton.py --dataset smartbugs
python scripts/test_triton.py --dataset solidifi
python scripts/test_triton.py --dataset custom --custom-dir /path/to/contracts

# Analyze single contract
python main.py /path/to/contract.sol --verbose
python main.py /path/to/contract.sol --target-vulnerability reentrancy --output results.json

# View results
cat results/triton_test_summary_*.txt
cat results/triton_test_results_*.json | python -m json.tool
ls -lht results/  # List results by modification time
```

## Example Complete Workflow

```bash
# 1. Activate environment
source triton_env/bin/activate

# 2. Download SmartBugs
python scripts/download_datasets.py --dataset smartbugs

# 3. Quick test on reentrancy contracts (small set)
python scripts/test_triton.py --dataset custom --custom-dir data/datasets/smartbugs/dataset/reentrancy

# 4. Check results
cat results/triton_test_summary_*.txt

# 5. If good, run full test
python scripts/test_triton.py --dataset smartbugs

# 6. Analyze results
cat results/triton_test_summary_*.txt
cat results/triton_test_results_*.json | python -m json.tool > results/formatted_results.json
```

## Getting Help

- Check `TESTING_GUIDE.md` for detailed documentation
- Review `main.py` for single contract analysis options
- Check logs: `cat triton.log`
- Enable verbose mode: `python main.py contract.sol --verbose`
