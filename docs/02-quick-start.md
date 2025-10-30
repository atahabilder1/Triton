# Quick Test Commands - Triton

## 🚀 Quick Start (Copy-Paste These Commands)

### Step 1: Activate Environment
```bash
cd /home/anik/code/Triton
source triton_env/bin/activate
```

### Step 2: Run Test on SmartBugs Curated (143 contracts)
```bash
python scripts/test_triton.py --dataset smartbugs --output-dir results/smartbugs
```

### Step 3: View Results
```bash
# View text summary with vulnerability breakdown table
ls -lt results/smartbugs/triton_test_summary_*.txt | head -1 | awk '{print $NF}' | xargs cat

# View markdown table (for copying to reports)
ls -lt results/smartbugs/triton_results_table_*.md | head -1 | awk '{print $NF}' | xargs cat
```

---

## 📊 What You'll Get

### 1. Console Output
Immediately see the results table in your terminal:
```
Vulnerability Type              | Total    | Detected   | Missed   | Detection %
----------------------------------------------------------------------------------------------------
Access Control                  | 18       | 17         | 1        |      94.44%
Arithmetic                      | 15       | 14         | 1        |      93.33%
Reentrancy                      | 31       | 30         | 1        |      96.77%
...
TOTAL                           | 143      | 134        | 9        |      93.71%
```

### 2. Three Output Files
- `triton_test_results_TIMESTAMP.json` - Detailed JSON results
- `triton_test_summary_TIMESTAMP.txt` - Text summary with table
- `triton_results_table_TIMESTAMP.md` - Markdown table for reports

---

## 🎯 Test Specific Vulnerability Types

### Test Only Reentrancy (31 contracts)
```bash
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs-curated/dataset/reentrancy \
    --output-dir results/reentrancy
```

### Test Only Access Control (18 contracts)
```bash
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs-curated/dataset/access_control \
    --output-dir results/access_control
```

### Test Only Arithmetic (15 contracts)
```bash
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs-curated/dataset/arithmetic \
    --output-dir results/arithmetic
```

---

## 🔬 Test on FORGE Dataset (81,390 files)

### Full FORGE Dataset (WARNING: Will take hours/days)
```bash
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/FORGE-Artifacts/dataset/contracts \
    --output-dir results/forge_full
```

### Sample of FORGE (100 random contracts)
```bash
# Create sample directory
mkdir -p data/datasets/FORGE-sample

# Get 100 random contracts
find data/datasets/FORGE-Artifacts/dataset/contracts -name "*.sol" | shuf | head -100 | \
    xargs -I {} cp {} data/datasets/FORGE-sample/

# Test on sample
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/FORGE-sample \
    --output-dir results/forge_sample
```

---

## 📋 Interactive Menu

Use the interactive menu for easier testing:
```bash
./run_tests.sh
```

Then select:
- Option 1: Test reentrancy (31 contracts)
- Option 2: Test arithmetic (15 contracts)
- Option 3: Test access control (18 contracts)
- Option 4: Test all vulnerabilities (143 contracts)

---

## 🐛 Troubleshooting

### Missing Dependencies
```bash
source triton_env/bin/activate
pip install torch-geometric mythril transformers pandas numpy
```

### Dataset Not Found
```bash
# Check if SmartBugs Curated is downloaded
ls data/datasets/smartbugs-curated/dataset/

# If not, download it
cd data/datasets
git clone https://github.com/smartbugs/smartbugs-curated.git
cd ../..
```

### Models Not Trained
If you get errors about missing model weights, that's expected if models aren't trained yet. The system will still run with default weights.

---

## ✅ Expected Output Format

When you run the test, you'll see output like this:

```
2025-10-30 15:30:00 - INFO - Testing on SmartBugs Curated dataset...
2025-10-30 15:30:05 - INFO - Testing reentrancy contracts...
2025-10-30 15:35:10 - INFO - Testing arithmetic contracts...
...
2025-10-30 16:00:00 - INFO - SmartBugs testing complete: 143/143 contracts analyzed

====================================================================================================
TRITON VULNERABILITY DETECTION - SUMMARY
====================================================================================================

Overall Metrics:
  Average Precision: 0.9250
  Average Recall: 0.9100
  Average F1: 0.9175
  Average Analysis Time: 2.3500
  Total Contracts Tested: 143

SMARTBUGS Dataset - Vulnerability Detection:
----------------------------------------------------------------------------------------------------
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

Results saved to:
  - Detailed JSON: results/smartbugs/triton_test_results_20251030_153000.json
  - Text Summary: results/smartbugs/triton_test_summary_20251030_153000.txt
  - Markdown Table: results/smartbugs/triton_results_table_20251030_153000.md
====================================================================================================
```

---

## 📊 Understanding the Table

| Column | Meaning |
|--------|---------|
| **Vulnerability Type** | Type of vulnerability (reentrancy, arithmetic, etc.) |
| **Total** | Total number of contracts with this vulnerability in the dataset |
| **Detected** | Number of vulnerabilities successfully detected by Triton |
| **Missed** | Number of vulnerabilities that Triton failed to detect |
| **Detection %** | Percentage of vulnerabilities detected (Detected/Total × 100) |

**Goal**: Detection % should be as close to 100% as possible for each category.

---

## 🎓 For Your Professor

Show your professor the markdown table file:
```bash
cat results/smartbugs/triton_results_table_*.md
```

This will give them a clean, professional table showing:
- How many of each vulnerability type were tested
- How many Triton detected correctly
- Detection percentage for each category
- Overall detection rate across all 143 contracts

They can see exact numbers like:
- "Triton detected 30 out of 31 reentrancy vulnerabilities (96.77%)"
- "Triton detected 17 out of 18 access control issues (94.44%)"
- "Overall detection rate: 93.71% (134 out of 143 vulnerabilities)"

---

**Note**: The example numbers shown above (93.71% overall, etc.) are just examples. Your actual results will depend on whether your models are fully trained and the specific implementation details.
