# 🚀 START HERE - Triton Testing with SmartBugs Curated

## ✅ What You Have Now

**SmartBugs Curated Dataset** - Downloaded and ready!
- 📍 Location: `/home/anik/code/Triton/data/datasets/smartbugs-curated/`
- 📊 Total: **143 real-world vulnerable contracts**
- 🎯 Categories: **10 vulnerability types**
- ✨ Quality: Line-by-line annotated ground truth

## 🎯 Quickest Way to Start Testing (Choose One)

### Option 1: Interactive Menu (Easiest!)

```bash
cd /home/anik/code/Triton
./run_tests.sh
```

You'll see:
```
1. Download SmartBugs Curated dataset (143 contracts) ⭐ RECOMMENDED
2. Download all datasets
3. Test on reentrancy contracts (31 contracts, ~5 min)
4. Test on arithmetic contracts (15 contracts, ~3 min)
5. Test on access control (18 contracts, ~4 min)
6. Test on ALL SmartBugs Curated (143 contracts, ~30 min)
7. Test single contract
8. View latest results
9. Clean results directory
0. Exit
```

**Recommended:** Choose option 3 (reentrancy) for your first test!

### Option 2: Command Line (Fast)

```bash
cd /home/anik/code/Triton
source triton_env/bin/activate

# Test reentrancy (31 contracts, ~5 minutes)
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs-curated/dataset/reentrancy \
    --output-dir results

# View results
cat results/triton_test_summary_*.txt
```

### Option 3: Test Single Contract (Very Fast)

```bash
cd /home/anik/code/Triton
source triton_env/bin/activate

# Test the classic DAO attack
python main.py data/datasets/smartbugs-curated/dataset/reentrancy/simple_dao.sol --verbose
```

## 📊 The Dataset Breakdown

| Category | Contracts | Time to Test | Recommended Order |
|----------|-----------|--------------|-------------------|
| **Reentrancy** | 31 | ~5 min | 1st (most important) |
| **Arithmetic** | 15 | ~3 min | 2nd (common vulnerability) |
| **Access Control** | 18 | ~4 min | 3rd (important) |
| **Unchecked Calls** | 52 | ~10 min | 4th (largest category) |
| **Bad Randomness** | 8 | ~2 min | 5th |
| **DoS** | 6 | ~2 min | 6th |
| **Time Manip** | 5 | ~1 min | 7th |
| **Front Running** | 4 | ~1 min | 8th |
| **Other** | 3 | ~1 min | 9th |
| **Short Addresses** | 1 | ~10 sec | 10th |
| **ALL** | **143** | **~30 min** | Final test |

## 🎓 Recommended Testing Strategy

### Step 1: Quick Validation (5 minutes)
Test that Triton works on reentrancy:
```bash
./run_tests.sh
# Choose option 3
```

### Step 2: Core Vulnerabilities (15 minutes)
Test the top 3 categories:
```bash
./run_tests.sh
# Choose option 3 (Reentrancy)
# Then run again and choose option 4 (Arithmetic)
# Then run again and choose option 5 (Access Control)
```

### Step 3: Full Evaluation (30 minutes)
Test all 143 contracts:
```bash
./run_tests.sh
# Choose option 6
```

### Step 4: Analyze Results (10 minutes)
```bash
# View summary
cat results/triton_test_summary_*.txt

# View detailed results
python3 -m json.tool results/triton_test_results_*.json | less
```

## 📈 What to Look for in Results

### Key Metrics

Your presentation targets **92.5% F1-score**. Look for:

| Metric | Good | Excellent | Your Target |
|--------|------|-----------|-------------|
| **Precision** | > 0.80 | > 0.90 | 0.925 |
| **Recall** | > 0.80 | > 0.90 | 0.925 |
| **F1-Score** | > 0.80 | > 0.90 | **0.925** |
| **Analysis Time** | < 5s | < 2s | ~2-3s |

### Understanding Results

**Precision** = (True Positives) / (True Positives + False Positives)
- High precision = few false alarms
- "When Triton says vulnerable, it's usually right"

**Recall** = (True Positives) / (True Positives + False Negatives)
- High recall = catches most vulnerabilities
- "Triton finds most of the real vulnerabilities"

**F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
- Balanced metric
- Your target: **92.5%**

## 🔍 Example Contract You Can Test Now

The classic DAO attack (reentrancy):

```bash
cat data/datasets/smartbugs-curated/dataset/reentrancy/simple_dao.sol
```

You'll see:
```solidity
function withdraw(uint amount) {
    if (credit[msg.sender]>= amount) {
        // <yes> <report> REENTRANCY
        bool res = msg.sender.call.value(amount)();  // ⚠️ VULNERABLE LINE 19
        credit[msg.sender]-=amount;  // State updated AFTER external call
    }
}
```

Test it:
```bash
python main.py data/datasets/smartbugs-curated/dataset/reentrancy/simple_dao.sol --verbose
```

Triton should detect the reentrancy vulnerability at line 19!

## 📚 Documentation Available

| Document | Purpose |
|----------|---------|
| **START_HERE.md** (this file) | Quick start guide |
| **SMARTBUGS_CURATED_OVERVIEW.md** | Complete dataset documentation |
| **READY_TO_TEST.md** | Testing checklist |
| **QUICKSTART.md** | 5-minute tutorial |
| **TESTING_GUIDE.md** | Comprehensive testing guide |
| **DATASET_AND_TESTING_SUMMARY.md** | General overview |

## 🛠️ Troubleshooting

### Problem: Script won't run

**Solution:**
```bash
cd /home/anik/code/Triton
chmod +x run_tests.sh
source triton_env/bin/activate
./run_tests.sh
```

### Problem: Dataset not found

**Solution:**
```bash
cd /home/anik/code/Triton
./run_tests.sh
# Choose option 1 to download
```

### Problem: Python errors

**Solution:**
```bash
# Make sure environment is activated
source triton_env/bin/activate

# Check Python version
python --version  # Should be 3.8+

# Reinstall dependencies if needed
pip install -r requirements.txt
```

### Problem: Out of memory

**Solution:**
```bash
# Test smaller subsets first
./run_tests.sh
# Choose option 3 or 4 (smaller datasets)
```

## ✨ Quick Command Reference

```bash
# Navigate to project
cd /home/anik/code/Triton

# Activate environment
source triton_env/bin/activate

# Interactive testing (recommended)
./run_tests.sh

# Test reentrancy (31 contracts)
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs-curated/dataset/reentrancy

# Test all (143 contracts)
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs-curated/dataset

# Test single contract
python main.py data/datasets/smartbugs-curated/dataset/reentrancy/simple_dao.sol

# View latest results
cat results/triton_test_summary_*.txt

# View detailed results (JSON)
python3 -m json.tool results/triton_test_results_*.json | less
```

## 🎯 Your Next Action

**Right now, do this:**

```bash
cd /home/anik/code/Triton
./run_tests.sh
```

Then press **3** to test reentrancy contracts (31 contracts, ~5 minutes).

This will give you:
- ✅ Quick validation that Triton works
- ✅ Performance metrics for the most important vulnerability type
- ✅ Results you can analyze immediately
- ✅ Confidence to run larger tests

## 📊 After Your First Test

You'll get results showing:
- How many vulnerabilities Triton detected
- Precision, Recall, F1-score
- Which contracts it got right/wrong
- Analysis time per contract
- Detailed breakdown by contract

Then you can:
1. **Analyze the results** - See what worked and what didn't
2. **Test more categories** - Run arithmetic, access control, etc.
3. **Test all 143 contracts** - Full benchmark evaluation
4. **Compare with baselines** - Run Slither, Mythril for comparison

## 🚀 Ready to Start!

Everything is set up and ready. Just run:

```bash
./run_tests.sh
```

Good luck! 🎉
