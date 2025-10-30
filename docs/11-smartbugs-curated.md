# SmartBugs Curated Dataset - Complete Overview

## 🎯 The Better Dataset for Testing!

**SmartBugs Curated** is the **largest curated dataset** of vulnerable Solidity smart contracts, developed for automated reasoning and testing research.

## 📍 Location

```
/home/anik/code/Triton/data/datasets/smartbugs-curated/
```

## 📊 Dataset Statistics

| Total Contracts | Vulnerability Types | Real-World Contracts | Ground Truth Labels |
|----------------|--------------------|--------------------|-------------------|
| **143** | **10** | ✅ Yes (from Etherscan) | ✅ Yes (annotated) |

## 📂 Dataset Structure

```
smartbugs-curated/
├── dataset/
│   ├── access_control/           # 18 contracts
│   ├── arithmetic/                # 15 contracts
│   ├── bad_randomness/            # 8 contracts
│   ├── denial_of_service/         # 6 contracts
│   ├── front_running/             # 4 contracts
│   ├── other/                     # 3 contracts
│   ├── reentrancy/                # 31 contracts ⭐ Largest category
│   ├── short_addresses/           # 1 contract
│   ├── time_manipulation/         # 5 contracts
│   └── unchecked_low_level_calls/ # 52 contracts ⭐ Second largest
│
├── vulnerabilities.json           # Complete metadata with line numbers
├── versions.csv                   # Solidity version info
└── README.md                      # Dataset documentation
```

## 🔥 Vulnerability Categories (by Count)

| Rank | Category | Count | % of Total | Description |
|------|----------|-------|-----------|-------------|
| 1 | **Unchecked Low Level Calls** | 52 | 36.4% | call(), send(), delegatecall() without checking return |
| 2 | **Reentrancy** | 31 | 21.7% | External calls allow reentering function |
| 3 | **Access Control** | 18 | 12.6% | Missing modifiers, tx.origin issues |
| 4 | **Arithmetic** | 15 | 10.5% | Integer overflow/underflow |
| 5 | **Bad Randomness** | 8 | 5.6% | Predictable randomness using block data |
| 6 | **Denial of Service** | 6 | 4.2% | Gas limit, unexpected revert |
| 7 | **Time Manipulation** | 5 | 3.5% | Timestamp dependence |
| 8 | **Front Running** | 4 | 2.8% | Transaction ordering dependence |
| 9 | **Other** | 3 | 2.1% | Miscellaneous vulnerabilities |
| 10 | **Short Addresses** | 1 | 0.7% | EVM padding issues |

## 📝 Contract Annotation Format

All contracts are annotated with:
- **Source** (`@source`) - Where the contract came from
- **Author** (`@author`) - Contract author (if known)
- **Vulnerable lines** (`@vulnerable_at_lines`) - Exact line numbers
- **Inline comments** - `// <yes> <report> CATEGORY` at vulnerable lines

### Example: simple_dao.sol (Reentrancy)

```solidity
/*
 * @source: http://blockchain.unica.it/projects/ethereum-survey/attacks.html#simpledao
 * @author: -
 * @vulnerable_at_lines: 19
 */

pragma solidity ^0.4.2;

contract SimpleDAO {
  mapping (address => uint) public credit;

  function donate(address to) payable {
    credit[to] += msg.value;
  }

  function withdraw(uint amount) {
    if (credit[msg.sender]>= amount) {
      // <yes> <report> REENTRANCY
      bool res = msg.sender.call.value(amount)();  // ⚠️ Line 19: Vulnerable!
      credit[msg.sender]-=amount;  // State updated AFTER external call
    }
  }
}
```

## 🎯 Why This Dataset is Better

| Feature | SmartBugs Curated | Previous Dataset |
|---------|------------------|------------------|
| **Size** | 143 contracts | 50 contracts |
| **Unique contracts** | 143 unique | 10 unique × 5 versions |
| **Organization** | By vulnerability type | By Solidity version |
| **Annotations** | Line-by-line annotations | Basic metadata |
| **Real-world** | Yes (Etherscan) | Yes |
| **Variety** | High diversity | Limited variety |
| **Ground truth** | Detailed (line numbers) | Basic |

## 🚀 Quick Start Testing

### Test Specific Vulnerability Types

```bash
# Test all reentrancy contracts (31 contracts)
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs-curated/dataset/reentrancy \
    --output-dir results

# Test arithmetic vulnerabilities (15 contracts)
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs-curated/dataset/arithmetic \
    --output-dir results

# Test access control (18 contracts)
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs-curated/dataset/access_control \
    --output-dir results
```

### Test All 143 Contracts

```bash
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs-curated/dataset \
    --output-dir results
```

### Test Single Contract

```bash
# Classic DAO attack
python main.py data/datasets/smartbugs-curated/dataset/reentrancy/simple_dao.sol --verbose

# Integer overflow
python main.py data/datasets/smartbugs-curated/dataset/arithmetic/BECToken.sol --verbose

# Access control issue
python main.py data/datasets/smartbugs-curated/dataset/access_control/rubixi.sol --verbose
```

## 📊 Sample Contracts by Category

### 1. Reentrancy (31 contracts)
- `simple_dao.sol` - Classic DAO attack
- `DAO.sol` - The famous DAO hack
- `etherpot_lotto.sol` - Lottery reentrancy
- `0x01f8c4e3fa3edeb29e514cba738d87ce8c091d3f.sol` - Real mainnet contract
- ... 27 more

### 2. Unchecked Low Level Calls (52 contracts)
- `unhandled_exception.sol` - Unchecked send()
- `lost_in_transfer.sol` - Unchecked call()
- `0x23a91059fdc9579a9fbd0edc5f2ea0bfdb70deb4.sol` - Real mainnet contract
- ... 49 more

### 3. Access Control (18 contracts)
- `rubixi.sol` - Wrong constructor name
- `tokensalechallenge.sol` - Missing access control
- `mycontract.sol` - tx.origin authentication
- ... 15 more

### 4. Arithmetic (15 contracts)
- `BECToken.sol` - Integer overflow in transfer
- `integer_overflow_mul.sol` - Multiplication overflow
- `integer_overflow_add.sol` - Addition overflow
- ... 12 more

### 5. Other Categories
- **Bad Randomness** (8): `crypto_roulette.sol`, `smart_billions.sol`, etc.
- **Denial of Service** (6): `governmental_survey.sol`, etc.
- **Time Manipulation** (5): `timed_crowdsale.sol`, etc.
- **Front Running** (4): `ERC20.sol`, etc.

## 🔍 Exploring the Dataset

### View Vulnerability Metadata

```bash
# View all vulnerability annotations
python3 -m json.tool data/datasets/smartbugs-curated/vulnerabilities.json | less

# Search for specific contract
python3 -m json.tool data/datasets/smartbugs-curated/vulnerabilities.json | grep -A 10 "simple_dao"

# Count vulnerabilities per category
python3 -m json.tool data/datasets/smartbugs-curated/vulnerabilities.json | grep '"category"' | sort | uniq -c
```

### List All Contracts

```bash
# List all reentrancy contracts
ls data/datasets/smartbugs-curated/dataset/reentrancy/

# Find all contracts across all categories
find data/datasets/smartbugs-curated/dataset -name "*.sol" | sort

# Count contracts per category
for dir in data/datasets/smartbugs-curated/dataset/*/; do
    echo "$(basename $dir): $(find $dir -name '*.sol' | wc -l)"
done
```

### View Contract Source

```bash
# View a reentrancy example
cat data/datasets/smartbugs-curated/dataset/reentrancy/simple_dao.sol

# View an overflow example
cat data/datasets/smartbugs-curated/dataset/arithmetic/BECToken.sol

# View with line numbers (to see vulnerable lines)
cat -n data/datasets/smartbugs-curated/dataset/reentrancy/simple_dao.sol
```

## 🧪 Recommended Testing Strategy

### Phase 1: Small-Scale Testing (30 minutes)

Test individual vulnerability categories to see how Triton performs:

```bash
# 1. Test reentrancy (31 contracts)
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs-curated/dataset/reentrancy

# 2. Test arithmetic (15 contracts)
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs-curated/dataset/arithmetic

# 3. Test access control (18 contracts)
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs-curated/dataset/access_control
```

### Phase 2: Full Dataset Testing (1-2 hours)

```bash
# Test all 143 contracts
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs-curated/dataset \
    --output-dir results

# View results
cat results/triton_test_summary_*.txt
```

### Phase 3: Error Analysis

```bash
# Analyze which vulnerabilities are detected well
python3 -m json.tool results/triton_test_results_*.json | grep -A 5 "vulnerability_type"

# Find false negatives (missed vulnerabilities)
python3 -m json.tool results/triton_test_results_*.json | grep -B 5 '"missed"'

# Find false positives
python3 -m json.tool results/triton_test_results_*.json | grep -B 5 '"false_positives"'
```

## 📈 Expected Performance Metrics

Based on similar research papers, good vulnerability detection tools should achieve:

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| **Precision** | > 0.70 | > 0.80 | > 0.90 |
| **Recall** | > 0.70 | > 0.80 | > 0.90 |
| **F1-Score** | > 0.70 | > 0.80 | > 0.90 |
| **False Positive Rate** | < 0.30 | < 0.20 | < 0.10 |
| **Analysis Time** | < 10s/contract | < 5s/contract | < 2s/contract |

Your presentation targets: **92.5% F1-score**

## 🔧 Integration with Test Script

The test script automatically works with SmartBugs Curated:

```python
# Test on SmartBugs Curated reentrancy category
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs-curated/dataset/reentrancy
```

The script will:
1. Load all contracts from the directory
2. Run Triton analysis on each
3. Compare with ground truth from `vulnerabilities.json`
4. Calculate metrics (precision, recall, F1)
5. Generate detailed reports

## 📚 References & Citations

**SmartBugs Framework:**
```
@inproceedings{smartbugs,
  title={SmartBugs: A Framework to Analyze Solidity Smart Contracts},
  author={Durieux, Thomas and Ferreira, Joao F. and Abreu, Rui and Cruz, Pedro},
  booktitle={2020 35th IEEE/ACM International Conference on Automated Software Engineering (ASE)},
  pages={1349--1352},
  year={2020}
}
```

**ICSE 2020 Study (used this dataset):**
```
@inproceedings{durieux2020empirical,
  title={Empirical review of automated analysis tools on 47,587 ethereum smart contracts},
  author={Durieux, Thomas and others},
  booktitle={ICSE 2020},
  year={2020}
}
```

## 🎉 Ready to Test!

You now have:
- ✅ **143 real-world vulnerable contracts**
- ✅ **10 vulnerability categories**
- ✅ **Detailed ground truth with line numbers**
- ✅ **Well-organized by vulnerability type**
- ✅ **Testing scripts ready to use**

### Start Testing Now:

```bash
cd /home/anik/code/Triton
source triton_env/bin/activate

# Quick test on reentrancy (31 contracts, ~5 minutes)
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs-curated/dataset/reentrancy

# Full test (143 contracts, ~30 minutes)
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs-curated/dataset
```

**Good luck with your testing!** 🚀
