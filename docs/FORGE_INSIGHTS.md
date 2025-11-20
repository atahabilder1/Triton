# FORGE Dataset Key Insights

## Analysis Summary

**Date**: November 19, 2025
**Dataset**: FORGE-Artifacts (6,454 audit reports)
**Total Findings**: 27,497 vulnerability findings
**Analysis Script**: `scripts/dataset/analyze_forge_audits.py`

---

## 🔍 Key Findings

### 1. CWE Code Distribution

**Only 18 unique CWE codes** appear in 27,434 CWE occurrences across all audits!

This is very different from the 303 CWEs mapped in `prepare_forge_dataset_accurate.py`.

**Top 10 Most Common CWEs:**

| Rank | CWE Code | Count | Percentage | Meaning |
|------|----------|-------|------------|---------|
| 1 | **CWE-710** | 8,880 | 32.37% | Improper Coding Practices |
| 2 | **CWE-284** | 6,121 | 22.31% | Improper Access Control |
| 3 | **CWE-703** | 3,762 | 13.71% | Improper Error Handling |
| 4 | **CWE-682** | 3,233 | 11.78% | Incorrect Calculation |
| 5 | **CWE-664** | 1,755 | 6.40% | Improper Control of Resources |
| 6 | **CWE-691** | 1,431 | 5.22% | Insufficient Control Flow |
| 7 | **CWE-435** | 625 | 2.28% | Improper Interaction Between Entities |
| 8 | **CWE-693** | 624 | 2.27% | Protection Mechanism Failure |
| 9 | **CWE-697** | 395 | 1.44% | Incorrect Comparison |
| 10 | **CWE-707** | 247 | 0.90% | Improper Neutralization |

**⚠️ Important Observation:**
- The top 4 CWEs account for **80% of all vulnerabilities**
- CWE-710 (Coding Practices) is the most common - often considered "code quality" not security
- Only 18 CWEs vs 303 in the mapping suggests many mapped CWEs don't appear in the data

### 2. Severity Distribution

**Most vulnerabilities are LOW severity!**

| Severity | Count | Percentage |
|----------|-------|------------|
| **Low** | 15,170 | 55.17% |
| Info | 3,456 | 12.57% |
| Medium | 3,367 | 12.24% |
| N/A | 2,406 | 8.75% |
| High | 2,094 | 7.62% |
| **Critical** | 1,004 | 3.65% |

**Key Insights:**
- Only **3.65% are critical** vulnerabilities
- **67.74% are low severity or informational**
- If training on all FORGE data, model may learn mostly "code quality" issues, not security exploits
- Consider filtering by severity for security-focused training

### 3. Multi-Label Reality

**56.44% of contracts have MULTIPLE vulnerabilities!**

| Number of CWEs | Contracts | Percentage |
|----------------|-----------|------------|
| 0 CWEs | 1,149 | 17.80% |
| **1 CWE** | 1,663 | 25.77% |
| **2 CWEs** | 1,374 | 21.29% |
| **3 CWEs** | 829 | 12.84% |
| **4 CWEs** | 562 | 8.71% |
| 5+ CWEs | 877 | 13.59% |

**Examples of contracts with 10+ CWEs:**
- SmartContract_Audit_Solidproof_Jamonswap: **12 CWEs**
- cryptex-finance: **12 CWEs**
- AstraDAO Smart Contract: **11 CWEs**

**⚠️ Critical Implication:**
Your current `forge_balanced_accurate` dataset uses **single-label classification** - each contract is assigned only ONE vulnerability class (the highest priority). This means:
- 56% of contracts lose information about other vulnerabilities
- Model won't learn that contracts can have multiple issues
- Real-world use will miss additional vulnerabilities

**Recommendation**: Consider multi-label classification!

### 4. CWE Co-Occurrence Patterns

**Which vulnerabilities appear together?**

Top 15 CWE pairs that frequently co-occur:

| CWE Pair | Co-occurrence | Insight |
|----------|---------------|---------|
| CWE-284 + CWE-710 | 1,474 | Access control with poor coding |
| CWE-284 + CWE-703 | 1,410 | Access control with error handling |
| CWE-682 + CWE-710 | 1,308 | Calculation errors with poor coding |
| CWE-703 + CWE-710 | 1,264 | Error handling with poor coding |
| CWE-284 + CWE-682 | 1,071 | Access control with calculation errors |
| CWE-682 + CWE-703 | 1,018 | Calculation errors with error handling |

**Pattern**: **CWE-710 (Coding Practices)** appears with almost everything, confirming it's a general "code quality" issue rather than a specific vulnerability.

### 5. Compiler Version Insights

**Solidity 0.8.x dominates, but has more vulnerabilities!**

| Version | Contracts | Avg Vulnerabilities per Contract |
|---------|-----------|----------------------------------|
| **0.8.x** | 3,268 (50.6%) | **3.70** |
| n/a | 1,502 (23.3%) | **7.31** (unknown) |
| **0.6.x** | 1,025 (15.9%) | **2.42** |
| 0.7.x | 321 (5.0%) | 3.12 |
| 0.5.x | 251 (3.9%) | 2.24 |
| 0.4.x | 110 (1.7%) | 3.85 |

**Insights:**
- Solidity 0.8.x is most common (50.6%)
- But 0.8.x has **higher average vulnerabilities** (3.70) than 0.6.x (2.42)
- This might be because 0.8.x is newer → more complex projects → more audit findings
- Or because auditors are more thorough with newer code
- 23.3% have no version info (n/a)

### 6. Blockchain Distribution

**BSC (Binance Smart Chain) dominates!**

| Chain | Contracts | Percentage |
|-------|-----------|------------|
| **BSC** | 4,095 | 63.45% |
| n/a | 1,501 | 23.26% |
| **Ethereum** | 742 | 11.50% |
| Polygon | 116 | 1.80% |

**Insights:**
- BSC contracts are **63.45%** of the dataset
- Ethereum is only **11.50%**
- If you're targeting Ethereum, this dataset is **heavily biased toward BSC**
- BSC and Ethereum may have different vulnerability patterns

---

## 🎯 Implications for Your Training

### 1. Dataset Quality Concerns

**Your `forge_balanced_accurate` dataset may have issues:**

❌ **Problem 1: CWE Mapping Mismatch**
- Script maps 303 CWEs → 10 classes
- But actual data only has 18 CWEs
- Many mapped CWEs never appear in the dataset

❌ **Problem 2: Low-Severity Bias**
- 55% of findings are "low" severity
- Only 3.65% are "critical"
- Model may learn code quality issues, not exploitable vulnerabilities

❌ **Problem 3: Single-Label Limitation**
- 56% of contracts have multiple vulnerabilities
- Single-label approach loses this information
- Model won't learn multi-vulnerability patterns

❌ **Problem 4: BSC Bias**
- 63% of contracts are BSC
- Only 11% are Ethereum
- Model may not generalize to Ethereum well

❌ **Problem 5: CWE-710 Dominance**
- 32% of all CWEs are CWE-710 (Coding Practices)
- This is not really a "vulnerability" - it's code quality
- May dilute security-focused training

### 2. Recommendations for Better Training

#### Option 1: Filter by Severity
```bash
# Only use HIGH and CRITICAL vulnerabilities
python scripts/dataset/prepare_forge_dataset_filtered.py \
    --forge-dir data/datasets/FORGE-Artifacts \
    --output-dir data/datasets/forge_high_severity \
    --min-severity high
```

#### Option 2: Multi-Label Classification
```python
# Instead of single label per contract, predict all CWEs
# Change architecture to multi-label:
#   - Output: sigmoid activation + BCE loss
#   - Each CWE is a binary classification
#   - Can detect multiple vulnerabilities per contract
```

#### Option 3: Exclude CWE-710
```python
# Filter out "coding practices" issues
# Focus on actual security vulnerabilities
exclude_cwes = ['CWE-710', 'CWE-703']  # Coding + Error handling
```

#### Option 4: Balance by Blockchain
```bash
# Create balanced dataset across chains
python scripts/dataset/prepare_forge_balanced_chains.py \
    --target-per-chain 1000
```

#### Option 5: Focus on Top CWEs Only
```python
# Only use the 10 most common, security-relevant CWEs
top_cwes = [
    'CWE-284',  # Access Control
    'CWE-682',  # Incorrect Calculation (arithmetic)
    'CWE-664',  # Resource Control
    'CWE-691',  # Control Flow
    'CWE-435',  # Interaction (reentrancy)
    'CWE-693',  # Protection Mechanism
    'CWE-697',  # Incorrect Comparison
    'CWE-707',  # Improper Neutralization
    'CWE-20',   # Input Validation
]
```

---

## 📊 What This Means for Your Current Dataset

### `forge_balanced_accurate` Issues

Looking at your current dataset:
```json
{
  "total_contracts": 7013,
  "splits": {
    "train": {
      "safe": 700,
      "access_control": 700,
      "reentrancy": 560,
      ...
    }
  }
}
```

**Based on the FORGE analysis:**

1. **"Safe" contracts (700)** might not be truly safe:
   - They might just have low-severity findings
   - Or only CWE-710 (coding practices)
   - Need to verify: are these 700 truly vulnerability-free?

2. **Access Control (700)** is good:
   - CWE-284 is 22.31% of all findings
   - Well-represented in actual data

3. **Reentrancy (560)** may be mapped from:
   - CWE-435 (Interaction): 625 occurrences (2.28%)
   - This is relatively rare in the dataset
   - 560 samples might be all available reentrancy cases

4. **Arithmetic (700)** is good:
   - CWE-682 is 11.78% of findings
   - Well-represented

5. **Under-represented classes** (bad_randomness: 160, short_addresses: 43):
   - Likely don't exist in FORGE at all
   - Or mapped from very rare CWEs
   - Consider removing these classes

---

## 🚀 Recommended Next Steps

### 1. Verify Your Current Dataset
```bash
# Check if your labels match FORGE audit severity
python scripts/dataset/verify_forge_labels.py \
    --dataset data/datasets/forge_balanced_accurate \
    --forge-audits data/datasets/FORGE-Artifacts/dataset/results
```

### 2. Create a High-Quality Subset
```bash
# Focus on high-severity, well-represented vulnerabilities
python scripts/dataset/prepare_forge_high_quality.py \
    --min-severity medium \
    --exclude-cwes CWE-710,CWE-703 \
    --min-samples-per-class 300 \
    --output data/datasets/forge_high_quality
```

### 3. Try Multi-Label Classification
```python
# Update your model to predict multiple vulnerabilities
# See: docs/MULTI_LABEL_GUIDE.md (to be created)
```

### 4. Compare Against `combined_labeled`
```bash
# Your 228-contract curated dataset might be higher quality
# Use it for initial testing before moving to FORGE
./start_training.sh static --train-dir data/datasets/combined_labeled/train
```

### 5. Consider Hybrid Approach
```python
# Combine high-quality curated + filtered FORGE
# Best of both worlds: quality + scale
python scripts/dataset/create_hybrid_dataset.py \
    --curated data/datasets/combined_labeled \
    --forge data/datasets/forge_high_quality \
    --output data/datasets/hybrid_balanced
```

---

## 📝 Summary

**What you discovered:**
1. FORGE has **only 18 CWEs**, not 303
2. **32% of findings** are CWE-710 (code quality, not security)
3. **56% of contracts** have multiple vulnerabilities (multi-label problem)
4. **55% of findings** are low severity
5. **63% of contracts** are BSC, only 11% Ethereum

**What this means:**
- Your `forge_balanced_accurate` dataset may be lower quality than expected
- Many "vulnerabilities" are just code quality issues
- Single-label approach loses information
- Dataset is biased toward BSC

**What to do:**
1. Filter FORGE by severity (medium+)
2. Exclude CWE-710 (coding practices)
3. Consider multi-label classification
4. Start with `combined_labeled` (228 high-quality contracts)
5. Create hybrid dataset: curated + filtered FORGE

**Run the analysis yourself:**
```bash
# Generate full FORGE insights
python3 scripts/dataset/analyze_forge_audits.py \
    --forge-dir data/datasets/FORGE-Artifacts \
    --output forge_audit_analysis.json

# View the report
cat forge_audit_analysis.json | python3 -m json.tool | less
```

---

**Files Created:**
- `/home/anik/code/Triton/scripts/dataset/analyze_forge_audits.py` - Analysis script
- `/home/anik/code/Triton/forge_audit_analysis.json` - JSON report
- `/home/anik/code/Triton/docs/FORGE_INSIGHTS.md` - This file
- `/home/anik/code/Triton/docs/DATASET_COMPARISON.md` - Comparison guide
