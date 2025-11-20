# Dataset Comparison Guide

## Overview

You have **3 different datasets** in your project, each created with different methods and purposes. Here's what each one is and how they differ:

---

## 1. FORGE-Artifacts (Original Dataset)

**Location**: `/home/anik/code/Triton/data/datasets/FORGE-Artifacts/`

### What It Is
- **Source**: Downloaded FORGE (Formal Audit Reports as Guidance for Exploitation) dataset
- **Original research dataset** from real-world security audits
- **Contains**: 6,618 smart contracts with their audit reports
- **Format**: Raw audit reports (PDF → JSON) + original contract files

### Structure
```
FORGE-Artifacts/
├── dataset/
│   ├── contracts/          # 6,618 smart contract folders
│   │   ├── $PePe coin Full Smart Contract Security Audit/
│   │   ├── $TMF-Audit-by-BlockSAFU/
│   │   └── ... (6,616 more projects)
│   └── results/            # Audit reports in JSON format
│       ├── $PePe coin Full Smart Contract Security Audit.pdf.json
│       └── ... (6,618 audit JSONs)
└── README.md
```

### What's Inside the Audit JSONs

Each audit report JSON contains:
```json
{
  "path": "artifacts/$PePe coin Full Smart Contract Security Audit.pdf",
  "project_info": {
    "address": "0xff593cb838547700c565024c42ce9a2a24511b01",
    "chain": "bsc",
    "compiler_version": ["v0.8.4+commit.c7e474f2"],
    "project_path": {
      "$PePe": "contracts/$PePe coin Full Smart Contract Security Audit/$PePe"
    }
  },
  "findings": [
    {
      "id": 0,
      "category": {
        "1": ["CWE-703"],
        "2": ["CWE-755"]
      },
      "title": "Out of gas exception",
      "description": "Functions use loops that can cause OUT_OF_GAS...",
      "severity": "low",
      "location": "Functions `includeInReward` and `_getCurrentSupply`"
    }
  ]
}
```

### Key Insights Available
✅ **CWE codes** for each vulnerability
✅ **Severity levels** (critical, high, medium, low)
✅ **Exact locations** of vulnerabilities
✅ **Real-world audit descriptions**
✅ **Multiple CWEs per contract** (more granular than single label)
✅ **Blockchain metadata** (address, chain, compiler version)

### Challenges
❌ Contracts have **interfaces** and **abstract contracts**
❌ **Missing dependencies** (imports not resolved)
❌ **No direct train/val/test split**
❌ **Needs preprocessing** to extract usable labels

---

## 2. combined_labeled (Small Curated Dataset)

**Location**: `/home/anik/code/Triton/data/datasets/combined_labeled/`

**Created By**: `scripts/dataset/combine_labeled_datasets.py`

### What It Is
- **Combines 4 well-known datasets**: SmartBugs Curated, SmartBugs Samples, SolidiFI, Not So Smart Contracts
- **Total**: 228 labeled contracts
- **Purpose**: Small, high-quality dataset for quick testing
- **Method**: Manual curation and validation by researchers

### Structure
```
combined_labeled/
├── access_control/              (29 contracts)
├── arithmetic/                  (17 contracts)
├── bad_randomness/              (10 contracts)
├── denial_of_service/           (9 contracts)
├── front_running/               (6 contracts)
├── reentrancy/                  (54 contracts)
├── short_addresses/             (2 contracts)
├── time_manipulation/           (11 contracts)
├── unchecked_low_level_calls/   (30 contracts)
├── safe/                        (60 contracts)
├── dataset_summary.json
└── train_val_test_splits.json
```

### Characteristics
| Class | Count | Percentage |
|-------|-------|------------|
| Total | 228 | 100% |
| Reentrancy | 54 | 23.7% |
| Safe | 60 | 26.3% |
| Access Control | 29 | 12.7% |
| Unchecked Calls | 30 | 13.2% |
| Arithmetic | 17 | 7.5% |

### Pros
✅ **High quality** - manually curated and validated
✅ **Clean compilation** - most contracts compile without issues
✅ **Well-documented** - known vulnerabilities
✅ **Good for testing** - quick validation of your model

### Cons
❌ **Small size** - only 228 contracts
❌ **Imbalanced** - some classes have very few examples
❌ **Limited diversity** - academic examples, not real-world
❌ **Not enough for training** - too small for deep learning

---

## 3. forge_balanced_accurate (Large Training Dataset)

**Location**: `/home/anik/code/Triton/data/datasets/forge_balanced_accurate/`

**Created By**: `scripts/dataset/prepare_forge_dataset_accurate.py`

### What It Is
- **Created from FORGE-Artifacts** using CWE → vulnerability mapping
- **Total**: 7,013 labeled contracts
- **Purpose**: Large, balanced dataset for training deep learning models
- **Method**: Automatic CWE code mapping to 11 vulnerability classes

### Structure
```
forge_balanced_accurate/
├── train/                  (70% - 4,909 contracts)
│   ├── access_control/           (700 contracts)
│   ├── arithmetic/               (700 contracts)
│   ├── reentrancy/               (560 contracts)
│   ├── safe/                     (700 contracts)
│   └── ... (11 classes total)
├── val/                    (15% - 1,051 contracts)
│   └── ... (same structure)
├── test/                   (15% - 1,053 contracts)
│   └── ... (same structure)
└── dataset_summary.json
```

### Class Distribution

| Class | Train | Val | Test | Total | Status |
|-------|-------|-----|------|-------|--------|
| **Safe** | 700 | 150 | 150 | 1,000 | ✅ Well-represented |
| **Access Control** | 700 | 150 | 150 | 1,000 | ✅ Well-represented |
| **Arithmetic** | 700 | 150 | 150 | 1,000 | ✅ Well-represented |
| **Unchecked Calls** | 700 | 150 | 150 | 1,000 | ✅ Well-represented |
| **Reentrancy** | 560 | 120 | 120 | 800 | ✅ Well-represented |
| **Other** | 700 | 150 | 150 | 1,000 | ✅ Well-represented |
| **Denial of Service** | 350 | 75 | 75 | 500 | ⚠️ Moderate |
| **Time Manipulation** | 210 | 45 | 45 | 300 | ⚠️ Moderate |
| **Front Running** | 147 | 31 | 32 | 210 | ⚠️ Under-represented |
| **Bad Randomness** | 112 | 24 | 24 | 160 | ❌ Under-represented |
| **Short Addresses** | 30 | 6 | 7 | 43 | ❌ Very rare |

### How Labels Were Created

The script maps **303 CWE codes** from FORGE audit reports to **10 vulnerability classes**:

```python
# Example CWE → Class Mapping
CWE_TO_CLASS = {
    # Access Control (50+ CWEs)
    'CWE-284': 'access_control',  # Improper Access Control
    'CWE-269': 'access_control',  # Improper Privilege Management

    # Arithmetic (9 CWEs)
    'CWE-682': 'arithmetic',      # Incorrect Calculation
    'CWE-190': 'arithmetic',      # Integer Overflow
    'CWE-191': 'arithmetic',      # Integer Underflow

    # Reentrancy (11 CWEs)
    'CWE-362': 'reentrancy',      # Race Condition
    'CWE-1265': 'reentrancy',     # Reentrant Call

    # ... 300+ more CWEs mapped
}
```

**Priority System**: If a contract has multiple CWEs, it's classified by the **most critical** vulnerability:

1. Reentrancy (highest priority - most dangerous)
2. Arithmetic
3. Bad randomness
4. Time manipulation
5. Short addresses
6. Front running
7. Denial of service
8. Unchecked calls
9. Access control
10. Other (lowest priority)

### Pros
✅ **Large scale** - 7,013 contracts for deep learning
✅ **Balanced** - most classes have 500-1,000 samples
✅ **Real-world** - from actual production contracts
✅ **Train/val/test ready** - pre-split 70/15/15
✅ **CWE-based** - follows security standards

### Cons
❌ **Contains interfaces** - some abstract contracts
❌ **Missing dependencies** - imports not resolved
❌ **Automatic labeling** - CWE mapping may not be perfect
❌ **Some rare classes** - bad_randomness (160), short_addresses (43)
❌ **Needs preprocessing** - flattening and validation required

---

## Key Differences Summary

| Feature | FORGE-Artifacts | combined_labeled | forge_balanced_accurate |
|---------|----------------|------------------|------------------------|
| **Source** | Original FORGE dataset | 4 curated datasets | FORGE + CWE mapping |
| **Size** | 6,618 contracts | 228 contracts | 7,013 contracts |
| **Labels** | CWE codes in JSON | Manual classification | Auto CWE → class |
| **Quality** | Raw (needs preprocessing) | High (curated) | Medium (automatic) |
| **Purpose** | Research/insights | Quick testing | Large-scale training |
| **Train/val/test** | ❌ No | ✅ Yes | ✅ Yes |
| **Dependencies** | ❌ Many missing | ✅ Mostly resolved | ❌ Many missing |
| **Best for** | Understanding vulnerabilities | Validation | Training models |
| **Ready to train** | ❌ No | ✅ Yes | ⚠️ Needs preprocessing |

---

## What Insights Can We Get from FORGE Original Data?

### 1. Detailed Vulnerability Information

Since FORGE contains **audit report JSONs**, you can extract:

```bash
# Example: Analyze all vulnerabilities by CWE code
cat data/datasets/FORGE-Artifacts/dataset/results/*.json | \
  jq -r '.findings[].category."1"[]' | \
  sort | uniq -c | sort -rn | head -20
```

This will show you:
- Most common CWE codes in real-world contracts
- Severity distribution
- Co-occurring vulnerabilities

### 2. Severity Analysis

```python
import json
from pathlib import Path
from collections import Counter

severity_counts = Counter()
cwe_severity = {}

for audit_file in Path("data/datasets/FORGE-Artifacts/dataset/results").glob("*.json"):
    with open(audit_file) as f:
        data = json.load(f)
        for finding in data.get("findings", []):
            severity = finding.get("severity", "unknown")
            severity_counts[severity] += 1

            # Map CWE to severity
            cwes = finding.get("category", {}).get("1", [])
            for cwe in cwes:
                cwe_severity[cwe] = severity

print("Severity Distribution:")
for severity, count in severity_counts.most_common():
    print(f"  {severity}: {count}")
```

### 3. Multi-Label Insights

FORGE contracts often have **multiple vulnerabilities**:

```python
# Find contracts with multiple CWEs
multi_vuln_contracts = []

for audit_file in audit_files:
    with open(audit_file) as f:
        data = json.load(f)
        all_cwes = set()
        for finding in data["findings"]:
            cwes = finding.get("category", {}).get("1", [])
            all_cwes.update(cwes)

        if len(all_cwes) > 1:
            multi_vuln_contracts.append({
                'contract': data['project_info']['project_path'],
                'cwes': list(all_cwes),
                'count': len(all_cwes)
            })
```

This helps you understand:
- Which vulnerabilities often appear together
- Whether your single-label approach is limiting
- Potential for multi-label classification

### 4. Compiler Version Patterns

```python
# Analyze which Solidity versions have more vulnerabilities
compiler_vulns = {}

for audit_file in audit_files:
    data = json.load(f)
    versions = data['project_info'].get('compiler_version', [])
    num_findings = len(data.get('findings', []))

    for version in versions:
        if version not in compiler_vulns:
            compiler_vulns[version] = []
        compiler_vulns[version].append(num_findings)
```

### 5. Blockchain-Specific Issues

```python
# Compare vulnerabilities across different chains
chain_vulns = {}

for audit_file in audit_files:
    data = json.load(f)
    chain = data['project_info'].get('chain', 'unknown')
    findings = data.get('findings', [])

    if chain not in chain_vulns:
        chain_vulns[chain] = []
    chain_vulns[chain].extend([f['category']['1'] for f in findings])
```

---

## Recommendations

### For Training Your Model

**Use**: `forge_balanced_accurate` (7,013 contracts)

**But first, preprocess**:
```bash
# 1. Flatten contracts to resolve dependencies
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_balanced_accurate/train \
    --output data/datasets/forge_flattened/train

# 2. Validate and filter problematic contracts
python scripts/dataset/validate_contracts.py \
    data/datasets/forge_flattened/train \
    --output-dir data/datasets/forge_clean/train \
    --copy-valid

# 3. Verify extraction works
./verify_contracts.sh data/datasets/forge_clean/train --max 100
```

### For Quick Testing

**Use**: `combined_labeled` (228 contracts)

This is clean and ready to go:
```bash
./start_training.sh static --train-dir data/datasets/combined_labeled/train
```

### For Research and Insights

**Explore**: `FORGE-Artifacts` audit JSONs

Create analysis scripts to:
1. Extract CWE statistics
2. Analyze severity distributions
3. Find multi-label examples
4. Study vulnerability co-occurrence

---

## Creating Custom Datasets

You can also create **hybrid datasets**:

```bash
# Combine high-quality curated + balanced FORGE
python scripts/dataset/create_hybrid_dataset.py \
    --base data/datasets/combined_labeled \
    --augment data/datasets/forge_balanced_accurate \
    --output data/datasets/hybrid_balanced \
    --target-per-class 1000
```

Or **re-balance** using different sampling:

```python
# Edit prepare_forge_dataset_accurate.py
samples_per_class = {
    'safe': 2000,              # More safe examples
    'reentrancy': 1500,        # More reentrancy
    'arithmetic': 1500,        # More arithmetic
    'bad_randomness': 500,     # Oversample rare class
    'short_addresses': 200,    # Oversample very rare
    ...
}
```

---

## Summary

You have **3 datasets** with different purposes:

1. **FORGE-Artifacts** (6,618 contracts)
   - Original research dataset
   - Rich audit metadata
   - Use for: Research, insights, understanding vulnerabilities

2. **combined_labeled** (228 contracts)
   - Small, curated, high-quality
   - Use for: Quick testing, validation

3. **forge_balanced_accurate** (7,013 contracts)
   - Large, balanced, real-world
   - Use for: Training deep learning models
   - **Needs preprocessing first!**

**Recommended workflow**:
1. Explore FORGE audit JSONs to understand vulnerability patterns
2. Test your pipeline on `combined_labeled` (clean, small)
3. Preprocess `forge_balanced_accurate` (flatten + validate)
4. Train final model on cleaned `forge_balanced_accurate`
5. Evaluate on held-out `forge_balanced_accurate/test`

This gives you the best of both worlds: **quality** from curated data + **scale** from FORGE!
