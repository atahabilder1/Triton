# How the Training Dataset Was Created

## Overview

Your `forge_balanced_accurate` dataset was created from the **FORGE-Artifacts** dataset using the script:
- **Script**: `scripts/dataset/prepare_forge_dataset_accurate.py`
- **Method**: Accurate CWE (Common Weakness Enumeration) mapping
- **Total Contracts**: 7,013 contracts
- **Classes**: 11 vulnerability classes (10 + safe)
- **Split**: 70% train / 15% val / 15% test

## Process

### Step 1: CWE → Vulnerability Class Mapping

The script maps **303 CWE codes** to **10 vulnerability classes**:

```python
CWE_TO_CLASS = {
    # Access Control (50+ CWEs mapped)
    'CWE-284': 'access_control',
    'CWE-269': 'access_control',
    ...

    # Arithmetic (9 CWEs mapped)
    'CWE-682': 'arithmetic',
    'CWE-190': 'arithmetic',  # Overflow
    'CWE-191': 'arithmetic',  # Underflow
    ...

    # Reentrancy (11 CWEs mapped)
    'CWE-362': 'reentrancy',  # Race condition
    'CWE-1265': 'reentrancy', # Reentrant call
    ...

    # And 7 more classes...
}
```

### Step 2: Parse FORGE Audit Reports

For each FORGE audit JSON file:
1. Extract all CWE codes from findings
2. Map CWEs to vulnerability class (using priority order)
3. Find corresponding smart contract files
4. Categorize contract by most critical vulnerability

**Priority Order** (most critical first):
1. Reentrancy
2. Arithmetic
3. Bad randomness
4. Time manipulation
5. Short addresses
6. Front running
7. Denial of service
8. Unchecked calls
9. Access control
10. Other

### Step 3: Balance Dataset

Sample contracts to create balanced classes:

```python
samples_per_class = {
    'safe': 1000,
    'access_control': 1000,
    'arithmetic': 1000,
    'unchecked_low_level_calls': 1000,
    'reentrancy': 800,
    'bad_randomness': 300,      # Limited in FORGE
    'denial_of_service': 500,
    'front_running': 300,       # Limited in FORGE
    'time_manipulation': 300,
    'short_addresses': 200,     # Very limited
    'other': 1000
}
```

### Step 4: Train/Val/Test Split

- **70%** → Training
- **15%** → Validation
- **15%** → Test

**Result**:
```json
{
  "train": {
    "safe": 700,
    "access_control": 700,
    "arithmetic": 700,
    "unchecked_low_level_calls": 700,
    "reentrancy": 560,
    "bad_randomness": 112,
    "denial_of_service": 350,
    "front_running": 147,
    "time_manipulation": 210,
    "short_addresses": 30,
    "other": 700
  }
}
```

### Step 5: Copy Contracts

Contracts organized by vulnerability type:

```
data/datasets/forge_balanced_accurate/
├── train/
│   ├── reentrancy/         (560 contracts)
│   ├── safe/               (700 contracts)
│   ├── access_control/     (700 contracts)
│   ├── arithmetic/         (700 contracts)
│   └── ... (11 classes total)
├── val/
│   └── ... (same structure, smaller counts)
└── test/
    └── ... (same structure, smaller counts)
```

## Dataset Characteristics

### Class Distribution

| Class | Train | Val | Test | Total | Notes |
|-------|-------|-----|------|-------|-------|
| safe | 700 | 150 | 150 | 1000 | Contracts with no vulnerabilities |
| access_control | 700 | 150 | 150 | 1000 | Most common vulnerability |
| arithmetic | 700 | 150 | 150 | 1000 | Overflows/underflows |
| unchecked_calls | 700 | 150 | 150 | 1000 | Missing error checks |
| reentrancy | 560 | 120 | 120 | 800 | Critical smart contract bug |
| other | 700 | 150 | 150 | 1000 | Code quality issues |
| denial_of_service | 350 | 75 | 75 | 500 | Gas limit, loops |
| time_manipulation | 210 | 45 | 45 | 300 | Timestamp dependency |
| front_running | 147 | 31 | 32 | 210 | Transaction ordering |
| bad_randomness | 112 | 24 | 24 | 160 | Weak randomness |
| short_addresses | 30 | 6 | 7 | 43 | Rare vulnerability |

### Balanced Features

✅ **Well-represented**:
- safe, access_control, arithmetic, unchecked_calls, other: 1000 samples each
- reentrancy: 800 samples (important but rarer)

⚠️ **Moderately represented**:
- denial_of_service: 500 samples
- time_manipulation: 300 samples
- front_running: ~210 samples

❌ **Under-represented**:
- bad_randomness: 160 samples
- short_addresses: 43 samples (very rare in practice)

## How to Recreate

If you want to recreate or modify the dataset:

```bash
python scripts/dataset/prepare_forge_dataset_accurate.py \
    --forge-dir /path/to/FORGE-Artifacts/dataset \
    --output-dir data/datasets/my_custom_dataset \
    --seed 42
```

### Customize Sampling

Edit the script to change samples per class:

```python
samples_per_class = {
    'safe': 2000,              # More safe contracts
    'reentrancy': 1500,        # More reentrancy
    'arithmetic': 1500,        # More arithmetic
    ...
}
```

## Quality Considerations

### Why Some Contracts Fail Training

Even though the dataset was well-organized, some contracts still fail during training because:

1. **Abstract Contracts**
   - FORGE includes interface/abstract contracts
   - Can't be compiled standalone
   - **Solution**: Flatten or filter

2. **Missing Dependencies**
   - Import statements not resolved
   - OpenZeppelin, external libraries
   - **Solution**: Flatten contracts

3. **Compilation Errors**
   - Solidity version mismatches
   - Syntax incompatibilities
   - **Solution**: Version management, validation

4. **Complex Contracts**
   - Very large contracts timeout
   - Deep inheritance chains
   - **Solution**: Increase timeouts

### Improving Dataset Quality

Use the preprocessing tools we created:

```bash
# 1. Flatten to resolve dependencies
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_balanced_accurate/train \
    --output data/datasets/forge_flattened/train \
    --batch

# 2. Validate and filter
python scripts/dataset/validate_contracts.py \
    data/datasets/forge_flattened/train \
    --output-dir data/datasets/forge_clean/train \
    --copy-valid

# 3. Verify extraction works
./verify_contracts.sh data/datasets/forge_clean/train --max 100
```

## Dataset Statistics

From `dataset_summary.json`:

```json
{
  "total_contracts": 7013,
  "cwe_mapping_coverage": {
    "total_cwes_in_dataset": 303,
    "mapped_cwes": 127,
    "unmapped_cwes": 191
  },
  "seed": 42
}
```

- **127 CWEs** explicitly mapped to classes
- **191 CWEs** defaulted to "other" class
- **Random seed 42** ensures reproducibility

## Alternative Datasets

If you want to try different data sources:

### 1. SmartBugs
```bash
# Located in your project
data/datasets/smartbugs/
```

### 2. Security Audits
```bash
# Located in your project
data/datasets/audits/
```

### 3. Create Custom Dataset

Use the template:

```python
# Custom dataset creation
python scripts/dataset/combine_labeled_datasets.py \
    --source1 data/datasets/smartbugs \
    --source2 data/datasets/audits \
    --output data/datasets/combined \
    --balance
```

## Summary

Your `forge_balanced_accurate` dataset is:

✅ **Well-organized** - Clear vulnerability classification
✅ **Balanced** - Most classes have 500-1000 samples
✅ **Research-based** - Uses standard CWE mapping
✅ **Reproducible** - Fixed random seed (42)
✅ **Ready for training** - train/val/test splits done

But can be **improved** with:
1. Flattening (resolve imports)
2. Validation (remove abstract contracts)
3. Verification (ensure PDG/AST extraction works)

Use the preprocessing pipeline we created for best results!
