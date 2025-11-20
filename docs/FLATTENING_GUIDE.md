# Solidity Contract Flattening Guide

## What is Flattening?

**Contract flattening** combines all imported dependencies into a single `.sol` file. This eliminates:
- Missing import errors
- Dependency resolution issues
- Abstract contract problems
- Path resolution errors

## Why Flatten for Training?

When training, Slither needs to analyze each contract independently. Flattening:
1. ✅ Removes all `import` statements
2. ✅ Includes all dependencies inline
3. ✅ Resolves library contracts
4. ✅ Makes contracts self-contained
5. ✅ Improves PDG extraction success rate

## Quick Start

### Install a Flattening Tool

**Option 1: Foundry (Recommended)**
```bash
# Install Foundry
curl -L https://foundry.paradigm.xyz | bash
foundryup

# Verify
forge --version
```

**Option 2: Hardhat**
```bash
npm install --global hardhat
```

**Option 3: sol-merger (Python)**
```bash
pip install sol-merger
```

### Flatten Your Dataset

```bash
# Flatten entire dataset
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_balanced_accurate/train \
    --output data/datasets/forge_flattened/train \
    --batch

# Do same for val and test
python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_balanced_accurate/val \
    --output data/datasets/forge_flattened/val \
    --batch

python scripts/dataset/flatten_contracts.py \
    data/datasets/forge_balanced_accurate/test \
    --output data/datasets/forge_flattened/test \
    --batch
```

### Update Config

```yaml
data:
  train_dir: "data/datasets/forge_flattened/train"
  val_dir: "data/datasets/forge_flattened/val"
  test_dir: "data/datasets/forge_flattened/test"
```

### Train

```bash
./start_training.sh static
```

## Flattening Tools Comparison

| Tool | Speed | Accuracy | Installation |
|------|-------|----------|--------------|
| **Forge** | ⚡⚡⚡ Fast | ✅ Excellent | Easy (curl) |
| **Hardhat** | ⚡⚡ Medium | ✅ Excellent | Easy (npm) |
| **sol-merger** | ⚡ Slow | ✅ Good | Easy (pip) |
| **Simple** | ⚡⚡ Medium | ⚠️ Basic | Built-in |

**Recommendation:** Use Forge if available, it's the fastest and most reliable.

## Usage Examples

### Single File

```bash
# Flatten one contract
python scripts/dataset/flatten_contracts.py \
    contracts/MyToken.sol \
    --output contracts/MyToken.flat.sol
```

### Batch Processing

```bash
# Flatten entire directory
python scripts/dataset/flatten_contracts.py \
    data/datasets/raw/train \
    --output data/datasets/flattened/train \
    --batch
```

### Specify Tool

```bash
# Use specific tool
python scripts/dataset/flatten_contracts.py \
    contracts/MyToken.sol \
    --output contracts/MyToken.flat.sol \
    --tool forge
```

### Print to stdout

```bash
# Print flattened code
python scripts/dataset/flatten_contracts.py contracts/MyToken.sol
```

## Before vs After

### Before Flattening

```solidity
// Token.sol
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "./Ownable.sol";

contract Token is ERC20, Ownable {
    constructor() ERC20("MyToken", "MTK") {}
}
```

**Problem:** Slither can't find `@openzeppelin` or `Ownable.sol`

### After Flattening

```solidity
// Token.flat.sol
pragma solidity ^0.8.0;

// ========== ERC20.sol ==========
contract ERC20 {
    // ... full ERC20 code here ...
}

// ========== Ownable.sol ==========
contract Ownable {
    // ... full Ownable code here ...
}

contract Token is ERC20, Ownable {
    constructor() ERC20("MyToken", "MTK") {}
}
```

**Result:** Self-contained, Slither can analyze it!

## Workflow

### Complete Preprocessing Pipeline

```bash
# 1. Flatten to resolve dependencies
python scripts/dataset/flatten_contracts.py \
    data/datasets/raw/train \
    --output data/datasets/flattened/train \
    --batch

# 2. Validate flattened contracts
python scripts/dataset/validate_contracts.py \
    data/datasets/flattened/train \
    --output-dir data/datasets/clean/train \
    --copy-valid

# 3. Update config
# Edit config.yaml to use data/datasets/clean/train

# 4. Train
./start_training.sh static
```

### Expected Results

**Before Flattening:**
- Import errors: ~40%
- Empty PDGs: ~30%
- Training success: ~60%

**After Flattening:**
- Import errors: ~0%
- Empty PDGs: ~5%
- Training success: ~95%

## Tool Installation

### Forge (Foundry)

```bash
# Install
curl -L https://foundry.paradigm.xyz | bash
source ~/.bashrc  # or ~/.zshrc
foundryup

# Test
forge flatten --help
```

### Hardhat

```bash
# Install Node.js first (if needed)
# Then install Hardhat
npm install --save-dev hardhat

# Or globally
npm install -g hardhat

# Test
npx hardhat flatten --help
```

### sol-merger

```bash
# Install via pip
pip install sol-merger

# Test
python -c "import sol_merger; print('OK')"
```

### truffle-flattener

```bash
npm install -g truffle-flattener

# Test
truffle-flattener --help
```

## Troubleshooting

### No Flattening Tools Available

If no tools are installed, the script uses a simple built-in flattener:

```bash
python scripts/dataset/flatten_contracts.py \
    contracts/MyToken.sol \
    --output contracts/MyToken.flat.sol \
    --tool simple
```

**Note:** Simple flattener only resolves relative imports, not node_modules.

### Flattening Fails on Some Contracts

```bash
# The script will copy the original if flattening fails
# Check logs for:
[50/100] ✗ ComplexContract.sol - Flattening failed
    Copied original instead
```

These contracts may still work or can be validated separately.

### Circular Dependencies

Some contracts have circular imports:
```
A.sol imports B.sol
B.sol imports A.sol
```

**Solution:** Forge handles this automatically. Other tools may fail.

### Large Contracts

Flattened contracts can be very large (> 10,000 lines).

**Solution:** This is normal and fine for analysis. Slither can handle it.

## Integration with Training

### Option 1: Flatten First (Recommended)

```bash
# Preprocess once
python scripts/dataset/flatten_contracts.py data/raw --output data/flat --batch
python scripts/dataset/validate_contracts.py data/flat --output data/clean --copy-valid

# Train many times on clean data
./start_training.sh static
```

### Option 2: On-the-Fly Flattening

Modify dataset loading to flatten during training (slower but automatic):

```python
# In StaticDataset.__init__
from scripts.dataset.flatten_contracts import SolidityFlattener

self.flattener = SolidityFlattener()

# In __getitem__
flattened_code = self.flattener.flatten(contract_path)[1]
```

## Best Practices

1. **Flatten before validation** - Resolves imports first, then validate
2. **Keep originals** - Store flattened in separate directory
3. **Use Forge** - Fastest and most reliable
4. **Batch process** - Flatten entire dataset at once
5. **Cache results** - Don't re-flatten unchanged contracts

## Performance

### Flattening Speed

| Dataset Size | Tool | Time |
|--------------|------|------|
| 1,000 contracts | Forge | ~2 min |
| 1,000 contracts | Hardhat | ~5 min |
| 1,000 contracts | sol-merger | ~10 min |
| 1,000 contracts | Simple | ~3 min |

### Storage

- Original: ~100 KB per contract
- Flattened: ~300-500 KB per contract (includes dependencies)

Plan for 3-5x storage increase.

## FAQ

**Q: Do I need to flatten for training?**
A: Not required, but highly recommended if you see import errors.

**Q: Will flattening change the contract?**
A: No, it only reorganizes the code. Functionality is identical.

**Q: Can I flatten contracts with node_modules imports?**
A: Yes, but you need Forge, Hardhat, or Truffle. Simple flattener can't resolve node_modules.

**Q: What about @openzeppelin imports?**
A: All tools can handle these if node_modules are installed.

**Q: Should I flatten before or after splitting train/val/test?**
A: Before is easier. Flatten the whole dataset, then split.

**Q: Does flattening help with abstract contracts?**
A: Yes! It inlines the abstract contracts, making them concrete.

## Summary

Flattening is a powerful preprocessing step that:
- ✅ Eliminates import/dependency errors
- ✅ Makes contracts self-contained
- ✅ Improves Slither analysis success rate
- ✅ Increases PDG extraction quality
- ✅ Results in better training

**Recommended workflow:**
1. Flatten dataset
2. Validate contracts
3. Train on clean, flattened data
