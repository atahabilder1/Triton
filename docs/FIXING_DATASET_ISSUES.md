# Fixing Dataset Issues - Abstract Contracts & Dependencies

## Problem

During static training, some smart contracts fail to encode properly because:
1. **Abstract contracts** - Contain unimplemented functions
2. **Missing dependencies** - Import statements that can't be resolved
3. **Compilation errors** - Syntax issues or version mismatches
4. **Too simple** - Not enough code to extract meaningful features

This causes Slither to fail and produces empty PDGs, reducing training quality.

## Solution

Use the contract validation script to filter your dataset before training.

## Quick Fix

### 1. Validate Your Dataset

```bash
# Check which contracts are problematic
python scripts/dataset/validate_contracts.py \
    data/datasets/forge_balanced_accurate/train

# This will create invalid_contracts.txt with details
```

### 2. Create Clean Dataset

```bash
# Copy only valid contracts to a new directory
python scripts/dataset/validate_contracts.py \
    data/datasets/forge_balanced_accurate/train \
    --output-dir data/datasets/forge_balanced_accurate_clean/train \
    --copy-valid
```

### 3. Update config.yaml

```yaml
data:
  train_dir: "data/datasets/forge_balanced_accurate_clean/train"
  val_dir: "data/datasets/forge_balanced_accurate_clean/val"
  test_dir: "data/datasets/forge_balanced_accurate_clean/test"
```

### 4. Train on Clean Dataset

```bash
./start_training.sh static
```

## What the Validator Checks

### 1. Abstract Contracts ❌
```solidity
// Will be filtered out
abstract contract Token {
    function transfer(address to, uint amount) public virtual;
}
```

### 2. Missing Dependencies ⚠️
```solidity
// May cause issues
import "./NonexistentFile.sol";
import "../contracts/MissingContract.sol";
```

### 3. Syntax Errors ❌
```solidity
// Will be filtered out
contract Broken {
    function test() {
        // Missing closing brace
}
```

### 4. Too Simple ❌
```solidity
// Will be filtered out (< 2 functions, no state)
contract Empty {
    function hello() public pure returns (string) {
        return "hi";
    }
}
```

### 5. Compilation Test ✅
- Runs Slither analysis
- Checks if PDG can be extracted
- Verifies output is usable

## Detailed Usage

### Validate and Get Statistics

```bash
python scripts/dataset/validate_contracts.py data/datasets/your_dataset/train
```

**Output:**
```
Found 1000 Solidity files

[1/1000] ✓ SafeToken.sol - VALID
[2/1000] ✗ AbstractERC20.sol - Abstract contract
[3/1000] ✗ SimpleHello.sol - Contract too simple
...

VALIDATION STATISTICS
================================================================================
Total contracts processed: 1000
Valid contracts: 750 (75.0%)
Abstract contracts: 50
Too simple: 100
Syntax errors: 20
Missing dependencies: 30
Compilation failed: 50
================================================================================
```

### Filter Dataset (Keep Only Valid)

```bash
# Clean train set
python scripts/dataset/validate_contracts.py \
    data/datasets/forge_balanced_accurate/train \
    --output-dir data/datasets/forge_clean/train \
    --copy-valid

# Clean validation set
python scripts/dataset/validate_contracts.py \
    data/datasets/forge_balanced_accurate/val \
    --output-dir data/datasets/forge_clean/val \
    --copy-valid

# Clean test set
python scripts/dataset/validate_contracts.py \
    data/datasets/forge_balanced_accurate/test \
    --output-dir data/datasets/forge_clean/test \
    --copy-valid
```

### Review Invalid Contracts

After validation, check `invalid_contracts.txt`:

```
Invalid Contracts Log
================================================================================

data/datasets/.../AbstractToken.sol
  Reason: Abstract contract

data/datasets/.../BrokenContract.sol
  Reason: Compilation failed: Syntax error

data/datasets/.../TinyContract.sol
  Reason: Contract too simple
```

## Improving Slither Compatibility

### Common Issues and Fixes

#### 1. Solidity Version Mismatch

**Problem:** Contract uses old Solidity version not installed

**Fix:** Install multiple Solidity versions
```bash
# Install solc-select
pip install solc-select

# Install common versions
solc-select install 0.4.26
solc-select install 0.5.17
solc-select install 0.6.12
solc-select install 0.7.6
solc-select install 0.8.20

# The validator will auto-switch versions
```

#### 2. Missing OpenZeppelin Imports

**Problem:** Contract imports @openzeppelin but files not found

**Fix:** These are usually safe to ignore. The validator allows @openzeppelin imports.

If needed, install:
```bash
npm install @openzeppelin/contracts
```

#### 3. Large Contracts Timeout

**Problem:** Slither takes too long on very large contracts

**Fix:** Increase timeout in config.yaml
```yaml
processing:
  slither_timeout: 120  # Increase from 60 to 120 seconds
```

## Training Configuration Updates

### Enable Better Error Handling

Update `config.yaml`:

```yaml
processing:
  use_cache: true
  slither_timeout: 60

  # Skip problematic contracts during training
  skip_on_error: true
  max_retries: 2

training:
  static:
    # Allow some failed contracts without stopping
    tolerate_failures: true
    max_failure_rate: 0.1  # Stop if > 10% fail
```

## Validation Workflow

### Recommended Process

1. **Initial Validation**
   ```bash
   # See what you're working with
   python scripts/dataset/validate_contracts.py data/datasets/raw/train
   ```

2. **Create Clean Dataset**
   ```bash
   # Copy valid contracts only
   python scripts/dataset/validate_contracts.py \
       data/datasets/raw/train \
       --output-dir data/datasets/clean/train \
       --copy-valid
   ```

3. **Update Config**
   ```yaml
   data:
     train_dir: "data/datasets/clean/train"
   ```

4. **Train**
   ```bash
   ./start_training.sh static
   ```

5. **Monitor**
   - Check logs for any remaining errors
   - Look for PDG extraction success rate
   - Verify model is learning

## Expected Results

### Before Validation
- ~30-40% of contracts fail during training
- Empty PDGs from abstract contracts
- Training stops on errors
- Poor model performance

### After Validation
- ~95%+ successful PDG extraction
- No abstract contracts
- Smooth training process
- Better model performance

## Checking Training Quality

### During Training

Monitor the logs for:
```
✓ Loaded 1000 contracts total
✓ Successfully extracted PDG with 50 nodes, 120 edges
✓ PDG cache hit rate: 85%
```

### After Training

Check results:
```
Static Encoder Test: 950/1000 successful (95.0% success rate)
```

Good: > 90% success rate
Needs work: < 80% success rate

## Troubleshooting

### Still Getting Errors?

1. **Check Solidity versions**
   ```bash
   solc-select versions
   ```
   Install missing versions

2. **Review failed contracts**
   ```bash
   cat data/datasets/your_dataset/invalid_contracts.txt
   ```

3. **Test individual contract**
   ```bash
   slither path/to/contract.sol
   ```

4. **Increase verbosity**
   ```python
   # In scripts, set:
   logging.basicConfig(level=logging.DEBUG)
   ```

### Very Low Success Rate (< 50%)

Possible causes:
- Dataset has many abstract/interface contracts
- Solidity version mismatches
- Complex dependency chains
- Very old Solidity (< 0.4.0)

Solution: Consider using a different dataset or manually curating contracts.

## Best Practices

1. **Always validate before training**
2. **Keep validation logs** for reference
3. **Start with small subset** to test
4. **Monitor first few batches** of training
5. **Compare clean vs raw** dataset performance

## Script Options

```bash
# Full options
python scripts/dataset/validate_contracts.py \
    INPUT_DIR \
    --output-dir OUTPUT_DIR \      # Where to copy valid contracts
    --copy-valid \                 # Enable copying
    --quick-check                  # Skip compilation test (faster)
```

## Integration with Training

The validation script is standalone but integrates with training:

1. **Pre-training:** Validate dataset
2. **Training:** Use clean dataset
3. **Post-training:** Review any remaining errors

Training scripts will still handle errors gracefully, but validation ensures maximum data quality.
