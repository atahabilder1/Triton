# Contract Extraction Verification Guide

## Quick Start

Verify that PDG and AST extraction works correctly on your contracts:

```bash
# Verify 100 contracts from reentrancy vulnerability folder
./verify_contracts.sh data/datasets/forge_balanced_accurate/train/reentrancy

# Verify 50 contracts from safe folder
./verify_contracts.sh data/datasets/forge_balanced_accurate/train/safe --max 50

# Verify with custom output report
./verify_contracts.sh data/datasets/forge_balanced_accurate/train --max 100 --output my_report.json
```

## What It Does

The verification script processes each Solidity contract and:

1. **Reads** the contract source code
2. **Flattens** the contract (if needed, configured in config.yaml)
3. **Extracts PDG** (Program Dependence Graph) using Slither
4. **Extracts AST** (Abstract Syntax Tree) using solc
5. **Reports** detailed statistics and success rates

## Output

### Console Output

```
================================================================================
[1/100]
================================================================================
Verifying: reentrancy_simple.sol
================================================================================
  PDG: ✓ PDG: 45 nodes, 67 edges
  AST: ✓ AST: 1 contracts, 5 functions
  ✅ BOTH SUCCESSFUL

================================================================================
[2/100]
================================================================================
Verifying: dao.sol
================================================================================
  PDG: ✓ PDG: 123 nodes, 189 edges
  AST: ✓ AST: 2 contracts, 12 functions
  ✅ BOTH SUCCESSFUL

...

================================================================================
VERIFICATION SUMMARY
================================================================================

Total Contracts: 100

PDG Extraction:
  ✓ Successful: 95 (95.0%)
  ✗ Failed:     5 (5.0%)

AST Extraction:
  ✓ Successful: 98 (98.0%)
  ✗ Failed:     2 (2.0%)

Both Successful: 93 (93.0%)

PDG Statistics (avg per successful contract):
  Nodes: 78.3
  Edges: 124.5

AST Statistics (avg per successful contract):
  Contracts:     1.2
  Functions:     8.7
  State Vars:    4.3

Failed/Partial Contracts (7):
  - complex_contract_1.sol
  - library_usage.sol
  ...

================================================================================
```

### JSON Report

```json
{
  "timestamp": "2025-11-19T18:30:00",
  "summary": {
    "total_contracts": 100,
    "pdg_success": 95,
    "pdg_failed": 5,
    "ast_success": 98,
    "ast_failed": 2,
    "both_success": 93
  },
  "pdg_stats": {
    "total_nodes": 7438,
    "total_edges": 11827
  },
  "ast_stats": {
    "total_contracts": 118,
    "total_functions": 853,
    "total_state_vars": 421
  },
  "failed_contracts": [
    "path/to/failed_contract_1.sol",
    "path/to/failed_contract_2.sol"
  ]
}
```

## Configuration

The verification uses settings from `config.yaml`:

```yaml
processing:
  # Timeouts
  slither_timeout: 60      # Timeout for Slither/PDG extraction
  solc_timeout: 30         # Timeout for solc/AST extraction

  # Solidity Compiler
  solc_path: "solc"        # Path to solc binary
  solc_version: "0.5.16"   # Solidity version
  enable_flattening: true  # Flatten before analysis
```

## Understanding Results

### PDG (Program Dependence Graph)
- **Nodes**: Represents statements, variables, and operations
- **Edges**: Represents data and control flow dependencies
- **Success**: PDG extracted successfully with nodes and edges
- **Failure**: Slither couldn't analyze the contract (compilation errors, timeout, etc.)

### AST (Abstract Syntax Tree)
- **Contracts**: Number of contract definitions in the file
- **Functions**: Total functions across all contracts
- **State Vars**: State variables in the contracts
- **Success**: AST extracted successfully with structure info
- **Failure**: solc couldn't compile the contract

### Success Rates
- **Both Successful**: Contract has both PDG and AST extracted ✅
- **80%+ Success Rate**: Good - most contracts process correctly
- **Below 80%**: Needs attention - check Solidity version compatibility

## Common Issues

### Low PDG Success Rate
**Causes:**
- Compilation errors in contracts
- Unsupported Solidity features
- Timeout (complex contracts)

**Solutions:**
1. Check Solidity version compatibility
2. Increase `slither_timeout` in config.yaml
3. Fix compilation errors in contracts
4. Use `solc-select` to switch Solidity versions

### Low AST Success Rate
**Causes:**
- Syntax errors
- Wrong Solidity version
- Missing imports

**Solutions:**
1. Verify solc version: `solc --version`
2. Update `solc_version` in config.yaml
3. Install correct Solidity version
4. Enable flattening for contracts with imports

### Timeout Issues
**Symptoms:**
- Script hangs on complex contracts
- Many failures with timeout errors

**Solutions:**
1. Increase timeouts in config.yaml:
   ```yaml
   processing:
     slither_timeout: 120  # Increase from 60
     solc_timeout: 60      # Increase from 30
   ```
2. Test with smaller batches: `--max 10`

## Advanced Usage

### Verify Specific Vulnerability Type
```bash
# Verify only reentrancy contracts
./verify_contracts.sh data/datasets/forge_balanced_accurate/train/reentrancy --max 100

# Verify only safe contracts
./verify_contracts.sh data/datasets/forge_balanced_accurate/train/safe --max 100
```

### Test All Vulnerability Types
```bash
for type in reentrancy safe access_control arithmetic; do
  echo "Testing $type..."
  ./verify_contracts.sh data/datasets/forge_balanced_accurate/train/$type --max 20
done
```

### Batch Verification
```bash
# Create a test script
cat > test_all.sh << 'EOF'
#!/bin/bash
for dir in data/datasets/forge_balanced_accurate/train/*/; do
  vuln_type=$(basename "$dir")
  echo "Testing $vuln_type..."
  ./verify_contracts.sh "$dir" --max 20 --output "report_${vuln_type}.json"
done
EOF

chmod +x test_all.sh
./test_all.sh
```

## Integration with Training

Before training, verify your dataset:

```bash
# Verify training data
./verify_contracts.sh data/datasets/forge_balanced_accurate/train --max 200

# If success rate is high (>80%), proceed with training
./start_training.sh static
```

## Interpreting Statistics

### Good Results
```
PDG Success: 90%+
AST Success: 95%+
Both Success: 85%+
```
✅ Dataset is ready for training

### Needs Improvement
```
PDG Success: 60-80%
AST Success: 70-90%
Both Success: 50-70%
```
⚠️ Check Solidity versions and fix compilation errors

### Poor Results
```
PDG Success: <60%
AST Success: <70%
Both Success: <50%
```
❌ Major issues - review contracts and tool configuration

## Next Steps

1. **Run verification**: `./verify_contracts.sh <dataset_dir>`
2. **Review results**: Check success rates and failed contracts
3. **Fix issues**: Update config.yaml or fix contracts if needed
4. **Verify again**: Re-run after fixes
5. **Train model**: Once verified, start training with confidence

## Troubleshooting

### Script fails with "command not found"
```bash
# Make sure script is executable
chmod +x verify_contracts.sh

# Run with explicit bash
bash verify_contracts.sh data/datasets/...
```

### Python import errors
```bash
# Ensure you're in project root
cd /path/to/Triton

# Check Python path
python -c "import sys; print(sys.path)"
```

### Slither not found
```bash
# Install Slither
pip install slither-analyzer

# Verify installation
slither --version
```

### solc not found
```bash
# Install Solidity compiler
sudo apt install solc  # Ubuntu/Debian

# Or use solc-select
pip install solc-select
solc-select install 0.5.16
solc-select use 0.5.16
```

## Files Created

- `verify_contracts.sh` - Main verification launcher
- `scripts/utils/verify_extraction.py` - Verification script
- `verification_report_*.json` - Generated reports
- `config.yaml` - Updated with Solidity settings
