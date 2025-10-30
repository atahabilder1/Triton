# SmartBugs Dataset Overview

## Dataset Location

**Main Directory:** `/home/anik/code/Triton/data/datasets/smartbugs/`

## Dataset Structure

```
/home/anik/code/Triton/data/datasets/smartbugs/
├── samples/
│   ├── 0.4.x/           # 10 real-world contracts (deployed on mainnet)
│   ├── 0.5.17/          # Same 10 contracts adapted for Solidity 0.5.17
│   ├── 0.6.12/          # Same 10 contracts adapted for Solidity 0.6.12
│   ├── 0.7.6/           # Same 10 contracts adapted for Solidity 0.7.6
│   ├── 0.8.24/          # Same 10 contracts adapted for Solidity 0.8.24
│   └── vulnerabilities.json  # Metadata with vulnerability locations
│
└── tools/               # Analysis tools (Mythril, Slither, etc.)
```

**Total Contracts:** 50 (10 unique contracts × 5 Solidity versions)

## Contracts in the Dataset

Based on `vulnerabilities.json`, here are the 10 contracts and their vulnerabilities:

### 1. **Rubixi.sol**
- **Source:** Trail of Bits - Not So Smart Contracts
- **Vulnerability:** Access Control
- **Lines:** 23-24
- **Description:** Wrong constructor name vulnerability

### 2. **BecToken.sol**
- **Source:** SWC Registry (SWC-101)
- **Vulnerability:** Arithmetic (Integer Overflow)
- **Lines:** 264
- **Description:** Integer overflow in multiplication

### 3. **SmartBillions.sol**
- **Source:** Etherscan (0x5ace17f8...)
- **Vulnerability:** Bad Randomness
- **Lines:** 523, 560, 700, 702, 704, 706, 708, 710, 712, 714, 716, 718
- **Description:** Multiple instances of weak randomness

### 4. **Government.sol**
- **Source:** Etherscan (0xf457175...)
- **Vulnerability:** Denial of Service
- **Lines:** 46, 48
- **Description:** DoS with unexpected revert

### 5. **ERC20.sol**
- **Source:** SWC Registry
- **Vulnerability:** Front Running
- **Lines:** 110, 113
- **Description:** Transaction order dependence

### 6. **OpenAddressLottery.sol**
- **Source:** Etherscan (0x741f192...)
- **Vulnerability:** Other
- **Lines:** 91
- **Description:** Miscellaneous vulnerability

### 7. **EtherLotto.sol**
- **Source:** Etherscan
- **Vulnerability:** Reentrancy
- **Description:** Classic reentrancy vulnerability

### 8. **SimpleDAO.sol**
- **Source:** Trail of Bits
- **Vulnerability:** Reentrancy
- **Description:** Simple DAO attack pattern (like The DAO hack)

### 9. **MyToken.sol**
- **Source:** Various
- **Vulnerability:** Multiple
- **Description:** Token contract with various issues

### 10. **ReturnValue.sol**
- **Source:** SWC Registry
- **Vulnerability:** Unchecked Return Value
- **Description:** Ignoring return values of external calls

## Vulnerability Categories

| Category | Count | Examples |
|----------|-------|----------|
| **access_control** | 1 | Wrong constructor name (Rubixi) |
| **arithmetic** | 1 | Integer overflow (BecToken) |
| **bad_randomness** | 12 | Weak randomness (SmartBillions) |
| **denial_of_service** | 2 | Unexpected revert (Government) |
| **front_running** | 2 | Transaction ordering (ERC20) |
| **reentrancy** | 2 | Classic reentrancy (SimpleDAO, EtherLotto) |
| **unchecked_return_value** | 1 | Ignored call returns (ReturnValue) |
| **other** | 1 | Miscellaneous (OpenAddressLottery) |

## File Types per Contract

Each contract has 3 files:

1. **`.sol`** - Solidity source code
2. **`.hex`** - Compiled bytecode
3. **`.rt.hex`** - Runtime bytecode

Example for `BecToken`:
- `BecToken.sol` - Source code
- `BecToken.hex` - Creation bytecode
- `BecToken.rt.hex` - Runtime bytecode

## Viewing Contracts

### List all contracts in one version:

```bash
ls /home/anik/code/Triton/data/datasets/smartbugs/samples/0.6.12/*.sol
```

### View a specific contract:

```bash
cat /home/anik/code/Triton/data/datasets/smartbugs/samples/0.6.12/SimpleDAO.sol
```

### View vulnerability metadata:

```bash
python3 -m json.tool /home/anik/code/Triton/data/datasets/smartbugs/samples/vulnerabilities.json
```

## Testing with Triton

### Test on all contracts (all versions):

```bash
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs/samples \
    --output-dir results
```

### Test on specific Solidity version only:

```bash
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs/samples/0.6.12 \
    --output-dir results
```

### Test single contract:

```bash
python main.py data/datasets/smartbugs/samples/0.6.12/SimpleDAO.sol --verbose
```

## Vulnerability Details Reference

### Access Control (Rubixi.sol)
```
Wrong constructor name - function should be constructor
Can be exploited to take ownership of contract
```

### Arithmetic (BecToken.sol)
```
Integer overflow in token balance calculation
Can create unlimited tokens
```

### Reentrancy (SimpleDAO.sol)
```
Classic DAO attack pattern
External call before state update
Attacker can drain funds
```

### Bad Randomness (SmartBillions.sol)
```
Using block.timestamp and block.number for randomness
Predictable by miners
Can manipulate lottery results
```

### Denial of Service (Government.sol)
```
External call in loop can revert entire transaction
Attacker can block all interactions
```

### Front Running (ERC20.sol)
```
approve/transferFrom pattern vulnerable to double-spend
Attacker can frontrun approve transaction
```

## Downloading More Datasets

If you want more contracts for testing:

### SmartBugs Curated (143 contracts):
```bash
cd data/datasets
git clone https://github.com/smartbugs/smartbugs-curated.git
```

### SolidiFI (9,369 contracts):
```bash
python scripts/download_datasets.py --dataset solidifi
```

### SmartBugs Wild (47,398 contracts):
```bash
python scripts/download_datasets.py --dataset wild
```

## Understanding the Dataset Files

### vulnerabilities.json Format

```json
{
    "name": "SimpleDAO.sol",
    "source": "https://github.com/trailofbits/...",
    "vulnerabilities": [
        {
            "lines": [15, 16, 17],
            "category": "reentrancy"
        }
    ]
}
```

This tells you:
- **name**: Contract filename
- **source**: Where it came from
- **lines**: Exact lines with vulnerability
- **category**: Type of vulnerability

## Quick Stats

```bash
# Count total .sol files
find data/datasets/smartbugs/samples -name "*.sol" | wc -l

# Count contracts per version
ls data/datasets/smartbugs/samples/0.6.12/*.sol | wc -l

# Show all contract names
ls data/datasets/smartbugs/samples/0.6.12/*.sol | xargs -n1 basename
```

## Example: Analyzing SimpleDAO

```bash
# View the vulnerable contract
cat data/datasets/smartbugs/samples/0.6.12/SimpleDAO.sol

# Analyze with Triton
python main.py data/datasets/smartbugs/samples/0.6.12/SimpleDAO.sol \
    --target-vulnerability reentrancy \
    --verbose \
    --output results/simpledao_analysis.json

# View results
cat results/simpledao_analysis.json | python3 -m json.tool
```

## Next Steps

1. **Explore the contracts**
   ```bash
   cd data/datasets/smartbugs/samples/0.6.12
   ls -la
   cat SimpleDAO.sol  # Look at a reentrancy example
   cat BecToken.sol   # Look at an overflow example
   ```

2. **Run Triton on the dataset**
   ```bash
   ./run_tests.sh
   # Choose option 3 or 4
   ```

3. **Analyze results**
   ```bash
   cat results/triton_test_summary_*.txt
   ```

## References

- **SmartBugs GitHub:** https://github.com/smartbugs/smartbugs
- **SmartBugs Curated:** https://github.com/smartbugs/smartbugs-curated
- **SWC Registry:** https://swcregistry.io/
- **Trail of Bits Examples:** https://github.com/crytic/not-so-smart-contracts
