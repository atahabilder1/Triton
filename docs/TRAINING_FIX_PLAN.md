# How to Fix 11% Accuracy Training

## 🚨 Root Causes Found

### 1. **PDGs Are Too Small** (3-10 nodes) ❌ **CRITICAL**

**Why**: Your Slither wrapper only extracts:
- Contract-level structure (state variables, functions)
- High-level dependencies (function calls, variable reads/writes)
- **BUT NOT**: Detailed control flow, statements, expression-level data flow

**Result**: PDGs with only 3-10 nodes (should be 50-500+)

### 2. **Dataset Too Small** (228 contracts) ❌

Deep learning needs **1000+** examples minimum

### 3. **Flattening Not Done** ❌

You're using `forge_no_abstract_not_flattened` - contracts still have unresolved imports!

---

## ✅ FIXES (In Order of Impact)

### Fix 1: Use Larger Dataset ⭐ **DO THIS FIRST**

**Current**: combined_labeled (228 contracts)
**Change to**: forge_no_abstract_not_flattened (3,746 contracts)

```bash
# Train on the filtered dataset directly (without flattening first - let's test)
./start_training.sh static \
    --train-dir data/datasets/forge_no_abstract_not_flattened/train \
    --val-dir data/datasets/forge_no_abstract_not_flattened/val \
    --test-dir data/datasets/forge_no_abstract_not_flattened/test
```

**Expected improvement**: 11% → 30-40% (just from more data)

---

### Fix 2: Flatten Contracts FIRST ⭐ **CRITICAL**

**Problem**: Contracts have imports that aren't resolved
```solidity
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";  // ← Can't compile!
```

**Solution**: Flatten BEFORE training

```bash
# Flatten all splits
for split in train val test; do
  python scripts/dataset/flatten_contracts.py \
      data/datasets/forge_no_abstract_not_flattened/$split \
      --output data/datasets/forge_flattened/$split \
      --tool simple \
      --batch
done

# Then train on flattened
./start_training.sh static \
    --train-dir data/datasets/forge_flattened/train \
    --val-dir data/datasets/forge_flattened/val \
    --test-dir data/datasets/forge_flattened/test
```

**About Flattening**:
- **"Flattening" = Resolving all imports into ONE file**
- Example:

**Before Flattening** (Won't compile):
```solidity
// MyToken.sol
import "./SafeMath.sol";  // ← Missing!

contract MyToken {
    using SafeMath for uint256;
    // ...
}
```

**After Flattening** (Will compile):
```solidity
// MyToken_flattened.sol
library SafeMath {  // ← Inlined from import!
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        return a + b;
    }
}

contract MyToken {
    using SafeMath for uint256;
    // ...
}
```

**Expected improvement**: 40% → 55-65%

---

### Fix 3: Improve PDG Extraction ⭐ **ADVANCED**

**Current issue**: PDG only has 3-10 nodes because it only extracts:
- Function names
- State variables
- High-level calls

**What's missing**:
- Individual statements within functions
- Expression-level data flow
- Control flow (if/else/loops)

**Solution**: Enhance Slither wrapper to extract **statement-level** PDG

This requires modifying `tools/slither_wrapper.py` to use Slither's CFG (Control Flow Graph) and add individual nodes.

**Expected improvement**: 65% → 75-85%

---

## 🚀 **QUICK FIX** (Do This NOW!)

### Step 1: Use Larger Dataset (5 minutes)

```bash
# Train on forge_no_abstract_not_flattened (3,746 contracts)
./start_training.sh static \
    --train-dir data/datasets/forge_no_abstract_not_flattened/train \
    --val-dir data/datasets/forge_no_abstract_not_flattened/val \
    --test-dir data/datasets/forge_no_abstract_not_flattened/test \
    --batch-size 8 \
    --num-epochs 10 \
    --max-samples 500
```

**Why `--max-samples 500`**: Quick test with 500 contracts to see if it improves

**Expected result**:
- If accuracy improves to 25-35%, dataset size was the issue ✅
- If still ~11%, PDG extraction is broken ❌

---

### Step 2: If Step 1 Works, Flatten Dataset (30 minutes)

```bash
# Flatten train/val/test
for split in train val test; do
  echo "Flattening $split..."
  python scripts/dataset/flatten_contracts.py \
      data/datasets/forge_no_abstract_not_flattened/$split \
      --output data/datasets/forge_flattened/$split \
      --tool simple \
      --batch
done

# Train on flattened
./start_training.sh static \
    --train-dir data/datasets/forge_flattened/train \
    --val-dir data/datasets/forge_flattened/val \
    --test-dir data/datasets/forge_flattened/test \
    --batch-size 16 \
    --num-epochs 50
```

**Expected**: 55-70% accuracy

---

### Step 3: If Still Low, Check PDG Quality

```bash
# Test PDG extraction on a few contracts
python3 << 'EOF'
from tools.slither_wrapper import SlitherWrapper
from pathlib import Path

slither = SlitherWrapper()

# Test on a reentrancy contract
contracts = list(Path("data/datasets/forge_no_abstract_not_flattened/train/reentrancy").glob("*.sol"))[:5]

for contract_file in contracts:
    with open(contract_file) as f:
        source = f.read()

    result = slither.analyze_contract(source)
    pdg = result.get('pdg')

    print(f"\n{contract_file.name}:")
    print(f"  Nodes: {pdg.number_of_nodes()}")
    print(f"  Edges: {pdg.number_of_edges()}")
    print(f"  Node types: {set(d.get('type') for n, d in pdg.nodes(data=True))}")

    if pdg.number_of_nodes() < 20:
        print(f"  ⚠️ WARNING: PDG too small!")
        # Print the contract to see why
        print(f"  Contract length: {len(source)} characters")
        print(f"  First 200 chars: {source[:200]}")
EOF
```

**If PDGs are still tiny (<20 nodes)**: PDG extraction needs fixing (Step 4)

---

### Step 4: Fix PDG Extraction (Advanced - 1-2 hours)

If PDGs are consistently small, the issue is in `tools/slither_wrapper.py`.

**Problem**: Line 154-201 only extracts high-level structure, not detailed CFG

**Solution**: Use Slither's CFG nodes to build detailed PDG

I can help you implement this if needed!

---

## 📊 Expected Accuracy Progression

| Fix | Dataset | Flattened | PDG Detail | Expected Accuracy |
|-----|---------|-----------|------------|-------------------|
| **Current** | combined_labeled (228) | ❌ No | Low | **11%** (random) |
| **Step 1** | forge_no_abstract (3,746) | ❌ No | Low | **30-40%** |
| **Step 2** | forge_no_abstract (3,746) | ✅ Yes | Low | **55-65%** |
| **Step 3** | forge_no_abstract (3,746) | ✅ Yes | ⭐ High | **75-85%** |

---

## 🎯 **DO THIS NOW** (Prioritized)

### Immediate (5 minutes):
```bash
# Quick test with larger dataset
./start_training.sh static \
    --train-dir data/datasets/forge_no_abstract_not_flattened/train \
    --val-dir data/datasets/forge_no_abstract_not_flattened/val \
    --test-dir data/datasets/forge_no_abstract_not_flattened/test \
    --max-samples 500 \
    --num-epochs 10
```

**Watch the training output**: If accuracy goes above 25%, you're on the right track!

### After Quick Test (30 minutes):
```bash
# Flatten dataset
for split in train val test; do
  python scripts/dataset/flatten_contracts.py \
      data/datasets/forge_no_abstract_not_flattened/$split \
      --output data/datasets/forge_flattened/$split \
      --tool simple \
      --batch
done
```

### Full Training (2-3 hours):
```bash
# Train on flattened dataset
./start_training.sh static \
    --train-dir data/datasets/forge_flattened/train \
    --val-dir data/datasets/forge_flattened/val \
    --test-dir data/datasets/forge_flattened/test \
    --batch-size 16 \
    --num-epochs 50
```

---

## 🔍 To Answer Your Flattening Question:

**"Flattening" means**:
1. Read the main contract file
2. Find all `import` statements
3. Read each imported file
4. **Combine everything into ONE file** (removing import statements)
5. Resolve all dependencies
6. Output a single `.sol` file with ALL code

**Result**: A contract that compiles **without any external dependencies**

**Example**:

**Original** (3 files):
```
SafeMath.sol
Utils.sol
MyToken.sol (imports SafeMath and Utils)
```

**Flattened** (1 file):
```
MyToken_flattened.sol (contains SafeMath + Utils + MyToken all in one)
```

**Why we need it**:
- Slither needs contracts to **compile**
- Contracts with unresolved imports = **compilation fails**
- Compilation fails = **no PDG** = **model can't learn**

---

## Summary

**Root cause**:
1. ❌ Dataset too small (228 contracts)
2. ❌ Contracts not flattened (imports fail)
3. ❌ PDG extraction too shallow (3-10 nodes)

**Fix priority**:
1. ⭐ Use larger dataset (NOW!)
2. ⭐ Flatten contracts (30 min)
3. ⭐ Improve PDG extraction (if still needed)

**Start NOW with**:
```bash
./start_training.sh static \
    --train-dir data/datasets/forge_no_abstract_not_flattened/train \
    --val-dir data/datasets/forge_no_abstract_not_flattened/val \
    --test-dir data/datasets/forge_no_abstract_not_flattened/test \
    --max-samples 500 \
    --num-epochs 10
```

This will tell you if the dataset size is the issue!
