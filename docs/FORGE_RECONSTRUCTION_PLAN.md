# FORGE Dataset Reconstruction Plan
## Proper Flattening from Source

---

## 📊 Current FORGE-Artifacts Structure

```
FORGE-Artifacts/dataset/
├── contracts/               (6,618 project folders)
│   ├── $joke/
│   │   └── JOKECOMMUNITY/
│   │       ├── JOKECOMMUNITY.sol        ← Main contract (1,241 lines)
│   │       └── (no dependencies - already flat!)
│   │
│   ├── $tkt/
│   │   └── ToonKing/
│   │       ├── ToonKing.sol             ← Main contract
│   │       ├── Context.sol              ← Dependency
│   │       ├── DividendPayingToken.sol  ← Dependency
│   │       ├── IERC20.sol               ← Interface
│   │       ├── IUniswapV2Router.sol     ← Interface
│   │       ├── Ownable.sol              ← Dependency
│   │       ├── SafeMath.sol             ← Dependency
│   │       └── ... (20+ files)
│   │
│   └── ... (6,616 more projects)
│
└── results/                 (6,454 audit JSON files)
    ├── $joke.pdf.json       ← Audit report with CWE codes
    ├── $tkt.pdf.json
    └── ...
```

### Key Observations:

1. **Some contracts are already flat** (single .sol file, no imports)
2. **Some have dependencies** (multiple .sol files in same folder)
3. **Each project folder = 1 audit report** (same name)
4. **Audit JSONs have CWE codes** → vulnerability labels

---

## 🎯 Reconstruction Workflow

### Phase 1: Flatten Each Project ⭐
```
For each project folder in FORGE-Artifacts/dataset/contracts/:
  1. Check if project has multiple .sol files
  2. If YES → Flatten the main contract
  3. If NO → Copy as-is (already flat)
  4. Output → forge_flattened_all/ProjectName.sol
```

### Phase 2: Map CWE → Vulnerability Class
```
For each audit JSON in FORGE-Artifacts/dataset/results/:
  1. Read CWE codes from findings
  2. Map to vulnerability class (using CWE_TO_CLASS mapping)
  3. Find corresponding flattened contract
  4. Label → ProjectName.sol belongs to "reentrancy" class
```

### Phase 3: Filter Bad Contracts
```
For each flattened contract:
  1. Check if interface/abstract
  2. Check if too small (<10 lines)
  3. Check if no implementations
  4. Remove if bad quality
```

### Phase 4: Balance & Organize
```
For each vulnerability class:
  1. Sample N contracts per class
  2. Create train/val/test split (70/15/15)
  3. Output → forge_reconstructed/train/reentrancy/*.sol
```

---

## 📝 Detailed Steps

### Step 1: Identify Main Contract File

**Problem**: Some projects have 20+ .sol files - which is the main one?

**Solution**: Use audit JSON `project_path`:
```json
"project_path": {
    "JOKECOMMUNITY": "contracts/$joke/JOKECOMMUNITY"
                      ↑                     ↑
                Main contract name     Project path
}
```

**Logic**:
```python
# From audit JSON
main_contract_name = "JOKECOMMUNITY"
project_path = "contracts/$joke/JOKECOMMUNITY"

# Find main file
main_file = f"{project_path}/{main_contract_name}.sol"
# → "contracts/$joke/JOKECOMMUNITY/JOKECOMMUNITY.sol"
```

---

### Step 2: Flatten the Project

**For projects with dependencies**:
```bash
# Example: $tkt/ToonKing (has 20+ files)
cd FORGE-Artifacts/dataset/contracts/$tkt/ToonKing/

# Flatten ToonKing.sol (includes all imports from same folder)
forge flatten ToonKing.sol -o ToonKing_flattened.sol
```

**For projects already flat** (single file):
```bash
# Example: $joke/JOKECOMMUNITY (only 1 file, no imports)
# Just copy as-is, no flattening needed
```

**Tool options**:
- `forge flatten` (best - handles local imports)
- `truffle-flattener`
- `hardhat flatten`
- Custom script (recursively resolve imports)

---

### Step 3: Extract Vulnerability Labels from Audit JSONs

**From audit JSON → vulnerability class**:

```python
# Read audit JSON
audit = json.load("$joke.pdf.json")

# Extract all CWEs
all_cwes = []
for finding in audit["findings"]:
    cwes = finding["category"]["1"]  # ["CWE-284", "CWE-269"]
    all_cwes.extend(cwes)

# Map to vulnerability class (using priority order)
CWE_TO_CLASS = {
    'CWE-284': 'access_control',
    'CWE-362': 'reentrancy',
    'CWE-682': 'arithmetic',
    ...
}

PRIORITY = [
    'reentrancy',      # Highest priority
    'arithmetic',
    'bad_randomness',
    ...
    'access_control',
    'other'            # Lowest priority
]

# Find highest priority vulnerability
vuln_class = None
for priority_vuln in PRIORITY:
    for cwe in all_cwes:
        if CWE_TO_CLASS.get(cwe) == priority_vuln:
            vuln_class = priority_vuln
            break
    if vuln_class:
        break

# Result: $joke → "access_control" (because CWE-284)
```

---

### Step 4: Filter Out Bad Contracts

**After flattening**, filter using same logic as before:

```python
is_bad = (
    is_interface(contract) or
    is_abstract_no_impl(contract) or
    is_too_small(contract) or
    has_no_implementations(contract)
)

if is_bad:
    skip_contract()
```

**Expected removal**: ~40-50% (interfaces, abstracts, tiny files)

---

### Step 5: Balance Dataset

**Sample per class**:
```python
samples_per_class = {
    'safe': 1000,
    'access_control': 1000,
    'arithmetic': 1000,
    'reentrancy': 800,
    'unchecked_low_level_calls': 1000,
    'denial_of_service': 500,
    'other': 1000,
    ...
}
```

**Split 70/15/15**:
```
Train: 70% of samples
Val:   15% of samples
Test:  15% of samples
```

---

## 🗂️ Final Output Structure

```
forge_reconstructed/
├── train/
│   ├── access_control/         700 contracts (flattened)
│   ├── arithmetic/             700 contracts (flattened)
│   ├── reentrancy/             560 contracts (flattened)
│   ├── safe/                   700 contracts (flattened)
│   └── ... (11 classes)
├── val/
│   └── ... (same structure, 15%)
├── test/
│   └── ... (same structure, 15%)
└── dataset_summary.json
```

**Key difference from before**:
- ✅ **ALL contracts fully flattened** (imports resolved)
- ✅ **Higher quality** (proper flattening with all dependencies)
- ✅ **No missing imports** (everything in one file)

---

## 📊 Expected Results

| Metric | Before (Current) | After (Reconstructed) |
|--------|-----------------|---------------------|
| **Total projects** | 6,618 | 6,618 |
| **After flattening** | N/A (not done) | ~6,200 (94% success) |
| **After filtering** | 3,746 | ~3,500-4,000 |
| **Flattening quality** | ❌ Not flattened | ✅ **Fully flattened** |
| **PDG success rate** | 20-30% (imports fail) | **80-90%** |
| **Training accuracy** | 11% (broken) | **60-75%** (working!) |

---

## ⏱️ Time Estimates

| Phase | Time |
|-------|------|
| **Phase 1**: Flatten 6,618 projects | 2-3 hours |
| **Phase 2**: Map CWE → labels | 10 minutes |
| **Phase 3**: Filter bad contracts | 30 minutes |
| **Phase 4**: Balance & organize | 15 minutes |
| **Total** | **3-4 hours** |

---

## 🛠️ Tools Needed

### Option A: Use Forge (Recommended)
```bash
# Install Foundry (includes forge)
curl -L https://foundry.paradigm.xyz | bash
foundryup

# Test flattening
cd FORGE-Artifacts/dataset/contracts/\$tkt/ToonKing/
forge flatten ToonKing.sol
```

**Pros**:
- ✅ Best tool for Solidity flattening
- ✅ Handles complex imports
- ✅ Maintains license comments
- ✅ Fast

### Option B: Use Truffle Flattener
```bash
npm install -g truffle-flattener

# Test
truffle-flattener ToonKing.sol > ToonKing_flat.sol
```

### Option C: Custom Python Script
```python
# Recursively resolve imports
# Read each import statement
# Copy imported file content
# Combine into one file
```

---

## 🚀 Implementation Strategy

### Quick Test First (30 minutes):
```bash
# Test on 10 projects
python scripts/dataset/reconstruct_forge_dataset.py \
    --forge-dir data/datasets/FORGE-Artifacts \
    --output-dir data/datasets/forge_reconstructed_test \
    --max-projects 10 \
    --tool forge
```

**Check**:
- Flattening success rate > 80%? ✅ Continue
- Flattening success rate < 60%? ⚠️ Try different tool

### Full Reconstruction (3-4 hours):
```bash
# Process all 6,618 projects
python scripts/dataset/reconstruct_forge_dataset.py \
    --forge-dir data/datasets/FORGE-Artifacts \
    --output-dir data/datasets/forge_reconstructed \
    --tool forge \
    --samples-per-class access_control:1000,arithmetic:1000,reentrancy:800,...
```

---

## ✅ Advantages of This Approach

1. **Proper Flattening**
   - All dependencies available (same project folder)
   - Flattening works correctly
   - No missing imports

2. **Better Quality**
   - Start from source (FORGE)
   - Use proper CWE labeling from audits
   - Filter after flattening (better decisions)

3. **Reproducible**
   - Can recreate anytime
   - Change sampling easily
   - Adjust CWE mapping if needed

4. **Expected Success**
   - 80-90% flattening success (vs 60-70% current)
   - 80-90% PDG extraction (vs 20-30% current)
   - 60-75% training accuracy (vs 11% current!)

---

## ❓ Next Steps / Questions

1. **Which flattening tool should we use?**
   - Forge (recommended - best for Solidity)
   - Truffle
   - Custom script

2. **Should we test on 10 projects first?**
   - Quick validation (30 min)
   - Or go straight to full reconstruction (3-4 hours)?

3. **Which samples per class?**
   - Use same as before (1000/800/500/...)?
   - Or different distribution?

4. **Keep intermediate files?**
   - Save flattened files before filtering?
   - Or only keep final organized dataset?

---

## 🎯 My Recommendation

**Test-driven approach**:
1. ✅ **Install Forge** (best tool)
2. ✅ **Test on 10 projects** (verify flattening works)
3. ✅ **Check success rate** (should be >80%)
4. ✅ **If good, run full reconstruction** (3-4 hours)
5. ✅ **Train on reconstructed dataset** (expect 60-75% accuracy!)

**What do you think?** Should we:
- Start with test (10 projects)?
- Create the reconstruction script?
- Install Forge first?

Let me know! 🚀
