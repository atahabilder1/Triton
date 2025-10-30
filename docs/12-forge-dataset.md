# FORGE Dataset Integration Guide

## 📚 About FORGE Dataset

**Paper:** "FORGE: An LLM-driven Framework for Large-Scale Smart Contract Vulnerability Dataset Construction"
**Published:** ICSE 2026 (Top-tier Software Engineering Conference)
**Authors:** Jiachi Chen et al., Sun Yat-sen University + Peking University + others

### 🎯 Why FORGE is Amazing

FORGE is the **first automated LLM-driven framework** for constructing smart contract vulnerability datasets from real-world audit reports.

**Key Innovations:**
1. **Automated Construction:** Uses LLMs to extract vulnerabilities from audit PDFs
2. **CWE Classification:** Maps to 296 CWE categories (standardized)
3. **Large Scale:** 81,390 Solidity files, 27,497 vulnerabilities
4. **High Quality:** 95.6% precision, k-α = 0.87 (high inter-rater agreement)
5. **Real-World:** From 6,454 professional audit reports

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Audit Reports** | 6,454 |
| **Total DApp Projects** | 6,579 |
| **Total Solidity Files** | 81,390 |
| **Total Vulnerability Findings** | 27,497 |
| **CWE Categories** | 296 |
| **Average Files per Project** | 12 |
| **Average Lines of Code per Project** | 2,575 |

### Compiler Versions

| Version | Projects |
|---------|----------|
| 0.4.x | 270 |
| 0.5.x | 478 |
| 0.6.x | 1,524 |
| 0.7.x | 360 |
| **0.8.x** | **3,791** (59% - Latest!) |
| Other | 31 |

## 📍 Dataset Location

```
/home/anik/code/Triton/data/datasets/FORGE-Artifacts/
├── dataset/
│   ├── contracts/          # 81,390 Solidity files
│   │   ├── ProjectA/       # Organized by audit report
│   │   ├── ProjectB/
│   │   └── ...
│   └── results/            # 6,454 JSON files with vulnerability info
│       ├── report1.pdf.json
│       ├── report2.pdf.json
│       └── ...
├── evaluation/             # Benchmark results for 13 tools
└── src/                    # FORGE framework source code
```

## 🔍 Understanding the Data Format

### Vulnerability JSON Structure

Each `.json` file corresponds to an audit report and contains:

```json
{
    "path": "artifacts/RocketPool.pdf",
    "project_info": {
        "url": "https://github.com/rocket-pool/rocketpool",
        "commit_id": "a65b203...",
        "address": "n/a",
        "chain": "eth",
        "compiler_version": ["v0.8.0+commit..."],
        "project_path": {
            "rocketpool": "contracts/RocketPool/rocketpool"
        }
    },
    "findings": [
        {
            "id": 0,
            "category": {
                "1": ["CWE-284"],        # Level 1: Pillar
                "2": ["CWE-269"],        # Level 2: Class
                "3": ["CWE-267"]         # Level 3: Base
            },
            "title": "Any network contract can change withdrawal address",
            "description": "RocketStorage uses eternal storage pattern...",
            "severity": "high",         # critical, high, medium, low, info
            "location": "RocketStorage.sol"
        }
    ]
}
```

### CWE Hierarchy

FORGE uses hierarchical CWE classification:
- **Level 1:** Pillar (e.g., CWE-284: Access Control)
- **Level 2:** Class (e.g., CWE-269: Improper Privilege Management)
- **Level 3:** Base (e.g., CWE-267: Privilege Defined With Unsafe Actions)

## 🚀 How to Test Triton on FORGE

### Quick Summary Stats

```bash
cd /home/anik/code/Triton/data/datasets/FORGE-Artifacts

# Count total contracts
find dataset/contracts -name "*.sol" | wc -l
# Output: 78,224 (some files may be missing)

# Count projects
ls dataset/contracts/ | wc -l
# Output: 6,579

# Count vulnerability findings
find dataset/results -name "*.json" -exec cat {} \; | grep -o '"id":' | wc -l
# Output: 27,497
```

### Method 1: Test on Specific CWE Categories

FORGE is organized by audit reports, not by vulnerability type. You'll need to extract contracts by CWE category first.

Let me create a script for you to do this.

### Method 2: Test on Random Sample

Test on a random sample of contracts:

```bash
# Get 100 random contracts
find /home/anik/code/Triton/data/datasets/FORGE-Artifacts/dataset/contracts \
    -name "*.sol" | shuf | head -100 > forge_sample_100.txt

# Test with Triton
python scripts/test_triton.py --dataset custom \
    --custom-dir /home/anik/code/Triton/data/datasets/FORGE-Artifacts/dataset/contracts \
    --output-dir results/forge
```

### Method 3: Test Specific Projects

Pick high-quality projects from the dataset:

```bash
# Test RocketPool (if available)
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/FORGE-Artifacts/dataset/contracts/RocketPool \
    --output-dir results/forge

# Test a specific project
ls data/datasets/FORGE-Artifacts/dataset/contracts/ | head -10
# Pick one and test
```

## 📊 Comparing with Other Datasets

| Dataset | Contracts | Vulnerabilities | Source | Organization | Quality |
|---------|-----------|----------------|--------|--------------|---------|
| **FORGE** | **81,390** | **27,497** | Real audits | By project | **95.6% precision** |
| SmartBugs Curated | 143 | 143 | Known vulns | By vuln type | Manual annotation |
| SolidiFI | 9,369 | 9,369 | Injected | By vuln type | Synthetic |
| SmartBugs Wild | 47,398 | Unknown | Etherscan | Random | No ground truth |

**FORGE Advantages:**
- ✅ Largest dataset (81K+ files)
- ✅ Real-world vulnerabilities from professional audits
- ✅ Standardized CWE classification
- ✅ High-quality automated extraction (95.6% precision)
- ✅ Includes severity levels
- ✅ Multiple compiler versions (mostly 0.8.x)

**FORGE Challenges for Testing:**
- ⚠️ Organized by project, not vulnerability type
- ⚠️ Need to parse JSON to get vulnerability info
- ⚠️ Very large (may need sampling)
- ⚠️ Some contracts may be complex (avg 2,575 LOC)

## 🛠️ Integration with Triton

I'll create a script to help you use FORGE with Triton. The script will:

1. Parse FORGE JSON files
2. Extract vulnerability ground truth
3. Map CWE categories to Triton's vulnerability types
4. Run Triton on selected contracts
5. Compare results with FORGE ground truth

## 📈 Expected Use Cases

### Use Case 1: Large-Scale Evaluation

Test Triton on FORGE to:
- Evaluate on 81K+ real-world contracts
- Compare with 13 existing tools (FORGE benchmarked them)
- Get comprehensive CWE category coverage
- Validate on latest Solidity versions (0.8.x)

### Use Case 2: Specific Vulnerability Testing

Extract contracts with specific CWE categories:
- CWE-284: Access Control (most common)
- CWE-703: Improper Input Validation
- CWE-691: Insufficient Control Flow Management
- etc.

### Use Case 3: Severity-Based Testing

Test on different severity levels:
- Critical vulnerabilities only
- High + Critical
- All severities

### Use Case 4: Compiler Version Testing

Test Triton's performance across different Solidity versions:
- Modern contracts (0.8.x) - 59% of dataset
- Legacy contracts (0.4.x - 0.7.x)

## 🎯 Recommended Testing Strategy

### Phase 1: Small Sample (1 hour)
```bash
# Test 50 random contracts to validate setup
# Expected: ~50 contracts in 10-15 minutes
```

### Phase 2: CWE-Specific Testing (2-3 hours)
```bash
# Extract and test contracts with common CWEs
# - CWE-284 (Access Control)
# - CWE-703 (Input Validation)
# - CWE-691 (Control Flow)
```

### Phase 3: Severity-Based Testing (3-4 hours)
```bash
# Test all high + critical vulnerabilities
# Expected: ~5,000-10,000 contracts
```

### Phase 4: Full Dataset (Days)
```bash
# Test all 81,390 contracts
# Will take days, use for final evaluation
```

## 📊 CWE to Triton Mapping

Triton detects 10 vulnerability types. Here's how they map to CWE:

| Triton Type | CWE Categories |
|-------------|----------------|
| **reentrancy** | CWE-841, CWE-663 |
| **overflow** | CWE-190 |
| **underflow** | CWE-191 |
| **access_control** | CWE-284, CWE-269, CWE-732 |
| **unchecked_call** | CWE-252, CWE-703 |
| **timestamp_dependency** | CWE-829 |
| **tx_origin** | CWE-477 |
| **delegatecall** | CWE-829 |
| **self_destruct** | CWE-1045 |
| **gas_limit** | CWE-400, CWE-770 |

## 🔧 Next Steps

I'll create a script that:
1. Parses FORGE JSON files
2. Extracts contracts with vulnerabilities
3. Maps CWEs to Triton types
4. Runs Triton and compares results

Would you like me to create this integration script now?

## 📚 References

**FORGE Paper:**
```bibtex
@misc{chen2025forgellmdrivenframeworklargescale,
      title={FORGE: An LLM-driven Framework for Large-Scale Smart Contract Vulnerability Dataset Construction},
      author={Jiachi Chen and Yiming Shen and Jiashuo Zhang and Zihao Li and John Grundy and Zhenzhe Shao and Yanlin Wang and Jiashui Wang and Ting Chen and Zibin Zheng},
      year={2025},
      eprint={2506.18795},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2506.18795},
}
```

**Dataset:**
- GitHub: https://github.com/shenyimings/FORGE-Artifacts
- Location: `/home/anik/code/Triton/data/datasets/FORGE-Artifacts/`

---

**Generated:** 2025-10-30
**Status:** FORGE dataset downloaded and ready for use
