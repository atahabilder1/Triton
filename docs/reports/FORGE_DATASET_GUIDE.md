# FORGE Dataset - Complete Guide

## **Where is the Data?**

```
📂 /data/llm_projects/triton_datasets/FORGE-Artifacts/dataset/
```

## **How is it Organized?**

### **NOT like your current dataset!**

Your current dataset (`combined_labeled`):
```
combined_labeled/
├── access_control/     ← Folder per vulnerability type
│   ├── contract1.sol
│   └── contract2.sol
├── reentrancy/
│   └── contract3.sol
└── ...
```

### **FORGE dataset structure:**

```
FORGE-Artifacts/dataset/
├── contracts/                    ← 6,616 project folders
│   ├── $joke/                   ← One audit report
│   │   └── JOKECOMMUNITY/       ← Contract name
│   │       └── JOKECOMMUNITY.sol  ← Solidity file
│   │
│   ├── $PePe coin Full Smart Contract Security Audit/
│   │   └── $PePe/
│   │       └── $PePe.sol
│   │
│   └── [6,614 more projects...]
│
└── results/                      ← 6,454 JSON label files
    ├── $joke.pdf.json           ← Labels for $joke project
    ├── $PePe coin Full Smart Contract Security Audit.pdf.json
    └── [6,452 more labels...]
```

**Key Point:** Each audit report becomes:
- 1 folder in `contracts/` (with .sol files)
- 1 JSON file in `results/` (with vulnerability labels)

---

## **What's in the JSON Files?**

### **Example:** `results/$joke.pdf.json`

```json
{
  "path": "artifacts/$joke.pdf",
  "project_info": {
    "address": "0x2df0c13487efdf4eb7f6c042273b7aca781b29a0",
    "chain": "bsc",
    "compiler_version": ["v0.6.12+commit.27d51765"],
    "project_path": {
      "JOKECOMMUNITY": "contracts/$joke/JOKECOMMUNITY"  ← Link to .sol files
    }
  },
  "findings": [                    ← Vulnerability list
    {
      "id": 0,
      "category": {
        "1": ["CWE-284"],          ← Primary CWE
        "2": ["CWE-269"]           ← Secondary CWE
      },
      "title": "Stop Transactions with Locked Flag",
      "description": "The contract owner has authority to stop...",
      "severity": "critical",
      "location": "https://bscscan.com/address/0x2df0...#L1234"
    },
    {
      "id": 1,
      "category": {
        "1": ["CWE-284"],
        "2": ["CWE-269"]
      },
      "title": "Stop Transactions with Max Amount",
      "severity": "critical"
    }
  ]
}
```

---

## **Total Classes: 296 CWE Categories!**

### **Top CWE Categories (from sample of 100 projects):**

```
CWE-710:  234 findings  → Coding Standard Violation
CWE-284:  123 findings  → Improper Access Control ✓ (maps to access_control)
CWE-682:   83 findings  → Incorrect Calculation ✓ (maps to arithmetic)
CWE-269:   68 findings  → Improper Privilege Management ✓ (maps to access_control)
CWE-703:   59 findings  → Improper Check/Handling ✓ (maps to unchecked_calls)
CWE-754:   43 findings  → Improper Check for Unusual Conditions
CWE-252:   18 findings  → Unchecked Return Value ✓ (maps to unchecked_calls)
CWE-190:    5 findings  → Integer Overflow ✓ (maps to arithmetic)
...
```

### **Your 10 Vulnerability Types vs FORGE's 296 CWEs:**

You need to MAP CWEs to your types:

```
YOUR TYPE              CWE CODES TO SEARCH FOR
─────────────────────  ───────────────────────────────────
access_control      →  CWE-284, CWE-269, CWE-732, CWE-285, CWE-862
arithmetic          →  CWE-190, CWE-191, CWE-682, CWE-369
reentrancy          →  CWE-841, CWE-362, CWE-667
unchecked_calls     →  CWE-252, CWE-703, CWE-476
bad_randomness      →  CWE-330, CWE-338
denial_of_service   →  CWE-400, CWE-835, CWE-770
front_running       →  CWE-362 (overlaps with reentrancy)
time_manipulation   →  CWE-829, CWE-347
short_addresses     →  (rare, may not have CWE mapping)
safe                →  Projects with NO findings
```

---

## **How to Browse the Data Yourself?**

### **1. Look at a JSON label file:**

```bash
cd /data/llm_projects/triton_datasets/FORGE-Artifacts/dataset
cat results/\$joke.pdf.json | python3 -m json.tool | less
```

### **2. Look at corresponding contract:**

```bash
cat contracts/\$joke/JOKECOMMUNITY/JOKECOMMUNITY.sol | less
```

### **3. Count contracts by CWE:**

```bash
cd results
grep -l "CWE-284" *.json | wc -l    # Count projects with access control issues
grep -l "CWE-190" *.json | wc -l    # Count projects with integer overflow
```

### **4. See all CWE categories:**

```bash
cd results
grep -oh "CWE-[0-9]*" *.json | sort -u | wc -l   # Count unique CWEs
```

---

## **Statistics:**

```
Total Projects:      6,616
Total .sol files:    78,223
Total JSON labels:   6,454
Total findings:      ~27,497 (across all projects)
Unique CWEs:         296 categories
Severities:          critical, high, medium, low
```

---

## **What You Need to Do:**

### **Option 1: Use My Script (Easiest)**

I'll create `scripts/create_forge_training_set.py` that:
1. Reads all JSON files
2. Maps CWE codes → your 10 types
3. Copies contracts to organized folders like:
   ```
   forge_balanced/
   ├── train/
   │   ├── access_control/
   │   ├── reentrancy/
   │   └── ...
   └── test/
   ```

### **Option 2: Manual Exploration**

```bash
# Browse projects
cd /data/llm_projects/triton_datasets/FORGE-Artifacts/dataset/contracts
ls | head -20

# Browse labels
cd ../results
ls *.json | head -20

# Read a specific project
cat "\$joke.pdf.json" | python3 -m json.tool
cat "../contracts/\$joke/JOKECOMMUNITY/JOKECOMMUNITY.sol"
```

---

## **Key Differences from Your Current Dataset:**

| Feature | Your Current Dataset | FORGE Dataset |
|---------|---------------------|---------------|
| **Organization** | By vulnerability type | By audit report |
| **Classes** | 10 types (predefined) | 296 CWE codes |
| **Size** | 228 contracts | 78,223 contracts |
| **Labels** | Folder name = type | JSON with CWE codes |
| **Format** | contract.sol in type folder | contract in project/module folders |
| **Ready to train** | ✅ Yes | ❌ Need preprocessing |

---

## **Next Steps:**

1. **Run analysis script:**
   ```bash
   python scripts/analyze_forge_dataset.py
   ```

2. **I'll create preprocessing script** to:
   - Map 296 CWEs → your 10 types
   - Extract balanced samples
   - Organize like your current dataset

3. **Train with 10x more data!**

---

## **Quick Navigation:**

**Contracts:** `/data/llm_projects/triton_datasets/FORGE-Artifacts/dataset/contracts/`
**Labels:** `/data/llm_projects/triton_datasets/FORGE-Artifacts/dataset/results/`
**Size:** 736MB contracts + 32MB labels = **768MB total**

**Bottom Line:** Data is there, just needs organization!
