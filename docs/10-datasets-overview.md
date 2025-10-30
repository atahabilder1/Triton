# Triton Testing Datasets - Complete Documentation

## Overview

This document describes all the benchmark datasets used for testing Triton's smart contract vulnerability detection capabilities. Each dataset has been carefully selected for its quality, coverage, and relevance to smart contract security research.

---

## Dataset 1: SmartBugs Curated

### Description
SmartBugs Curated is a manually curated dataset of vulnerable Solidity smart contracts organized according to the DASP taxonomy. It represents the **most widely used benchmark dataset** in smart contract security research.

### Statistics
- **Total Contracts:** 143
- **Vulnerability Types:** 10 (DASP categories)
- **Source:** Real-world contracts from Ethereum mainnet (verified on Etherscan)
- **Organization:** By vulnerability type
- **Quality:** Manually annotated with line-by-line ground truth

### Vulnerability Categories
1. Reentrancy (31 contracts)
2. Access Control (18 contracts)
3. Arithmetic (15 contracts)
4. Unchecked Low Level Calls (52 contracts)
5. Denial of Service (6 contracts)
6. Bad Randomness (8 contracts)
7. Front Running (4 contracts)
8. Time Manipulation (5 contracts)
9. Short Addresses (1 contract)
10. Other (3 contracts)

### Citation
```bibtex
@inproceedings{durieux2020empirical,
  title={Empirical Review of Automated Analysis Tools on 47,587 Ethereum Smart Contracts},
  author={Durieux, Thomas and Ferreira, Jo{\~a}o F and Abreu, Rui and Cruz, Pedro},
  booktitle={Proceedings of the ACM/IEEE 42nd International Conference on Software Engineering},
  pages={530--541},
  year={2020},
  organization={ACM}
}
```

### Authors & Affiliation
- **Authors:** Thomas Durieux, João F. Ferreira, Rui Abreu, Pedro Cruz
- **Institution:** University of Luxembourg, IST Portugal
- **Conference:** ICSE 2020 (International Conference on Software Engineering)

### Location in Triton
```
/home/anik/code/Triton/data/datasets/smartbugs-curated/
```

### Usage
```bash
# Test on SmartBugs Curated
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs-curated/dataset
```

### Strengths
- ✅ Well-organized by vulnerability type
- ✅ Manually verified and annotated
- ✅ Line-by-line vulnerability locations
- ✅ Widely used in research (reproducible results)
- ✅ Real-world contracts

### Limitations
- ⚠️ Small dataset (143 contracts)
- ⚠️ Limited diversity (10 vulnerability types)
- ⚠️ Older Solidity versions

---

## Dataset 2: FORGE Artifacts

### Description
**FORGE is an LLM-driven framework** that automatically constructs large-scale smart contract vulnerability datasets from real-world audit reports. It is the **first automated approach** for dataset construction in this domain and produces the **largest and most comprehensive** vulnerability dataset to date.

⚠️ **IMPORTANT NOTE:** This dataset was **artificially created by researchers using automated LLM extraction** from audit PDFs. The vulnerabilities are real (from professional audits), but the extraction and classification process is automated.

### Statistics
- **Total Audit Reports:** 6,454 professional security audits
- **Total DApp Projects:** 6,579
- **Total Solidity Files:** 81,390
- **Total Vulnerability Findings:** 27,497
- **CWE Categories:** 296 (standardized classification)
- **Average Files per Project:** 12
- **Average Lines of Code per Project:** 2,575

### Compiler Version Distribution
| Version | Projects | Percentage |
|---------|----------|------------|
| 0.8.x | 3,791 | 59.0% (Latest) |
| 0.6.x | 1,524 | 23.7% |
| 0.5.x | 478 | 7.4% |
| 0.7.x | 360 | 5.6% |
| 0.4.x | 270 | 4.2% |
| Other | 31 | 0.5% |

### Methodology
FORGE uses a 4-module automated pipeline:
1. **Semantic Chunker:** Segments audit reports into meaningful chunks
2. **MapReduce Extractor:** Extracts vulnerability information from chunks
3. **Hierarchical Classifier:** Classifies vulnerabilities into CWE hierarchy using tree-of-thoughts reasoning
4. **Code Fetcher:** Retrieves source code from GitHub, Etherscan, BSCScan, etc.

### Quality Metrics
- **Extraction Precision:** 95.6%
- **Inter-Rater Agreement (Krippendorff's Alpha):** 0.87 (high consistency with human experts)
- **Classification:** Hierarchical CWE (Pillar → Class → Base)

### Citation
```bibtex
@misc{chen2025forgellmdrivenframeworklargescale,
      title={FORGE: An LLM-driven Framework for Large-Scale Smart Contract Vulnerability Dataset Construction},
      author={Jiachi Chen and Yiming Shen and Jiashuo Zhang and Zihao Li and John Grundy and Zhenzhe Shao and Yanlin Wang and Jiashui Wang and Ting Chen and Zibin Zheng},
      year={2025},
      eprint={2506.18795},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2506.18795}
}
```

### Authors & Affiliation
- **Lead Authors:**
  - Jiachi Chen (Sun Yat-sen University + Zhejiang University)
  - Yiming Shen (Sun Yat-sen University)
  - Jiashuo Zhang (Peking University) - *Corresponding Author*
- **Co-Authors:**
  - Zihao Li (The Hong Kong Polytechnic University)
  - John Grundy (Monash University, Australia)
  - Zhenzhe Shao (Sun Yat-sen University)
  - Yanlin Wang (Sun Yat-sen University)
  - Jiashui Wang (Zhejiang University)
  - Ting Chen (University of Electronic Science and Technology of China)
  - Zibin Zheng (Sun Yat-sen University)

- **Conference:** ICSE 2026 (46th International Conference on Software Engineering)
- **Paper Type:** Research Track (Accepted)

### Location in Triton
```
/home/anik/code/Triton/data/datasets/FORGE-Artifacts/
├── dataset/
│   ├── contracts/      # 81,390 Solidity files
│   └── results/        # 6,454 JSON files with vulnerability info
├── evaluation/         # Benchmark results
└── src/                # FORGE framework source
```

### Data Format
Each JSON file contains:
- Project metadata (GitHub URL, commit ID, chain, compiler version)
- Vulnerability findings with:
  - CWE category (hierarchical: Pillar → Class → Base)
  - Title and description
  - Severity (critical, high, medium, low, info)
  - Location (file name, sometimes line numbers)

### Usage
```bash
# Test on FORGE dataset (sample)
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/FORGE-Artifacts/dataset/contracts
```

### Strengths
- ✅ **Largest dataset** (81K+ files, 27K+ vulnerabilities)
- ✅ **Standardized CWE classification** (296 categories)
- ✅ **Real-world vulnerabilities** from professional audits
- ✅ **High quality** (95.6% precision, k-α = 0.87)
- ✅ **Modern Solidity** (59% use v0.8.x)
- ✅ **Multiple chains** (Ethereum, BSC, Polygon, Base)
- ✅ **Severity levels** included
- ✅ **Automated construction** (can be updated with new audits)

### Limitations
- ⚠️ **Artificially created:** Extracted by LLM, not manually verified
- ⚠️ **Organized by project**, not vulnerability type (harder to filter)
- ⚠️ **Very large:** May need sampling for testing
- ⚠️ **Complex contracts:** Average 2,575 LOC per project
- ⚠️ **CWE mapping needed:** Must map 296 CWEs to Triton's 10 types
- ⚠️ **Some missing code:** Not all projects have complete source

---

## Dataset Comparison Summary

| Feature | SmartBugs Curated | FORGE Artifacts |
|---------|-------------------|-----------------|
| **Size** | 143 contracts | 81,390 files |
| **Vulnerabilities** | 143 | 27,497 |
| **Source** | Real-world (Etherscan) | Real-world (audit reports) |
| **Organization** | By vulnerability type | By project |
| **Quality** | Manual annotation | LLM extraction (95.6% precision) |
| **Classification** | DASP (10 types) | CWE (296 categories) |
| **Ground Truth** | Line-by-line | File-level + severity |
| **Best For** | Initial testing, reproducibility | Large-scale evaluation |
| **Creation Method** | Manual curation | **Automated LLM extraction** |
| **Solidity Versions** | Mixed (older) | Mostly 0.8.x (59%) |
| **Complexity** | Simple to moderate | Moderate to complex (2,575 LOC avg) |

---

## Recommended Testing Strategy

### Stage 1: Initial Validation (SmartBugs Curated)
**Why:** Small, well-organized, manually verified
**Time:** 30-60 minutes
```bash
python scripts/test_triton.py --dataset custom \
    --custom-dir data/datasets/smartbugs-curated/dataset
```

### Stage 2: Large-Scale Evaluation (FORGE - Sample)
**Why:** Test on modern, real-world contracts
**Time:** 2-4 hours
```bash
# Test random sample of 500 contracts
# (Create sampling script)
```

### Stage 3: Comprehensive Evaluation (FORGE - Full)
**Why:** Complete benchmarking
**Time:** Days
```bash
# Test all 81,390 contracts
# Use distributed testing if available
```

---

## Attribution Requirements

When publishing results using these datasets, please cite:

### For SmartBugs Curated:
> Durieux, T., Ferreira, J. F., Abreu, R., & Cruz, P. (2020). Empirical Review of Automated Analysis Tools on 47,587 Ethereum Smart Contracts. In ICSE 2020.

### For FORGE:
> Chen, J., Shen, Y., Zhang, J., Li, Z., Grundy, J., Shao, Z., Wang, Y., Wang, J., Chen, T., & Zheng, Z. (2025). FORGE: An LLM-driven Framework for Large-Scale Smart Contract Vulnerability Dataset Construction. In ICSE 2026.

**Important Note:** When using FORGE, acknowledge that it is an **artificially created dataset using automated LLM extraction** from audit reports.

---

## Explanation for Your Professor

### SmartBugs Curated
"SmartBugs Curated is a **manually curated benchmark dataset** containing 143 vulnerable smart contracts from Ethereum. It's the **gold standard** in smart contract security research, published at ICSE 2020 (top-tier software engineering conference). Each vulnerability is **manually verified** and annotated with exact line numbers."

### FORGE
"FORGE is a **state-of-the-art LLM-driven framework** published at ICSE 2026 that **automatically extracts vulnerabilities** from professional audit reports. While the vulnerabilities are real (from expert auditors), the dataset construction is **automated using AI**. It's the **largest dataset** available (81K+ files, 27K+ vulnerabilities) with 95.6% extraction precision and uses **standardized CWE classification**."

### Key Distinction
"SmartBugs is **manually created** (human experts), FORGE is **artificially created** (AI-automated extraction). Both use real-world vulnerabilities, but SmartBugs has human verification while FORGE has automated extraction with high precision (95.6%)."

---

## Dataset Licenses

### SmartBugs Curated
- **License:** Apache 2.0 / MIT (check individual contracts)
- **Usage:** Free for research and academic purposes
- **Source:** https://github.com/smartbugs/smartbugs-curated

### FORGE
- **License:** See LICENSE file in repository
- **Usage:** Free for research and academic purposes
- **Source:** https://github.com/shenyimings/FORGE-Artifacts

---

## Contact Information

### SmartBugs Curated
- **Maintainer:** Thomas Durieux
- **GitHub:** https://github.com/smartbugs/smartbugs-curated
- **Issues:** Submit on GitHub

### FORGE
- **Corresponding Author:** Jiashuo Zhang (Peking University)
- **Contact:** shenym7@mail2.sysu.edu.cn (Yiming Shen)
- **GitHub:** https://github.com/shenyimings/FORGE-Artifacts
- **Issues:** Submit on GitHub or email authors

---

**Document Version:** 1.0
**Last Updated:** 2025-10-30
**Maintained By:** Triton Project Team
