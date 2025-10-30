# Triton Testing Datasets - Summary for Professor

## Overview
We are using **two complementary benchmark datasets** to evaluate Triton's smart contract vulnerability detection performance.

---

## Dataset 1: SmartBugs Curated (Manually Created)

### Basic Information
- **Size:** 143 vulnerable smart contracts
- **Source:** Real-world contracts from Ethereum blockchain
- **Quality:** Manually curated and verified by human experts
- **Organization:** By vulnerability type (10 categories)
- **Creation Method:** **MANUAL** - Human experts selected and annotated contracts

### Published Paper
**Title:** "Empirical Review of Automated Analysis Tools on 47,587 Ethereum Smart Contracts"

**Authors:** Thomas Durieux, João F. Ferreira, Rui Abreu, Pedro Cruz

**Affiliation:** University of Luxembourg, IST Portugal

**Conference:** ICSE 2020 (International Conference on Software Engineering - Rank A*)

**Citation:**
```
Durieux, T., Ferreira, J. F., Abreu, R., & Cruz, P. (2020).
Empirical Review of Automated Analysis Tools on 47,587 Ethereum Smart Contracts.
In Proceedings of ICSE 2020, pages 530-541. ACM.
```

### Why We Use It
- ✅ **Gold standard** benchmark in smart contract security research
- ✅ **Manually verified** - highest quality ground truth
- ✅ Widely used - enables **comparison** with other tools
- ✅ Well-organized by vulnerability type

---

## Dataset 2: FORGE Artifacts (Artificially Created by LLM)

### Basic Information
- **Size:** 81,390 Solidity files, 27,497 vulnerabilities
- **Source:** Extracted from 6,454 professional security audit reports
- **Quality:** 95.6% precision, validated by human experts
- **Organization:** By project (from audit reports)
- **Creation Method:** **ARTIFICIAL** - Automated LLM extraction from PDF audit reports

⚠️ **Important:** While the vulnerabilities are real (from professional auditors), the **dataset was constructed using AI/LLM automation**, not manual human curation.

### Published Paper
**Title:** "FORGE: An LLM-driven Framework for Large-Scale Smart Contract Vulnerability Dataset Construction"

**Authors:**
- Jiachi Chen¹'² (Lead Author)
- Yiming Shen¹ (Lead Author)
- Jiashuo Zhang³* (Corresponding Author)
- +6 co-authors

**Affiliations:**
1. Sun Yat-sen University, China
2. Zhejiang University (State Key Lab of Blockchain), China
3. Peking University, China (Corresponding)
4. Hong Kong Polytechnic University, Hong Kong
5. Monash University, Australia

**Conference:** ICSE 2026 (46th International Conference on Software Engineering - Rank A*)

**Status:** Accepted (2025)

**Citation:**
```
Chen, J., Shen, Y., Zhang, J., et al. (2025).
FORGE: An LLM-driven Framework for Large-Scale Smart Contract
Vulnerability Dataset Construction.
In Proceedings of ICSE 2026. arXiv:2506.18795
```

### Why We Use It
- ✅ **Largest dataset** available (81K+ files vs. 143)
- ✅ **Real-world audit data** from professional security firms
- ✅ **Standardized CWE classification** (296 categories)
- ✅ **Modern Solidity** (59% use latest v0.8.x compiler)
- ✅ Enables **large-scale evaluation**

### Dataset Construction Methodology (FORGE Framework)
FORGE uses a 4-module LLM-driven pipeline:

1. **Semantic Chunker**: Segments audit PDFs into meaningful chunks
2. **MapReduce Extractor**: LLM extracts vulnerability information
3. **Hierarchical Classifier**: LLM classifies into CWE categories using tree-of-thoughts
4. **Code Fetcher**: Retrieves source code from GitHub/Etherscan/BSCScan

**Validation:** Manual assessment shows 95.6% precision and k-α=0.87 inter-rater agreement with human experts.

---

## Dataset Comparison

| Aspect | SmartBugs Curated | FORGE Artifacts |
|--------|-------------------|-----------------|
| **Creation** | **Manual** (human experts) | **Artificial** (LLM automation) |
| **Size** | 143 contracts | 81,390 files |
| **Vulnerabilities** | 143 | 27,497 |
| **Quality** | Human verified | 95.6% precision (automated) |
| **Source** | Ethereum blockchain | Professional audit reports |
| **Best For** | Initial testing, reproducibility | Large-scale evaluation |
| **Publication** | ICSE 2020 | ICSE 2026 |

---

## Testing Strategy

### Phase 1: SmartBugs Curated (Initial Validation)
- **Purpose:** Validate Triton works correctly
- **Time:** ~30 minutes
- **Expected Results:** Precision, Recall, F1-score on 143 contracts
- **Comparison:** With 13 existing tools (from original paper)

### Phase 2: FORGE (Large-Scale Evaluation)
- **Purpose:** Test on large-scale, modern, real-world contracts
- **Time:** Several hours (can sample if needed)
- **Expected Results:** Performance on 81K+ files, 27K+ vulnerabilities
- **Benefits:** Test on latest Solidity (0.8.x), standardized CWE classification

---

## Key Points for Professor

1. **Both datasets use real vulnerabilities** from actual smart contracts
2. **SmartBugs is manually created** (human experts) - highest quality, small scale
3. **FORGE is artificially created** (LLM automation) - high quality (95.6%), large scale
4. **Both are from top-tier conferences** (ICSE - Rank A* in software engineering)
5. **Complementary strengths:** SmartBugs for precision, FORGE for scale
6. **Proper attribution:** We will cite both papers when publishing results

---

## Expected Triton Performance

Based on our presentation, Triton targets:
- **F1-Score:** 92.5%
- **Speed:** 73% faster than baseline
- **Throughput:** 3.8× higher

We will evaluate on both datasets to demonstrate:
- **SmartBugs:** Comparison with 13 existing tools (reproducibility)
- **FORGE:** Large-scale, modern contract evaluation (scalability)

---

## References

[1] Durieux et al., "Empirical Review of Automated Analysis Tools on 47,587 Ethereum Smart Contracts," ICSE 2020

[2] Chen et al., "FORGE: An LLM-driven Framework for Large-Scale Smart Contract Vulnerability Dataset Construction," ICSE 2026

---

**Note:** When discussing FORGE with your professor, emphasize that while it's **artificially constructed using LLM**, it has been **validated by human experts** (95.6% precision, k-α=0.87) and published at a **top-tier conference** (ICSE 2026). This automated approach is actually a **significant research contribution** - it's the first framework to automatically construct high-quality vulnerability datasets at scale.
