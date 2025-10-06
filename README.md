cat > README.md << 'EOF'
# Triton

**Agentic Multimodal Representation for Smart Contract Vulnerability Detection**

Triton is a research project that combines static analysis, dynamic execution traces, and semantic understanding through cross-modal fusion to detect vulnerabilities in Solidity smart contracts with high accuracy and low false positive rates.

## Overview

Smart contract vulnerabilities have led to billions of dollars in losses. Existing detection tools suffer from high false positive rates and miss complex vulnerabilities. Triton addresses this through three key innovations:

### Three-Dimensional Approach

**1. Static Structure Analysis (PDG)**
- Extracts Program Dependence Graphs capturing control flow and data dependencies
- Encoded using graph transformers with edge-aware attention
- Captures structural patterns like improper access control

**2. Dynamic Execution Analysis (Novel)**
- First system to learn embeddings from symbolic execution traces
- Uses Mythril to generate execution paths
- LSTM-based encoding of opcode sequences with execution context
- Detects runtime behaviors like reentrancy loops

**3. Semantic Code Understanding**
- Fine-tuned GraphCodeBERT on 60K+ smart contracts
- Captures high-level vulnerability patterns
- Provides semantic context beyond syntax

### Advanced Fusion Architecture

Instead of simple concatenation, Triton uses:
- Cross-modal attention mechanisms
- Learned adaptive weighting per vulnerability type
- Context-aware modality importance (e.g., dynamic traces weighted 60% for reentrancy detection)

### Agentic Orchestration

LLM-driven iterative refinement:
- Initial analysis with all three modalities
- Confidence-based decision making
- Selective deep analysis with targeted tool execution
- Up to 5 refinement iterations with early stopping

## Key Features

- **Novel Dynamic Modality**: First to learn representations from symbolic execution traces rather than hand-crafted features
- **Adaptive Fusion**: Vulnerability-aware modality weighting (reentrancy: 60% dynamic, access control: 70% static)
- **Agentic Workflow**: Iterative refinement with confidence thresholds (θ = 0.9)
- **Comprehensive Coverage**: 10+ vulnerability types including reentrancy, access control, integer overflow, timestamp dependence
- **Low False Positives**: Target 12% FPR vs 16-18% in baselines

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM

### Setup
```bash