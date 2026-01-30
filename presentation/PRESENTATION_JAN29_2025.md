# Triton: Multi-Modal Smart Contract Vulnerability Detection
## Comprehensive Presentation - January 29, 2025

---

# Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Learning Paradigm](#2-learning-paradigm)
3. [Models Used](#3-models-used-detailed)
4. [Data Requirements & Current Status](#4-data-requirements--current-status)
5. [System Architecture](#5-system-architecture)
6. [Modality Overlap Analysis](#6-modality-overlap-analysis)
7. [Loss Function Design (Recall-Focused)](#7-loss-function-design-recall-focused)
8. [Proposed Improvements](#8-proposed-improvements)
9. [Current Limitations](#9-current-limitations)
10. [Future Roadmap](#10-future-roadmap)

---

# 1. Problem Statement

## 1.1 What Are We Solving?

**Smart contracts** are programs that run on Ethereum blockchain and handle millions of dollars. Vulnerabilities in these contracts have caused massive financial losses:

| Incident | Year | Loss | Vulnerability |
|----------|------|------|---------------|
| The DAO Hack | 2016 | $60M | Reentrancy |
| Parity Wallet | 2017 | $150M | Access Control |
| bZx Flash Loan | 2020 | $8M | Price Manipulation |
| Ronin Bridge | 2022 | $625M | Access Control |

**Goal**: Automatically detect vulnerabilities BEFORE deployment.

## 1.2 Vulnerability Types We Detect (11 Classes)

| # | Vulnerability | Description | Example |
|---|---------------|-------------|---------|
| 1 | **Reentrancy** | Attacker re-enters function before state update | Withdraw before balance reset |
| 2 | **Arithmetic** | Integer overflow/underflow | `uint8(255) + 1 = 0` |
| 3 | **Access Control** | Missing permission checks | Anyone can call admin functions |
| 4 | **Unchecked Calls** | Not checking return value of `.call()` | Silently failing transfers |
| 5 | **Denial of Service** | Attacker can block contract operations | Unbounded loops |
| 6 | **Bad Randomness** | Predictable random numbers | Using `block.timestamp` as seed |
| 7 | **Front Running** | Transaction order manipulation | Seeing pending tx and acting first |
| 8 | **Time Manipulation** | Relying on block.timestamp | Miners can manipulate by ~15 sec |
| 9 | **Short Addresses** | Input validation on address length | Parameter packing attacks |
| 10 | **Other** | Miscellaneous vulnerabilities | Various |
| 11 | **Safe** | No vulnerability detected | Clean contract |

---

# 2. Learning Paradigm

## 2.1 Type of Learning: SUPERVISED CLASSIFICATION

```
┌─────────────────────────────────────────────────────────────────┐
│                    SUPERVISED LEARNING                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input (X):  Smart Contract Source Code                         │
│              ↓                                                  │
│  Features:   PDG graphs, execution traces, code tokens          │
│              ↓                                                  │
│  Output (Y): Vulnerability Class (1 of 11)                      │
│                                                                 │
│  Training:   Learn mapping X → Y from labeled examples          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Type: Multi-class Classification (11 classes)
      with potential for Multi-label (contract can have multiple vulnerabilities)
```

## 2.2 Why Supervised Learning?

| Approach | Pros | Cons | Why Not Used |
|----------|------|------|--------------|
| **Supervised** (Our choice) | Learns complex patterns, high accuracy potential | Needs labeled data | - |
| Unsupervised | No labels needed | Can't classify specific vulnerabilities | No class prediction |
| Rule-based (Slither) | Fast, interpretable | Limited to known patterns | Can't learn new patterns |
| Semi-supervised | Uses unlabeled data | Complex implementation | Future work |

## 2.3 Learning Components

```
┌─────────────────────────────────────────────────────────────────┐
│                   WHAT THE MODEL LEARNS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STATIC ENCODER (Graph Neural Network):                         │
│  - Learns patterns in program structure (control flow, data flow)│
│  - "Functions that call external contracts before updating state"│
│                                                                 │
│  DYNAMIC ENCODER (Recurrent Neural Network):                    │
│  - Learns patterns in execution sequences                       │
│  - "Call depth increases, then SSTORE happens after call returns"│
│                                                                 │
│  SEMANTIC ENCODER (Transformer):                                │
│  - Learns patterns in code text/semantics                       │
│  - "Code contains 'call' followed by state modification"        │
│                                                                 │
│  FUSION MODULE (Attention):                                     │
│  - Learns which modality to trust for which vulnerability       │
│  - "For reentrancy, trust dynamic > semantic > static"          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# 3. Models Used (Detailed)

## 3.1 Overview of All Models

| Component | Model Type | Base Architecture | Parameters | Output Dim |
|-----------|------------|-------------------|------------|------------|
| Static Encoder | GNN | Graph Attention Network (GAT) | ~2M | 768 |
| Dynamic Encoder | RNN | Bidirectional LSTM + Attention | ~3M | 512 |
| Semantic Encoder | Transformer | GraphCodeBERT (Microsoft) | ~125M | 768 |
| Fusion Module | Attention | Cross-Modal Attention | ~5M | 768 |
| **TOTAL** | - | - | **~135M** | - |

## 3.2 Static Encoder: Graph Attention Network (GAT)

### What It Does
Converts smart contract into a **Program Dependence Graph (PDG)** and processes it with a Graph Neural Network.

### Architecture Details

```
                    Smart Contract
                          │
                          ▼
              ┌───────────────────────┐
              │   SLITHER ANALYSIS    │
              │   (Static Analysis)   │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Program Dependence   │
              │       Graph (PDG)     │
              │                       │
              │  Nodes = Functions,   │
              │          Variables,   │
              │          Modifiers    │
              │                       │
              │  Edges = calls,       │
              │          reads,       │
              │          writes,      │
              │          uses_modifier│
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   NODE ENCODER        │
              │   Linear(28 → 64)     │
              │   ReLU                │
              │   Linear(64 → 128)    │
              │   ReLU                │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   GAT LAYER 1         │
              │   8 attention heads   │
              │   Edge-aware (4-dim)  │
              │   128 → 256*8         │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   GAT LAYER 2         │
              │   8 attention heads   │
              │   2048 → 256*8        │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   GAT LAYER 3         │
              │   1 attention head    │
              │   2048 → 256          │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   GLOBAL MEAN POOL    │
              │   (Aggregate nodes)   │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   PROJECTION          │
              │   Linear(256 → 512)   │
              │   ReLU, Dropout       │
              │   Linear(512 → 768)   │
              └───────────┬───────────┘
                          │
                          ▼
              768-dimensional embedding
```

### Node Features (28 dimensions)

```python
# Feature breakdown for each node in PDG:

TYPE ENCODING (3 dims):
  [1,0,0] = function
  [0,1,0] = variable
  [0,0,1] = modifier

GRAPH STRUCTURE (2 dims):
  in_degree / 10.0
  out_degree / 10.0

SEMANTIC FLAGS (10 dims - binary):
  is_payable          # Can receive ETH
  is_view             # Read-only
  is_pure             # No state access
  can_reenter         # Potential reentrancy
  can_send_eth        # Sends ETH
  has_assembly        # Inline assembly
  has_require         # Has require()
  has_assert          # Has assert()
  has_selfdestruct    # Can destroy contract
  has_delegatecall    # Dangerous delegation

CONTROL FLOW (3 dims):
  has_loop            # Contains loops
  loop_depth / 5.0    # Nesting level
  has_conditional     # Has if/else

CALL COUNTS (3 dims - normalized):
  internal_calls / 10
  external_calls / 10
  low_level_calls / 5

STATE ACCESS (2 dims - normalized):
  vars_read / 10
  vars_written / 10

FUNCTION TYPES (4 dims - binary):
  is_constructor
  is_fallback
  is_receive
  is_external

TOTAL: 3+2+10+3+3+2+4 = 28 dimensions
```

### Why GAT?

| Property | Benefit for Our Task |
|----------|---------------------|
| Attention mechanism | Focuses on important nodes (vulnerable functions) |
| Edge-aware | Distinguishes call/read/write relationships |
| Permutation invariant | Works regardless of node ordering |
| Variable-size graphs | Handles contracts of any size |

---

## 3.3 Dynamic Encoder: Bidirectional LSTM + Attention

### What It Does
Processes **execution traces** (sequence of EVM opcodes) from symbolic execution.

### Architecture Details

```
                    Smart Contract
                          │
                          ▼
              ┌───────────────────────┐
              │   MYTHRIL ANALYSIS    │
              │ (Symbolic Execution)  │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Execution Trace     │
              │                       │
              │   List of steps:      │
              │   - opcode (CALL,     │
              │     SSTORE, etc.)     │
              │   - gas remaining     │
              │   - call depth        │
              │   - stack state       │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   OPCODE EMBEDDING    │
              │   50 opcodes → 128-dim│
              │   + Positional Encoding│
              └───────────┬───────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
    ┌─────────────────┐    ┌─────────────────┐
    │ Context Encoder │    │    (opcodes)    │
    │ 7 features →    │    │                 │
    │ 64 → 128 dim    │    │                 │
    └────────┬────────┘    └────────┬────────┘
             │                      │
             └──────────┬───────────┘
                        │ Concatenate
                        ▼
              ┌───────────────────────┐
              │   BIDIRECTIONAL LSTM  │
              │   3 layers            │
              │   256 hidden units    │
              │   Dropout: 0.2        │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   MULTI-HEAD ATTENTION│
              │   8 heads             │
              │   512 dim (bi-dir)    │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   PROJECTION          │
              │   512 → 256 → 512     │
              └───────────┬───────────┘
                          │
                          ▼
              512-dimensional embedding
```

### Context Features (7 dimensions per step)

```python
# For each execution step:
gas = min(step['gas'] / 1000000, 1.0)      # Gas remaining
depth = min(step['depth'] / 10, 1.0)        # Call depth
stack_size = min(len(stack) / 100, 1.0)     # Stack size

is_storage = 1.0 if opcode in ['SSTORE', 'SLOAD'] else 0.0
is_call = 1.0 if 'CALL' in opcode else 0.0
is_jump = 1.0 if 'JUMP' in opcode else 0.0
is_arithmetic = 1.0 if opcode in ['ADD','SUB','MUL','DIV','MOD'] else 0.0
```

### Opcode Vocabulary (50 opcodes)

```
STOP, ADD, MUL, SUB, DIV, MOD, EXP, NOT,
LT, GT, EQ, AND, OR, XOR, BYTE, SHL, SHR,
PUSH1-PUSH32, DUP1-DUP16, SWAP1-SWAP16,
MLOAD, MSTORE, SLOAD, SSTORE, JUMP, JUMPI,
CALL, DELEGATECALL, STATICCALL, CREATE, CREATE2,
RETURN, REVERT, SELFDESTRUCT, ...
```

### Why LSTM + Attention?

| Property | Benefit for Our Task |
|----------|---------------------|
| Sequential processing | Captures order of operations |
| Bidirectional | Context from past AND future |
| Attention | Focuses on critical execution points |
| Variable length | Handles traces of any length |

---

## 3.4 Semantic Encoder: GraphCodeBERT (Transformer)

### What It Does
Uses a **pretrained code understanding model** to extract semantic features from source code.

### Architecture Details

```
                    Smart Contract
                          │
                          ▼
              ┌───────────────────────┐
              │   PREPROCESSING       │
              │   - Remove comments   │
              │   - Normalize spaces  │
              │   - Highlight keywords│
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   TOKENIZATION        │
              │   (BPE tokenizer)     │
              │   Max length: 512     │
              └───────────┬───────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────┐
    │        GRAPHCODEBERT (Microsoft)        │
    │                                         │
    │   Pretrained on:                        │
    │   - 2.3M functions from CodeSearchNet   │
    │   - 6 programming languages             │
    │   - Code + documentation pairs          │
    │                                         │
    │   Architecture:                         │
    │   - 12 Transformer layers               │
    │   - 768 hidden dimensions               │
    │   - 12 attention heads                  │
    │   - ~125M parameters                    │
    │                                         │
    └───────────────────┬─────────────────────┘
                        │
                        ▼
              ┌───────────────────────┐
              │   POOLED OUTPUT       │
              │   [CLS] token → 768   │
              └───────────┬───────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
    ┌─────────────────┐    ┌─────────────────┐
    │ Vuln Embedding  │    │   (pooled)      │
    │ 11 types → 64   │    │                 │
    └────────┬────────┘    └────────┬────────┘
             │                      │
             └──────────┬───────────┘
                        │ Concatenate: 768 + 64 = 832
                        ▼
              ┌───────────────────────┐
              │   PROJECTION          │
              │   832 → 512 → 768     │
              └───────────┬───────────┘
                          │
                          ▼
              768-dimensional embedding
```

### Why GraphCodeBERT?

| Property | Benefit for Our Task |
|----------|---------------------|
| Pretrained on code | Already understands programming patterns |
| Transfer learning | Leverages knowledge from 2.3M functions |
| Semantic understanding | Captures meaning, not just syntax |
| Attention over tokens | Focuses on security-relevant code |

### Preprocessing for Solidity

```python
def preprocess_solidity_code(source_code):
    # Remove comments
    source_code = re.sub(r'//.*?\n', '\n', source_code)
    source_code = re.sub(r'/\*.*?\*/', '', source_code, flags=re.DOTALL)

    # Highlight security-relevant keywords
    keywords = [
        'function', 'modifier', 'contract', 'require', 'assert',
        'msg.sender', 'msg.value', 'block.timestamp', 'tx.origin',
        'call', 'delegatecall', 'selfdestruct', 'transfer', 'send'
    ]
    for keyword in keywords:
        source_code = source_code.replace(keyword, f' {keyword} ')

    return source_code
```

---

## 3.5 Fusion Module: Cross-Modal Attention

### What It Does
Combines embeddings from all three modalities with **attention-based weighting**.

### Architecture Details

```
        Static           Dynamic          Semantic
        (768-dim)        (512-dim)        (768-dim)
           │                │                │
           ▼                ▼                ▼
    ┌────────────┐   ┌────────────┐   ┌────────────┐
    │  Project   │   │  Project   │   │  Project   │
    │  768→512   │   │  512→512   │   │  768→512   │
    └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
          │                │                │
          └───────────┬────┴────────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │   CROSS-MODAL ATTENTION    │
         │                            │
         │   Each modality attends    │
         │   to all three modalities  │
         │                            │
         │   8 attention heads        │
         │   Layer normalization      │
         │   Residual connections     │
         └────────────┬───────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │   ADAPTIVE WEIGHTING       │
         │                            │
         │   Learn weights per        │
         │   vulnerability type:      │
         │                            │
         │   Reentrancy:   [0.2, 0.6, 0.2]   │
         │   Access Ctrl:  [0.7, 0.1, 0.2]   │
         │   Arithmetic:   [0.3, 0.4, 0.3]   │
         │   Default:      [0.33,0.33,0.34]  │
         └────────────┬───────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │   CONTEXT-AWARE FUSION     │
         │                            │
         │   3 fusion layers          │
         │   Gating mechanism         │
         │   512 → 768 output         │
         └────────────┬───────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │   CLASSIFICATION HEAD      │
         │   768 → 384 → 11           │
         │   + Confidence estimator   │
         └────────────────────────────┘
```

### Adaptive Weighting by Vulnerability Type

```python
# Different vulnerabilities need different modality emphasis:

default_weights = {
    'reentrancy':           [0.2, 0.6, 0.2],  # Dynamic-heavy (runtime behavior)
    'access_control':       [0.7, 0.1, 0.2],  # Static-heavy (code structure)
    'overflow':             [0.3, 0.4, 0.3],  # Balanced
    'timestamp_dependency': [0.4, 0.2, 0.4],  # Static + Semantic
    'delegatecall':         [0.3, 0.5, 0.2],  # Dynamic-heavy
    'default':              [0.33, 0.33, 0.34] # Equal weights
}
# Weights: [static, dynamic, semantic]
```

---

# 4. Data Requirements & Current Status

## 4.1 How Much Data Is Needed?

### General Deep Learning Guidelines

| Model Type | Typical Data Requirements | Our Context |
|------------|---------------------------|-------------|
| Simple CNN | 1K-10K samples | Not applicable |
| GAT (graphs) | 5K-50K graphs | Need ~10K contracts |
| LSTM | 10K-100K sequences | Need ~10K traces |
| BERT fine-tuning | 1K-100K examples | 1K+ can work with pretrained |
| Multi-class (11 classes) | ~1K per class minimum | Need ~11K total |

### For Vulnerability Detection Specifically

```
┌─────────────────────────────────────────────────────────────────┐
│                DATA REQUIREMENTS ESTIMATE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  MINIMUM for basic learning:                                    │
│  - 500-1000 samples per vulnerability class                     │
│  - Total: 5,500-11,000 labeled contracts                        │
│                                                                 │
│  RECOMMENDED for good performance:                              │
│  - 2000-5000 samples per class                                  │
│  - Total: 22,000-55,000 labeled contracts                       │
│                                                                 │
│  STATE-OF-THE-ART systems use:                                  │
│  - 50,000-200,000 contracts                                     │
│  - Augmentation techniques                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 4.2 How Much Data Do We Have?

### Current Dataset: FORGE (Reconstructed)

```
┌─────────────────────────────────────────────────────────────────┐
│                   CURRENT DATASET STATUS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Total contracts in FORGE:        6,449                         │
│  Successfully labeled:            1,148  (17.8%)                │
│  ────────────────────────────────────────                       │
│  Filtered out:                                                  │
│  - Interface-only contracts:      5,117                         │
│  - Abstract contracts:                3                         │
│  - Too small:                       124                         │
│  - No implementation:                33                         │
│  - No audit found:                   24                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Training/Validation/Test Split

```
Split        | Contracts | Percentage
-------------|-----------|------------
Training     |    841    |    70%
Validation   |   ~150    |    15%
Test         |   ~150    |    15%
TOTAL        |  ~1,148   |   100%
```

### Per-Class Distribution (Training Set)

```
Vulnerability Type        | Count | Percentage | Status
--------------------------|-------|------------|--------
arithmetic                |   289 |   34.4%    | OK
other                     |   209 |   24.9%    | OK
unchecked_low_level_calls |   142 |   16.9%    | OK
access_control            |   105 |   12.5%    | Borderline
denial_of_service         |    71 |    8.4%    | LOW
safe                      |    20 |    2.4%    | VERY LOW
time_manipulation         |     5 |    0.6%    | CRITICAL
--------------------------|-------|------------|--------
TOTAL                     |   841 |   100%     |

MISSING CLASSES:
- reentrancy:       0 samples (should have ~500+)
- bad_randomness:   0 samples (should have ~300+)
- front_running:    0 samples (should have ~300+)
- short_addresses:  0 samples (should have ~300+)
```

## 4.3 Data Gap Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA GAP ANALYSIS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HAVE:           ~1,148 labeled contracts                       │
│  NEED:           ~11,000+ for good performance                  │
│  GAP:            ~10x more data needed                          │
│                                                                 │
│  CLASS IMBALANCE:                                               │
│  - arithmetic: 289 samples (OK)                                 │
│  - time_manipulation: 5 samples (UNUSABLE)                      │
│  - Ratio: 58:1 (severe imbalance)                               │
│                                                                 │
│  MISSING CLASSES:                                               │
│  - reentrancy (0 samples) - most important vulnerability!       │
│  - bad_randomness (0 samples)                                   │
│  - front_running (0 samples)                                    │
│  - short_addresses (0 samples)                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 4.4 The Label Quality Problem (Critical Issue)

```
┌─────────────────────────────────────────────────────────────────┐
│              THE LABEL QUALITY CEILING PROBLEM                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HOW CURRENT LABELS WERE CREATED:                               │
│  ────────────────────────────────                               │
│  1. Run Slither (static analysis tool) on contracts             │
│  2. Use Slither's detection as ground truth labels              │
│                                                                 │
│  THE PROBLEM:                                                   │
│  ────────────────────────────────                               │
│  - Slither itself has false positives and false negatives       │
│  - Slither accuracy: ~50-70% (varies by vulnerability)          │
│  - Our model learns to MIMIC Slither, not detect real vulns     │
│                                                                 │
│  CONSEQUENCE:                                                   │
│  ────────────────────────────────                               │
│  Even with perfect model, accuracy is CAPPED at Slither's       │
│  accuracy. We're learning tool behavior, not real vulnerabilities│
│                                                                 │
│       ┌─────────────────────────────────────────┐               │
│       │  If Slither is 70% accurate:            │               │
│       │  → Maximum achievable: 70%              │               │
│       │  → Current: 28-50%                      │               │
│       │  → Ceiling reached, not model's fault   │               │
│       └─────────────────────────────────────────┘               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 4.5 Solution: Human-Verified Audit Data

### Available Sources

| Source | Contracts | Label Quality | Pros | Cons |
|--------|-----------|---------------|------|------|
| **Code4rena** | 500+ | Expert-verified | High quality, PoC included | Smaller volume |
| **Sherlock** | 300+ | Expert-verified | Real findings | Limited categories |
| **Solodit** | 10,000+ | Aggregated | Large volume | May have duplicates |
| **Trail of Bits** | 100+ | Professional audit | Highest quality | Very small volume |

### Proposed Data Strategy

```
PHASE 1: Verify current data quality
  - Sample 100 contracts
  - Manual label verification
  - Quantify label accuracy

PHASE 2: Collect audit data
  - Download Code4rena findings
  - Map to our 11 categories
  - Create clean dataset

PHASE 3: Retrain with quality data
  - Expect significant accuracy improvement
  - Remove Slither accuracy ceiling
```

---

# 5. System Architecture

## 5.1 Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TRITON ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        INPUT: Smart Contract (.sol)                         │
│                                    │                                        │
│              ┌─────────────────────┼─────────────────────┐                  │
│              │                     │                     │                  │
│              ▼                     ▼                     ▼                  │
│     ┌────────────────┐    ┌────────────────┐    ┌────────────────┐         │
│     │   SLITHER      │    │    MYTHRIL     │    │  TOKENIZER     │         │
│     │ (Static Tool)  │    │(Symbolic Exec) │    │ (BPE)          │         │
│     └───────┬────────┘    └───────┬────────┘    └───────┬────────┘         │
│             │                     │                     │                  │
│             ▼                     ▼                     ▼                  │
│     ┌────────────────┐    ┌────────────────┐    ┌────────────────┐         │
│     │      PDG       │    │   Execution    │    │     Token      │         │
│     │     Graph      │    │    Traces      │    │    Sequence    │         │
│     └───────┬────────┘    └───────┬────────┘    └───────┬────────┘         │
│             │                     │                     │                  │
│             ▼                     ▼                     ▼                  │
│     ┌────────────────┐    ┌────────────────┐    ┌────────────────┐         │
│     │    STATIC      │    │    DYNAMIC     │    │   SEMANTIC     │         │
│     │   ENCODER      │    │    ENCODER     │    │   ENCODER      │         │
│     │                │    │                │    │                │         │
│     │   3-layer GAT  │    │ Bi-LSTM + Attn │    │ GraphCodeBERT  │         │
│     │   8 heads      │    │ 3 layers       │    │ 12 layers      │         │
│     │   ~2M params   │    │ ~3M params     │    │ ~125M params   │         │
│     └───────┬────────┘    └───────┬────────┘    └───────┬────────┘         │
│             │                     │                     │                  │
│             │ 768-dim             │ 512-dim             │ 768-dim          │
│             │                     │                     │                  │
│             └──────────────┬──────┴──────────────┬──────┘                  │
│                            │                     │                         │
│                            ▼                     ▼                         │
│                   ┌─────────────────────────────────────┐                  │
│                   │         CROSS-MODAL FUSION          │                  │
│                   │                                     │                  │
│                   │  1. Cross-Modal Attention (8 heads) │                  │
│                   │  2. Adaptive Modality Weighting     │                  │
│                   │  3. Context-Aware Fusion (3 layers) │                  │
│                   │  4. Gating Mechanism                │                  │
│                   │                                     │                  │
│                   │  ~5M parameters                     │                  │
│                   └──────────────┬──────────────────────┘                  │
│                                  │                                         │
│                                  │ 768-dim fused embedding                 │
│                                  │                                         │
│                                  ▼                                         │
│                   ┌─────────────────────────────────────┐                  │
│                   │       CLASSIFICATION HEAD           │                  │
│                   │                                     │                  │
│                   │  768 → 384 → 11 classes             │                  │
│                   │  + Confidence Score (0-1)           │                  │
│                   └──────────────┬──────────────────────┘                  │
│                                  │                                         │
│                                  ▼                                         │
│                                                                            │
│                   OUTPUT: Vulnerability Type + Confidence                  │
│                                                                            │
│                   Example: "reentrancy" (confidence: 0.87)                 │
│                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 5.2 Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: Data Preparation                                                  │
│  ──────────────────────────                                                 │
│  1. Load contracts from train/val/test directories                          │
│  2. Extract PDG using Slither (with caching)                                │
│  3. Extract traces using Mythril (with caching)                             │
│  4. Tokenize source code                                                    │
│  5. Compute class weights (inverse frequency)                               │
│                                                                             │
│  PHASE 2: Individual Encoder Training (Optional)                            │
│  ───────────────────────────────────────────────                            │
│  - Can train each encoder separately first                                  │
│  - Useful for debugging and ablation studies                                │
│                                                                             │
│  PHASE 3: End-to-End Training                                               │
│  ────────────────────────────                                               │
│  For each epoch:                                                            │
│    For each batch:                                                          │
│      1. Forward pass through all encoders                                   │
│      2. Fuse embeddings                                                     │
│      3. Compute Focal Loss (recall-focused)                                 │
│      4. Backpropagate gradients                                             │
│      5. Update weights (Adam optimizer)                                     │
│    Validate on val set                                                      │
│    Save best model (by F1 score)                                            │
│    Early stopping if no improvement for 5 epochs                            │
│                                                                             │
│  PHASE 4: Evaluation                                                        │
│  ─────────────────────                                                      │
│  - Per-class precision, recall, F1                                          │
│  - Confusion matrix                                                         │
│  - Overall accuracy                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# 6. Modality Overlap Analysis

## 6.1 The Key Question

**Do the three modalities detect the SAME vulnerabilities or DIFFERENT ones?**

- If SAME (redundant) → Fusion won't help much
- If DIFFERENT (complementary) → Fusion should improve significantly

## 6.2 Analysis Results

```
┌─────────────────────────────────────────────────────────────────┐
│               MODALITY OVERLAP ANALYSIS RESULTS                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Test Set Analysis (overlapping samples):                       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │  Both Static & Semantic CORRECT:      ~15%  (redundant)  │   │
│  │  Only Static correct:                 ~13%  (S helps)    │   │
│  │  Only Semantic correct:               ~35%  (Sem helps)  │   │
│  │  Both WRONG:                          ~37%  (hard cases) │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  INTERPRETATION:                                                │
│  ───────────────                                                │
│  - 48% complementary (fusion CAN help)                          │
│  - 15% redundant (fusion won't help)                            │
│  - 37% both fail (neither helps alone)                          │
│                                                                 │
│  → Expected improvement from fusion: 5-10%                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 6.3 Are Modalities Additive?

```
THEORETICAL MAXIMUM (if fully independent):
────────────────────────────────────────────
Combined = 1 - (1 - Static)(1 - Dynamic)(1 - Semantic)
         = 1 - (1 - 0.28)(1 - 0.20)(1 - 0.50)
         = 1 - (0.72)(0.80)(0.50)
         = 1 - 0.288
         = 71.2%

THEORETICAL MINIMUM (if fully redundant):
──────────────────────────────────────────
Combined = max(Static, Dynamic, Semantic)
         = max(28%, 20%, 50%)
         = 50%

REALITY (partial overlap):
────────────────────────────
Expected: 55-60% (between extremes)

WHY NOT ADDITIVE:
- Modalities share some common signal
- Some vulnerabilities detectable by all methods
- Correlation in what they get right/wrong
```

## 6.4 Per-Vulnerability Modality Strength

```
Vulnerability Type    | Static | Dynamic | Semantic | BEST FOR
----------------------|--------|---------|----------|------------------
Reentrancy            |  LOW   | MEDIUM  |  HIGH    | Semantic/Dynamic
Access Control        |  LOW   |   LOW   |  HIGH    | Semantic
DoS                   | HIGH   | MEDIUM  | MEDIUM   | Static
Arithmetic            | MEDIUM |   LOW   | MEDIUM   | Static/Semantic
Unchecked Calls       | MEDIUM | MEDIUM  |  HIGH    | Semantic
Time Manipulation     | MEDIUM |   LOW   | MEDIUM   | Static
Bad Randomness        |  LOW   | MEDIUM  |  HIGH    | Semantic/Dynamic
```

---

# 7. Loss Function Design (Recall-Focused)

## 7.1 Why Prioritize Recall?

```
┌─────────────────────────────────────────────────────────────────┐
│                      COST ASYMMETRY                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FALSE NEGATIVE (Miss a vulnerability):                         │
│  ─────────────────────────────────────                          │
│  - Vulnerable contract gets deployed                            │
│  - Attacker exploits it                                         │
│  - Millions of dollars lost                                     │
│  - Reputation destroyed                                         │
│  - CATASTROPHIC COST                                            │
│                                                                 │
│  FALSE POSITIVE (Flag a safe contract):                         │
│  ─────────────────────────────────────                          │
│  - Contract sent for manual review                              │
│  - Reviewer finds no issue                                      │
│  - Some time wasted                                             │
│  - ACCEPTABLE COST                                              │
│                                                                 │
│  CONCLUSION: Better to have false alarms than miss real vulns   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 7.2 Current Implementation: Focal Loss

```python
class FocalLoss(nn.Module):
    """
    Focal Loss with asymmetric class weights

    Key insight: Down-weight easy examples, focus on hard ones
    Plus: Penalize missing vulnerabilities more than false alarms
    """

    def __init__(self, alpha=None, gamma=2.0):
        self.alpha = alpha  # Per-class importance
        self.gamma = gamma  # Focusing parameter

    def forward(self, predictions, targets):
        # Standard cross-entropy
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')

        # Probability of correct class
        pt = torch.exp(-ce_loss)

        # Focal term: (1 - pt)^gamma
        # When pt is HIGH (easy, correct): (1 - pt)^gamma is SMALL → low loss
        # When pt is LOW (hard, wrong):    (1 - pt)^gamma is LARGE → high loss
        focal_term = (1 - pt) ** self.gamma

        # Apply focal weighting
        loss = focal_term * ce_loss

        # Apply class importance weights
        if self.alpha is not None:
            loss = self.alpha[targets] * loss

        return loss.mean()
```

### Alpha Weights Configuration

```python
# Current configuration:
alpha_weights = torch.ones(11)

# Vulnerability classes (0-9): HIGH importance
for i in range(10):
    alpha_weights[i] = 2.0

# Safe class (10): LOW importance
alpha_weights[10] = 0.25

# Effect:
# Missing a vulnerability → 2.0 × loss
# False alarm on safe    → 0.25 × loss
# Ratio: 8× more penalty for missing vulnerabilities
```

## 7.3 Additional Recall Strategies

### Strategy 1: Lower Decision Threshold

```python
# Standard: vulnerable if P(vuln) > 0.5
# Recall-focused: vulnerable if P(vuln) > 0.3

def predict_with_threshold(logits, threshold=0.3):
    probs = F.softmax(logits, dim=-1)
    max_vuln_prob = probs[:, :-1].max(dim=-1)  # Exclude safe class
    is_vulnerable = max_vuln_prob > threshold
    return is_vulnerable
```

### Strategy 2: F2 Score for Evaluation

```
F1 = 2 × (P × R) / (P + R)           # Equal weight

F2 = 5 × (P × R) / (4P + R)          # 2× weight on Recall

     where P = Precision, R = Recall
```

---

# 8. Proposed Improvements

## 8.1 Two-Stage Classification (Binary First)

```
┌─────────────────────────────────────────────────────────────────┐
│                   TWO-STAGE CLASSIFICATION                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                      Smart Contract                             │
│                           │                                     │
│                           ▼                                     │
│             ┌─────────────────────────┐                         │
│             │   STAGE 1: BINARY       │                         │
│             │   "Is this vulnerable?" │                         │
│             │                         │                         │
│             │   Output: Yes/No        │                         │
│             │   Threshold: 0.3        │                         │
│             │   Target: 95% Recall    │                         │
│             └───────────┬─────────────┘                         │
│                         │                                       │
│                    Yes? │ No?                                   │
│                         │                                       │
│            ┌────────────┴────────────┐                          │
│            │                         │                          │
│            ▼                         ▼                          │
│  ┌─────────────────┐       ┌─────────────────┐                  │
│  │  STAGE 2: TYPE  │       │  Return "SAFE"  │                  │
│  │  CLASSIFICATION │       │  (high conf)    │                  │
│  │                 │       └─────────────────┘                  │
│  │  10 vuln types  │                                            │
│  │  (no safe class)│                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│    Vulnerability Type                                           │
│    + Confidence Score                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

WHY THIS HELPS:
- Binary classification is simpler (2 classes vs 11)
- Stage 1 can optimize purely for recall
- Stage 2 works on smaller, focused subset
- Separates "Is there a problem?" from "What kind?"
```

## 8.2 One-vs-All Classification

```
┌─────────────────────────────────────────────────────────────────┐
│                   ONE-VS-ALL APPROACH                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Instead of: 11-class classifier                                │
│  Use:        11 binary classifiers                              │
│                                                                 │
│  Classifier 1: Reentrancy vs ALL others                         │
│  Classifier 2: Arithmetic vs ALL others                         │
│  Classifier 3: Access Control vs ALL others                     │
│  ...                                                            │
│  Classifier 11: Safe vs ALL others                              │
│                                                                 │
│  ADVANTAGES:                                                    │
│  - Can tune threshold per vulnerability type                    │
│  - Handle class imbalance per classifier                        │
│  - Simpler decision boundary (1 vs many)                        │
│  - Can detect multiple vulnerabilities (multi-label)            │
│                                                                 │
│  DISADVANTAGES:                                                 │
│  - More models to train (11 instead of 1)                       │
│  - May have conflicting predictions                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 8.3 Hybrid Approach: ML + Tools

```
Stage 1: ML Model (our system)
         - High recall binary classification
         - Flag all potentially vulnerable contracts

Stage 2: Existing Tools (Slither, Mythril)
         - Run only on flagged contracts
         - Categorize vulnerability type
         - Provide specific line numbers

RATIONALE:
- ML excels at pattern recognition (anomaly detection)
- Rule-based tools excel at categorization (known patterns)
- Combine strengths of both
```

---

# 9. Current Limitations

## 9.1 Data Limitations

| Issue | Impact | Solution |
|-------|--------|----------|
| Small dataset (1,148 contracts) | Underfitting | Get more data |
| Tool-generated labels | 50-70% accuracy ceiling | Use audit data |
| Missing classes (reentrancy, etc.) | Can't learn these | Add samples |
| Severe class imbalance (58:1) | Biased predictions | Weighted sampling |

## 9.2 Model Limitations

| Issue | Impact | Solution |
|-------|--------|----------|
| PDG extraction fails on 30% | Reduced coverage | Improve Slither wrapper |
| Mythril timeout on complex contracts | No dynamic features | Optimize or skip |
| Max 512 tokens | Truncates large contracts | Chunking strategy |
| Fusion not tested end-to-end | Unknown improvement | Run full experiment |

## 9.3 Evaluation Limitations

| Issue | Impact | Solution |
|-------|--------|----------|
| Different test sets per modality | Unfair comparison | Unified test set |
| Small test set (~150) | High variance | Cross-validation |
| No baseline comparison | Unknown relative performance | Benchmark against Slither |

---

# 10. Future Roadmap

## 10.1 Immediate (Next 2 Weeks)

```
□ Verify dataset quality (sample 100, manually check)
□ Implement binary classification head
□ Create unified test set for all modalities
□ Run end-to-end fusion training
```

## 10.2 Short-term (1-2 Months)

```
□ Collect Code4rena audit data
□ Map audit findings to our 11 categories
□ Retrain on human-verified labels
□ Implement one-vs-all experiment
□ Achieve >70% accuracy target
```

## 10.3 Medium-term (3-6 Months)

```
□ Deploy as API service
□ Build web interface
□ Integrate with development tools (Hardhat, Foundry)
□ Publish results paper
```

---

# Summary for Professor

## Key Takeaways

1. **Learning Type**: Supervised multi-class classification (11 classes)

2. **Models Used**:
   - Static: 3-layer GAT (~2M params)
   - Dynamic: Bi-LSTM + Attention (~3M params)
   - Semantic: GraphCodeBERT (~125M params, pretrained)
   - Fusion: Cross-modal attention (~5M params)
   - **Total: ~135M parameters**

3. **Data Status**:
   - Have: ~1,148 labeled contracts
   - Need: ~11,000+ for good performance
   - Gap: ~10x more data needed
   - Quality issue: Labels from Slither (50-70% accurate)

4. **Current Performance**:
   - Static: 28.24%
   - Dynamic: 20.45%
   - Semantic: ~50%
   - Fusion: Not fully tested yet

5. **Key Insight**: Data quality is the bottleneck, not model architecture

6. **Proposed Solutions**:
   - Verify and fix dataset quality
   - Binary classification first
   - Recall-focused training (already implemented)
   - One-vs-all classification experiment

---

*Document prepared for professor presentation - January 29, 2025*
