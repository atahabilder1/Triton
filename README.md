# Triton

**Agentic Multimodal Representation for Smart Contract Vulnerability Detection**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Triton is a cutting-edge research project that revolutionizes smart contract vulnerability detection by combining static analysis, dynamic execution traces, and semantic understanding through advanced cross-modal fusion and agentic orchestration.

## 🚀 Overview

Smart contract vulnerabilities have led to billions of dollars in losses across DeFi protocols. Traditional detection tools suffer from high false positive rates (16-18%) and miss complex vulnerabilities that span multiple execution contexts. Triton addresses these limitations through three groundbreaking innovations:

### 🔬 Three-Dimensional Analysis

**1. Static Structure Analysis (PDG-Based)**
- Extracts Program Dependence Graphs capturing control flow and data dependencies
- Graph Attention Networks (GAT) with edge-aware attention mechanisms
- Identifies structural vulnerabilities like improper access control patterns
- 70% weight for access control vulnerabilities

**2. Dynamic Execution Analysis (Novel Approach)**
- **First system** to learn embeddings from symbolic execution traces
- Leverages Mythril for comprehensive execution path generation
- LSTM-based encoding of opcode sequences with execution context
- Detects runtime behaviors like reentrancy loops and gas limit attacks
- 60% weight for reentrancy vulnerabilities

**3. Semantic Code Understanding**
- Fine-tuned GraphCodeBERT on 60K+ smart contracts
- Captures high-level vulnerability patterns beyond syntax
- Advanced tokenization and preprocessing for Solidity code
- Context-aware vulnerability type embeddings

### 🧠 Advanced Fusion Architecture

**Cross-Modal Attention Mechanisms**
- Multi-head attention between all modality pairs
- Learned adaptive weighting per vulnerability type
- Context-aware modality importance scoring
- Residual connections and layer normalization

**Adaptive Modality Weighting**
- Vulnerability-specific weight distributions:
  - Reentrancy: 20% static, 60% dynamic, 20% semantic
  - Access Control: 70% static, 10% dynamic, 20% semantic
  - Integer Overflow: 30% static, 40% dynamic, 30% semantic
- Temperature-scaled softmax for weight calibration

### 🤖 Agentic Orchestration

**Iterative Refinement Workflow**
- Initial analysis with all three modalities
- Confidence-based decision making (θ = 0.9)
- Selective deep analysis with targeted tool execution
- Up to 5 refinement iterations with early stopping
- Evidence accumulation and consistency checking

**Decision Engine**
- Phase selection based on modality confidence gaps
- Uncertainty quantification and calibration
- Early stopping criteria for efficiency

## ✨ Key Features

- **🚀 Novel Dynamic Modality**: First to learn representations from symbolic execution traces
- **⚖️ Adaptive Fusion**: Vulnerability-aware modality weighting with learned importance
- **🔄 Agentic Workflow**: Iterative refinement with confidence thresholds
- **📊 Comprehensive Coverage**: 10+ vulnerability types with specialized detection
- **🎯 Low False Positives**: Target 12% FPR vs 16-18% in state-of-the-art baselines
- **⚡ Efficient Analysis**: Early stopping and selective deep analysis

## 🛠 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM for large contract analysis
- Git for repository management

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Triton.git
cd Triton

# Create virtual environment
python3 -m venv triton_env
source triton_env/bin/activate  # On Windows: triton_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse

# Install project dependencies
pip install -r requirements.txt

# Install external analysis tools
pip install slither-analyzer mythril

# Verify installation
python -c "import torch; import transformers; import networkx; print('✅ All dependencies installed successfully!')"
```

### External Tools Verification

```bash
# Test Slither
slither --version

# Test Mythril
myth version

# If tools are not found, ensure they're in your PATH
which slither
which myth
```

## 🚀 Quick Start

### Basic Usage

```bash
# Analyze a smart contract
python main.py contract.sol

# With specific vulnerability targeting
python main.py contract.sol --target-vulnerability reentrancy

# Verbose output with detailed logging
python main.py contract.sol --verbose --output results.json
```

### Example Vulnerable Contract

Create a test contract:

```solidity
// vulnerable_contract.sol
pragma solidity ^0.8.0;

contract VulnerableBank {
    mapping(address => uint256) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    // Vulnerable to reentrancy
    function withdraw() public {
        uint256 amount = balances[msg.sender];
        require(amount > 0, "Insufficient balance");

        // External call before state change (vulnerable!)
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");

        balances[msg.sender] = 0;  // State change after external call
    }

    // Vulnerable to access control
    function emergencyWithdraw() public {
        // Missing onlyOwner modifier!
        payable(msg.sender).transfer(address(this).balance);
    }
}
```

### Run Analysis

```bash
# Basic analysis
python main.py vulnerable_contract.sol --contract-name VulnerableBank

# Target specific vulnerability
python main.py vulnerable_contract.sol \
    --target-vulnerability reentrancy \
    --confidence-threshold 0.85 \
    --max-iterations 3 \
    --output analysis_results.json \
    --verbose
```

### Expected Output

```
================================================================================
TRITON VULNERABILITY ANALYSIS SUMMARY
================================================================================
Vulnerability Detected: YES
Vulnerability Type: reentrancy
Confidence Score: 0.943
Analysis Iterations: 2
Early Stopping: YES

Modality Contributions:
  - Static: 0.201
  - Dynamic: 0.612
  - Semantic: 0.187

Reasoning: Refinement analysis combining all modalities with ensemble confidence 0.943

Phases Executed: initial, deep_dynamic, refinement
================================================================================
```

## 🏗 Architecture

### Project Structure

```
Triton/
├── main.py                    # CLI application entry point
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── LICENSE                   # MIT license
│
├── encoders/                 # Multimodal encoders
│   ├── static_encoder.py     # Graph Attention Networks for PDG
│   ├── dynamic_encoder.py    # LSTM for execution traces
│   ├── semantic_encoder.py   # GraphCodeBERT for code understanding
│   └── __init__.py
│
├── fusion/                   # Cross-modal fusion
│   ├── cross_modal_fusion.py # Attention mechanisms & adaptive weighting
│   └── __init__.py
│
├── orchestrator/             # Agentic workflow
│   ├── agentic_workflow.py   # Iterative refinement logic
│   └── __init__.py
│
├── tools/                    # External tool integrations
│   ├── slither_wrapper.py    # Static analysis via Slither
│   ├── mythril_wrapper.py    # Dynamic analysis via Mythril
│   └── __init__.py
│
├── utils/                    # Utilities and metrics
│   ├── data_loader.py        # Dataset loading and batching
│   ├── metrics.py            # Evaluation metrics and calibration
│   └── __init__.py
│
├── tests/                    # Test suite
│   ├── test_triton.py        # Comprehensive unit tests
│   └── __init__.py
│
├── data/                     # Data directory (create manually)
│   ├── datasets/             # Training datasets
│   ├── processed/            # Processed features
│   └── raw/                  # Raw contract files
│
├── experiments/              # Experiment outputs (create manually)
│   ├── checkpoints/          # Model checkpoints
│   ├── logs/                 # Training logs
│   └── results/              # Analysis results
│
└── configs/                  # Configuration files (create manually)
```

### Component Details

#### 📊 Static Encoder (`encoders/static_encoder.py`)
- **Input**: Program Dependence Graphs from Slither
- **Architecture**: Graph Attention Networks with edge-aware attention
- **Features**: Node types, control flow, data dependencies
- **Output**: 768-dimensional static representations

#### 🔄 Dynamic Encoder (`encoders/dynamic_encoder.py`)
- **Input**: Symbolic execution traces from Mythril
- **Architecture**: LSTM with positional encoding and attention
- **Features**: Opcode sequences, gas consumption, call depth
- **Output**: 512-dimensional dynamic representations

#### 🧠 Semantic Encoder (`encoders/semantic_encoder.py`)
- **Input**: Raw Solidity source code
- **Architecture**: Fine-tuned GraphCodeBERT
- **Features**: Tokenized code, vulnerability patterns, AST structure
- **Output**: 768-dimensional semantic representations

#### 🤝 Cross-Modal Fusion (`fusion/cross_modal_fusion.py`)
- **Multi-head cross-attention** between all modality pairs
- **Adaptive weighting** based on vulnerability type and confidence
- **Context-aware fusion** with residual connections
- **Confidence estimation** with uncertainty quantification

#### 🤖 Agentic Orchestrator (`orchestrator/agentic_workflow.py`)
- **Phase management**: Initial → Deep Analysis → Refinement → Final
- **Decision engine**: Confidence-based phase transitions
- **Evidence accumulation**: Consistency checking across iterations
- **Early stopping**: Efficiency optimization with quality preservation

## 🎯 Supported Vulnerability Types

| Vulnerability Type | Detection Method | Primary Modality | Confidence Threshold |
|-------------------|------------------|------------------|---------------------|
| **Reentrancy** | Call-state patterns | Dynamic (60%) | 0.85 |
| **Access Control** | Modifier analysis | Static (70%) | 0.90 |
| **Integer Overflow/Underflow** | Arithmetic operations | Dynamic (40%) | 0.88 |
| **Timestamp Dependency** | Block variables | Semantic (40%) | 0.82 |
| **Unchecked Call Return** | Call result handling | Static (50%) | 0.85 |
| **Delegatecall** | Call patterns | Dynamic (50%) | 0.87 |
| **Self-Destruct** | Suicide patterns | Static (60%) | 0.88 |
| **Gas Limit** | Loop patterns | Dynamic (50%) | 0.83 |
| **TX Origin** | Authentication | Semantic (50%) | 0.85 |

## 📈 Performance Benchmarks

### Accuracy Metrics (Target vs Baseline)

| Metric | Triton (Target) | Baseline Tools | Improvement |
|--------|-----------------|----------------|-------------|
| **Precision** | 94.2% | 87.1% | +7.1% |
| **Recall** | 91.8% | 85.3% | +6.5% |
| **F1-Score** | 93.0% | 86.2% | +6.8% |
| **False Positive Rate** | 12.0% | 16.8% | -4.8% |
| **AUC-ROC** | 0.967 | 0.912 | +5.5% |

### Modality Contribution Analysis

```
Reentrancy Detection:
  Static:   ██████░░░░ 20%
  Dynamic:  ██████████████████████ 60%
  Semantic: ████░░░░░░ 20%

Access Control:
  Static:   ██████████████████████ 70%
  Dynamic:  ██░░░░░░░░ 10%
  Semantic: ████░░░░░░ 20%

Integer Overflow:
  Static:   ██████████████░░░░░░ 30%
  Dynamic:  ████████████████░░░░ 40%
  Semantic: ██████████████░░░░░░ 30%
```

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_triton.py::TestStaticEncoder -v
python -m pytest tests/test_triton.py::TestDynamicEncoder -v
python -m pytest tests/test_triton.py::TestCrossModalFusion -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ --cov=. --cov-report=html
```

### Manual Testing

```bash
# Test individual components
python -c "from encoders.static_encoder import StaticEncoder; print('✅ Static encoder imports successfully')"
python -c "from encoders.dynamic_encoder import DynamicEncoder; print('✅ Dynamic encoder imports successfully')"
python -c "from fusion.cross_modal_fusion import CrossModalFusion; print('✅ Fusion module imports successfully')"

# Test external tools
slither --help
myth --help
```

## 🔧 Configuration

### CLI Options

```bash
python main.py CONTRACT_FILE [OPTIONS]

Options:
  --contract-name TEXT            Name of the contract to analyze
  --target-vulnerability CHOICE   Specific vulnerability type to target
                                  [reentrancy|overflow|underflow|access_control|
                                   unchecked_call|timestamp_dependency|tx_origin|
                                   delegatecall|self_destruct|gas_limit]
  --confidence-threshold FLOAT    Confidence threshold for early stopping [0.9]
  --max-iterations INTEGER        Maximum analysis iterations [5]
  --device [cpu|cuda]            Computation device [cpu]
  --output PATH                  Output file for results (JSON format)
  --verbose                      Enable verbose logging
  --help                         Show help message
```

### Environment Variables

```bash
# Optional: Configure model cache directory
export TRANSFORMERS_CACHE=/path/to/model/cache

# Optional: Configure logging level
export TRITON_LOG_LEVEL=DEBUG

# Optional: Configure device
export TRITON_DEVICE=cuda
```

## 📊 Output Format

### JSON Results Structure

```json
{
  "final_result": {
    "vulnerability_detected": true,
    "vulnerability_type": "reentrancy",
    "confidence": 0.943,
    "reasoning": "Refinement analysis combining all modalities...",
    "phase": "refinement",
    "modality_contributions": {
      "static": 0.201,
      "dynamic": 0.612,
      "semantic": 0.187
    }
  },
  "workflow_summary": {
    "total_iterations": 2,
    "early_stopping": true,
    "final_confidence": 0.943,
    "phases_executed": ["initial", "deep_dynamic", "refinement"]
  },
  "analysis_history": [
    {
      "phase": "initial",
      "confidence": 0.756,
      "vulnerability_type": "reentrancy",
      "reasoning": "Initial analysis detected reentrancy...",
      "modality_contributions": {...}
    }
  ]
}
```

## 🤝 Contributing

We welcome contributions to improve Triton! Here's how to get started:

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/Triton.git
cd Triton

# Create development environment
python -m venv dev_env
source dev_env/bin/activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Run pre-commit checks
black .
flake8 .
mypy .
```

### Contribution Guidelines

1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality
3. **Follow code style**: Use Black for formatting
4. **Update documentation** for API changes
5. **Commit messages**: Use past tense with capital letters
   - ✅ "Added support for new vulnerability type"
   - ✅ "Fixed memory leak in dynamic encoder"
   - ✅ "Improved cross-modal attention performance"

### Areas for Contribution

- 🐛 **Bug Fixes**: Report and fix issues
- 🚀 **Performance**: Optimize model inference speed
- 📊 **Datasets**: Contribute labeled vulnerability datasets
- 🔍 **Vulnerability Types**: Add support for new vulnerability patterns
- 📚 **Documentation**: Improve guides and examples
- 🧪 **Testing**: Expand test coverage

## 📚 Research & Citations

### Publications

```bibtex
@article{triton2024,
  title={Triton: Agentic Multimodal Representation for Smart Contract Vulnerability Detection},
  author={[Authors]},
  journal={[Journal]},
  year={2024},
  url={https://github.com/yourusername/Triton}
}
```

### Related Work

- **GraphCodeBERT**: [Microsoft Research](https://github.com/microsoft/CodeBERT)
- **Slither**: [Trail of Bits](https://github.com/crytic/slither)
- **Mythril**: [ConsenSys](https://github.com/ConsenSys/mythril)
- **Smart Contract Vulnerabilities**: [SWC Registry](https://swcregistry.io/)

## 🔐 Security Considerations

### Limitations

- **Experimental Research**: Not intended for production security audits
- **Tool Dependencies**: Accuracy depends on Slither and Mythril capabilities
- **Model Bias**: Training data may contain biases affecting detection
- **Computational Requirements**: GPU recommended for optimal performance

### Responsible Use

- ✅ **Research and Education**: Academic studies and learning
- ✅ **Development Testing**: Pre-deployment vulnerability scanning
- ✅ **Tool Comparison**: Benchmarking against other detection methods
- ❌ **Production Audits**: Do not rely solely on Triton for security audits
- ❌ **Financial Decisions**: Do not base investment decisions on Triton results

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Microsoft Research** for GraphCodeBERT
- **Trail of Bits** for Slither static analyzer
- **ConsenSys** for Mythril symbolic execution engine
- **PyTorch Geometric** team for graph neural network tools
- **Hugging Face** for transformer model infrastructure
- **Smart Contract Security Community** for vulnerability research

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/Triton/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/Triton/discussions)
- **Email**: [your.email@example.com]
- **Twitter**: [@yourusername]

## 📈 Roadmap

### Version 1.1 (Coming Soon)
- [ ] Real-time vulnerability monitoring
- [ ] Web interface for contract analysis
- [ ] Integration with popular development frameworks
- [ ] Enhanced visualization of vulnerability patterns

### Version 1.2 (Future)
- [ ] Support for additional blockchain platforms
- [ ] Federated learning for privacy-preserving model updates
- [ ] Advanced explainability features
- [ ] REST API for programmatic access

---

**⚡ Ready to detect vulnerabilities like never before? Get started with Triton today!**

```bash
git clone https://github.com/yourusername/Triton.git
cd Triton && python main.py your_contract.sol
```

*Built with ❤️ for the smart contract security community*