# 📚 Triton Documentation

Welcome to the comprehensive documentation for **Triton v2.0** - A Multi-Modal Smart Contract Vulnerability Detection System.

---

## 📖 Table of Contents

### Part I: Introduction & Overview
1. [**Project Overview**](01-project-overview.md)
   - What is Triton?
   - Key Features
   - Novel Contributions
   - Target Performance

2. [**Quick Start Guide**](02-quick-start.md)
   - Installation
   - Running Your First Test
   - Understanding Results
   - Common Issues

3. [**System Architecture**](03-system-architecture.md)
   - Overall Design
   - Component Overview
   - Data Flow Pipeline
   - Architecture Diagrams

---

### Part II: Technical Deep Dive

4. [**GAT (Graph Attention Networks)**](04-gat-explained.md)
   - What is GAT?
   - Why GAT for Smart Contracts?
   - How GAT Works
   - Implementation Details

5. [**LSTM (Long Short-Term Memory)**](05-lstm-explained.md)
   - What is LSTM?
   - Why LSTM for Smart Contracts?
   - How LSTM Works
   - Implementation Details

6. [**PDG (Program Dependency Graph)**](06-pdg-explained.md)
   - What is PDG?
   - PDG Construction
   - PDG as Static Encoder Input
   - Code Examples

7. [**GraphCodeBERT Integration**](07-graphcodebert.md)
   - Semantic Understanding
   - Fine-Tuning Strategy
   - Vulnerability-Aware Training
   - Implementation

8. [**Cross-Modal Fusion**](08-fusion-module.md)
   - Intelligent Fusion Strategy
   - Adaptive Attention
   - Contribution #2 Explained
   - Code Walkthrough

9. [**Agentic Orchestration**](09-agentic-orchestration.md)
   - RL-Based Decision Making
   - Iterative Refinement
   - Contribution #3 Explained
   - Workflow Details

---

### Part III: Datasets & Testing

10. [**Datasets Overview**](10-datasets-overview.md)
    - SmartBugs Curated
    - FORGE Artifacts
    - Dataset Comparison
    - Attribution & Citations

11. [**SmartBugs Curated Dataset**](11-smartbugs-curated.md)
    - Dataset Details (143 contracts)
    - Vulnerability Categories
    - Manual Curation Process
    - Usage Guide

12. [**FORGE Dataset**](12-forge-dataset.md)
    - Dataset Details (81,390 contracts)
    - LLM-Driven Construction
    - CWE Classification
    - Usage Guide

13. [**Testing Guide**](13-testing-guide.md)
    - Running Tests
    - Interpreting Results
    - Vulnerability Breakdown Tables
    - Performance Metrics

14. [**Training Guide**](14-training-guide.md)
    - Do I Need to Train?
    - Training Requirements
    - Training Pipeline
    - Expected Performance

---

### Part IV: Implementation Details

15. [**Static Encoder Implementation**](15-static-encoder-implementation.md)
    - Code Structure
    - GAT Layers
    - PDG Processing
    - Feature Extraction

16. [**Dynamic Encoder Implementation**](16-dynamic-encoder-implementation.md)
    - Code Structure
    - LSTM Layers
    - Trace Processing
    - Temporal Pattern Detection

17. [**Semantic Encoder Implementation**](17-semantic-encoder-implementation.md)
    - GraphCodeBERT Setup
    - Tokenization
    - Fine-Tuning
    - Output Processing

18. [**Fusion Module Implementation**](18-fusion-implementation.md)
    - Cross-Attention Mechanism
    - Weight Calculation
    - Feature Combination
    - Output Generation

19. [**Orchestrator Implementation**](19-orchestrator-implementation.md)
    - Workflow Phases
    - Confidence Evaluation
    - Decision Engine
    - Iterative Refinement

---

### Part V: Vulnerability Detection

20. [**Reentrancy Detection**](20-reentrancy-detection.md)
    - Pattern Recognition
    - Multi-Modal Analysis
    - Detection Strategy
    - Examples

21. [**Overflow/Underflow Detection**](21-overflow-detection.md)
    - Arithmetic Patterns
    - Detection Methods
    - Solidity 0.8.x Considerations
    - Examples

22. [**Access Control Detection**](22-access-control-detection.md)
    - Permission Patterns
    - Modifier Analysis
    - Detection Strategy
    - Examples

23. [**Other Vulnerabilities**](23-other-vulnerabilities.md)
    - Unchecked Calls
    - Timestamp Dependency
    - Bad Randomness
    - Front Running
    - Denial of Service

---

### Part VI: Research & Comparison

24. [**Related Work**](24-related-work.md)
    - Existing Tools (Slither, Mythril, etc.)
    - Previous Approaches
    - Limitations
    - How Triton is Different

25. [**Performance Comparison**](25-performance-comparison.md)
    - Baseline Tools
    - Benchmark Results
    - Speed Comparison
    - Accuracy Metrics

26. [**Novel Contributions**](26-novel-contributions.md)
    - Contribution #1: Vulnerability-Aware Fine-Tuning
    - Contribution #2: Intelligent Adaptive Fusion
    - Contribution #3: RL-Based Orchestration
    - Research Impact

---

### Part VII: For Your Thesis

27. [**Thesis Guide**](27-thesis-guide.md)
    - How to Present Triton
    - Key Points for Defense
    - Common Questions
    - Talking Points

28. [**Paper Writing Guide**](28-paper-writing.md)
    - Abstract Template
    - Introduction Structure
    - Methodology Section
    - Results Presentation

29. [**Presentation Guide**](29-presentation-guide.md)
    - Slide Structure
    - Visualizations
    - Demo Strategy
    - Q&A Preparation

---

### Part VIII: Appendices

30. [**API Reference**](30-api-reference.md)
    - Class Documentation
    - Function References
    - Parameters
    - Return Values

31. [**Configuration Guide**](31-configuration.md)
    - Config Files
    - Hyperparameters
    - Environment Setup
    - Customization

32. [**Troubleshooting**](32-troubleshooting.md)
    - Common Errors
    - Solutions
    - Debugging Tips
    - FAQ

33. [**Glossary**](33-glossary.md)
    - Technical Terms
    - Acronyms
    - Definitions
    - References

34. [**Bibliography**](34-bibliography.md)
    - Papers Cited
    - Tools Referenced
    - Datasets
    - External Resources

---

## 🚀 Quick Navigation

### For Beginners
Start here: [Quick Start Guide](02-quick-start.md) → [Testing Guide](13-testing-guide.md)

### For Understanding the Architecture
Read: [System Architecture](03-system-architecture.md) → [GAT](04-gat-explained.md) → [LSTM](05-lstm-explained.md) → [Fusion](08-fusion-module.md)

### For Implementation
Check: [Static Encoder](15-static-encoder-implementation.md) → [Dynamic Encoder](16-dynamic-encoder-implementation.md) → [Orchestrator](19-orchestrator-implementation.md)

### For Your Thesis
See: [Thesis Guide](27-thesis-guide.md) → [Paper Writing](28-paper-writing.md) → [Presentation Guide](29-presentation-guide.md)

### For Testing & Results
Go to: [Testing Guide](13-testing-guide.md) → [Performance Comparison](25-performance-comparison.md)

---

## 📊 Visual Guide

```
┌─────────────────────────────────────────────────────────────┐
│                    TRITON SYSTEM                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Smart Contract                                             │
│       │                                                     │
│       ├──────────┬──────────┬──────────┐                  │
│       │          │          │          │                  │
│       ↓          ↓          ↓          ↓                  │
│   [Static]   [Dynamic]  [Semantic]  [Other]              │
│    (GAT)      (LSTM)  (GraphCodeBERT)                    │
│       │          │          │          │                  │
│       └──────────┴──────────┴──────────┘                  │
│                  │                                         │
│                  ↓                                         │
│          [Fusion Module]                                  │
│                  │                                         │
│                  ↓                                         │
│          [Orchestrator]                                   │
│                  │                                         │
│                  ↓                                         │
│       [Vulnerability Report]                              │
│                                                            │
└────────────────────────────────────────────────────────────┘

Read more: Chapter 3 - System Architecture
```

---

## 🎯 Learning Paths

### Path 1: Quick Understanding (1 hour)
1. [Project Overview](01-project-overview.md) (10 min)
2. [Quick Start](02-quick-start.md) (20 min)
3. [System Architecture](03-system-architecture.md) (15 min)
4. [Testing Guide](13-testing-guide.md) (15 min)

### Path 2: Technical Deep Dive (4 hours)
1. [GAT Explained](04-gat-explained.md) (45 min)
2. [LSTM Explained](05-lstm-explained.md) (45 min)
3. [PDG Explained](06-pdg-explained.md) (30 min)
4. [Fusion Module](08-fusion-module.md) (30 min)
5. [Orchestration](09-agentic-orchestration.md) (30 min)
6. [Implementation Chapters](15-static-encoder-implementation.md) (1 hour)

### Path 3: Thesis Preparation (2 hours)
1. [Novel Contributions](26-novel-contributions.md) (30 min)
2. [Thesis Guide](27-thesis-guide.md) (30 min)
3. [Paper Writing](28-paper-writing.md) (30 min)
4. [Presentation Guide](29-presentation-guide.md) (30 min)

### Path 4: Complete Mastery (2 days)
Read all chapters in order from 1-34.

---

## 📝 Document Status

| Chapter | Status | Last Updated |
|---------|--------|--------------|
| 01-34 | ✅ Complete | 2025-10-30 |

---

## 🤝 Contributing

This documentation is for the Triton project. For questions or updates:
- Check the [Troubleshooting Guide](32-troubleshooting.md)
- Review the [FAQ](32-troubleshooting.md#faq)
- Consult your supervisor

---

## 📄 License

This documentation is part of the Triton project.

---

## 🎓 Citation

If you use Triton or reference this documentation, please cite:

```bibtex
@misc{triton2025,
  title={Triton: A Multi-Modal Smart Contract Vulnerability Detection System},
  author={Your Name},
  year={2025},
  note={Master's Thesis}
}
```

---

**Last Updated**: October 30, 2025
**Version**: 2.0
**Maintainer**: Triton Project Team

---

## 💡 Tips for Reading

- **🔰 Beginner?** Start with Quick Start Guide
- **🔬 Technical?** Jump to Deep Dive chapters
- **📊 Need Results?** Go to Testing & Performance
- **🎓 Writing Thesis?** Check Thesis Guide first
- **❓ Questions?** See Troubleshooting & FAQ

**Navigate using the links above - happy reading! 📚**
