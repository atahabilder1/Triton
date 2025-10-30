# 📖 How to Read This Documentation

Welcome to the Triton documentation! This guide will help you navigate the documentation like a book.

---

## 🗺️ Navigation Map

```
START HERE → docs/README.md (Table of Contents)
                  │
                  ├─→ Quick Start Path (1 hour)
                  │   └─→ 01 → 02 → 13 → 13-example
                  │
                  ├─→ Technical Path (4 hours)
                  │   └─→ 03 → 04-05 → 06 → 14
                  │
                  └─→ Thesis Path (2 hours)
                      └─→ 01 → 10 → 11-12 → 13
```

---

## 📚 Reading Like a Book

### Chapter Structure

Each chapter follows this format:

```
[← Previous Chapter] | [Back to Index] | [Next Chapter →]

# Chapter X: Title

## Table of Contents
- Section 1
- Section 2
- ...

## Content
...

[← Previous] | [Back to Index] | [Next →]
```

### How to Navigate

1. **Start**: Open `docs/README.md`
2. **Click a chapter**: Links take you to that chapter
3. **Navigate**: Use ← Previous | Next → at top/bottom
4. **Return**: Click "Back to Index" anytime

---

## 🎯 Reading Paths

### 🔰 Path 1: Absolute Beginner (1 hour)

**Goal**: Understand what Triton is and run your first test

```
1. Read: 01-project-overview.md (10 min)
   - What is Triton?
   - What problem does it solve?
   - What are the 3 novel contributions?

2. Read: 02-quick-start.md (20 min)
   - Installation
   - Running first test
   - Understanding output

3. Read: 13-testing-guide.md (20 min)
   - How to test all vulnerabilities
   - How to interpret results
   - Understanding the breakdown table

4. Read: 13-example-output.md (10 min)
   - See what results look like
   - Understand the format
```

---

### 🔬 Path 2: Technical Understanding (4 hours)

**Goal**: Understand how Triton works internally

```
1. Read: 03-system-architecture.md (30 min)
   - Overall architecture
   - How components work together
   - Data flow pipeline

2. Read: 04-05-gat-lstm-explained.md (60 min)
   - What is GAT? (30 min)
   - What is LSTM? (30 min)
   - How they work together

3. Quick Reference: 04-05-gat-lstm-quick-reference.md (15 min)
   - Quick lookup
   - Key differences
   - Important points for thesis defense

4. Read: 06-pdg-explained.md (30 min)
   - What is PDG?
   - How PDG is constructed
   - Why PDG as GAT input?

5. Read: 14-training-guide.md (30 min)
   - Do I need to train?
   - What happens without training?
   - Training requirements

6. Optional Deep Dive:
   - Static encoder implementation
   - Dynamic encoder implementation
   - Fusion module details
```

---

### 🎓 Path 3: Thesis Preparation (2 hours)

**Goal**: Prepare for thesis defense and paper writing

```
1. Read: 01-project-overview.md (15 min)
   - Novel contributions
   - Key features
   - Target performance

2. Read: 10-datasets-overview.md (30 min)
   - SmartBugs Curated (manual)
   - FORGE (LLM-extracted)
   - Comparison and citations

3. Read: 10-datasets-summary.md (15 min)
   - What to tell your professor
   - Key distinctions
   - Attribution requirements

4. Read: 11-smartbugs-curated.md (15 min)
   - 143 contracts
   - 10 vulnerability categories
   - Manual curation process

5. Read: 12-forge-dataset.md (15 min)
   - 81,390 contracts
   - LLM-driven construction
   - 95.6% precision

6. Read: 13-testing-guide.md (30 min)
   - How to test
   - Results format
   - Performance metrics

7. Optional:
   - Related work comparison
   - Paper writing templates
   - Presentation guide
```

---

### 🚀 Path 4: Implementation Focus (3 hours)

**Goal**: Understand and modify the code

```
1. Read: 03-system-architecture.md (30 min)
   - Architecture overview
   - Component interactions

2. Read: 15-static-encoder-implementation.md (45 min)
   - GAT implementation
   - PDG processing
   - Code walkthrough

3. Read: 16-dynamic-encoder-implementation.md (45 min)
   - LSTM implementation
   - Trace processing
   - Temporal patterns

4. Read: 17-semantic-encoder-implementation.md (30 min)
   - GraphCodeBERT setup
   - Fine-tuning strategy

5. Read: 18-fusion-implementation.md (30 min)
   - Cross-attention mechanism
   - Adaptive weighting
```

---

## 📖 Chapter Summary

### Part I: Getting Started
- **Chapter 01**: Project overview, contributions, goals
- **Chapter 02**: Quick start, installation, first test
- **Chapter 03**: System architecture, complete design

### Part II: Technical Concepts
- **Chapter 04-05**: GAT and LSTM explained in detail
- **Chapter 06**: PDG (Program Dependency Graph)
- **Chapter 07**: GraphCodeBERT integration (planned)
- **Chapter 08**: Cross-modal fusion (planned)
- **Chapter 09**: Agentic orchestration (planned)

### Part III: Datasets & Testing
- **Chapter 10**: Datasets overview and comparison
- **Chapter 11**: SmartBugs Curated (143 contracts)
- **Chapter 12**: FORGE (81,390 contracts)
- **Chapter 13**: Testing guide and examples
- **Chapter 14**: Training requirements

### Part IV: Implementation (Planned)
- **Chapter 15-19**: Detailed code walkthroughs
- **Chapter 20-23**: Vulnerability detection strategies

### Part V: Research (Planned)
- **Chapter 24-26**: Related work, comparison, contributions

### Part VI: Thesis (Planned)
- **Chapter 27-29**: Thesis writing, presentation, defense

### Part VII: Reference (Planned)
- **Chapter 30-34**: API docs, config, troubleshooting, glossary

---

## 🔍 Quick Find

### I want to...

**...understand what Triton is**
→ Read: `01-project-overview.md`

**...run my first test**
→ Read: `02-quick-start.md`

**...understand GAT and LSTM**
→ Read: `04-05-gat-lstm-explained.md`

**...know about datasets**
→ Read: `10-datasets-overview.md`

**...see test results format**
→ Read: `13-example-output.md`

**...know if I need training**
→ Read: `14-training-guide.md`

**...prepare my thesis**
→ Read: `10-datasets-summary.md` + `01-project-overview.md`

**...see system architecture**
→ Read: `03-system-architecture.md`

---

## 💡 Reading Tips

### 1. **Use the Navigation Links**
Every chapter has:
- `← Previous` at the top
- `Next →` at the top
- `Back to Index` anytime

### 2. **Follow a Path**
Don't try to read everything at once. Pick a path:
- Beginner? → Quick Start Path
- Technical? → Technical Path
- Thesis? → Thesis Path

### 3. **Use Quick Reference**
For GAT/LSTM, use the quick reference card instead of full chapter if you're in a hurry.

### 4. **Bookmark Important Chapters**
Key chapters to bookmark:
- `README.md` - Table of contents
- `02-quick-start.md` - Quick start
- `13-testing-guide.md` - Testing
- `04-05-gat-lstm-quick-reference.md` - Quick ref

### 5. **Check Status**
Look for these icons:
- ✅ Complete and ready
- ⏳ In progress
- 📝 Planned

---

## 🎯 Recommended Order

### For First-Time Readers:
```
1. 01-project-overview.md
2. 02-quick-start.md
3. 03-system-architecture.md
4. 04-05-gat-lstm-explained.md
5. 13-testing-guide.md
```

### For Thesis Writing:
```
1. 01-project-overview.md
2. 10-datasets-overview.md
3. 10-datasets-summary.md
4. 13-testing-guide.md
5. 14-training-guide.md
```

### For Code Understanding:
```
1. 03-system-architecture.md
2. 04-05-gat-lstm-explained.md
3. 06-pdg-explained.md
4. (Then implementation chapters when ready)
```

---

## 📱 Accessing Documentation

### On GitHub:
```
Navigate to: docs/README.md
Click any link to jump to that chapter
```

### Locally:
```
Open: docs/README.md in your browser or markdown viewer
All links work with relative paths
```

### In IDE:
```
Most IDEs (VS Code, PyCharm) render markdown with clickable links
Open docs/README.md and start clicking!
```

---

## ❓ Still Confused?

1. **Start here**: `docs/README.md`
2. **Quick start**: `docs/02-quick-start.md`
3. **Can't find something?**: `docs/DOCUMENTATION_INDEX.md`

**Happy reading! 📚**

---

[Back to Documentation Home](README.md)
