# How PDG Works as Static Encoder Input

## Quick Answer

**PDG (Program Dependency Graph)** is the **input representation** for the static encoder (GAT). It's not the encoder itself - it's what the encoder analyzes!

```
Solidity Code → [Convert to PDG] → [Feed to GAT] → Static Features
```

---

## The Full Pipeline

### Step 1: Solidity Contract (Raw Code)
```solidity
contract Vulnerable {
    uint public balance;

    function withdraw() public {
        msg.sender.call{value: balance}("");  // Line 4
        balance = 0;                          // Line 5
    }
}
```

### Step 2: Convert to PDG (Program Dependency Graph)
```
Nodes:
  N1: balance (state variable)
  N2: withdraw (function)
  N3: msg.sender (variable)
  N4: Line 4 (external call)
  N5: Line 5 (state change)

Edges (Dependencies):
  E1: N2 → N4 (function contains call)
  E2: N2 → N5 (function contains assignment)
  E3: N1 → N4 (balance is read at line 4)
  E4: N1 → N5 (balance is written at line 5)
  E5: N4 → N5 (line 4 happens before line 5)
  E6: N3 → N4 (msg.sender used in call)
```

### Step 3: Feed PDG to GAT (Static Encoder)
```python
# GAT processes the graph
x = node_features  # Features for each node (N1, N2, N3, N4, N5)
edge_index = edges  # Connections (E1, E2, E3, E4, E5, E6)

output = GAT(x, edge_index)  # GAT analyzes the graph
```

### Step 4: GAT Output (Static Features)
```
768-dimensional vector encoding:
- Structural patterns
- Data dependencies
- Control flow relationships
- Vulnerability signatures
```

---

## Understanding PDG (Program Dependency Graph)

### What is PDG?

**PDG** = A graph representation of code that shows **dependencies**

**Two main types of dependencies:**

1. **Data Dependency**: "Variable X is used by Statement Y"
2. **Control Dependency**: "Statement Y executes only if Condition X is true"

### PDG Components

#### 1. Nodes (Vertices)
Nodes represent **program elements**:
- Variables (balance, msg.sender)
- Statements (assignments, calls, returns)
- Expressions (arithmetic, comparisons)
- Functions
- Control flow elements (if, while, for)

#### 2. Edges (Dependencies)
Edges represent **relationships**:
- **Data flow**: "balance flows into call"
- **Control flow**: "call happens before assignment"
- **Usage**: "function uses variable"
- **Definition**: "statement defines variable"

---

## Example: Building PDG from Code

### Code:
```solidity
contract Example {
    uint balance;  // Line 2

    function withdraw() public {  // Line 4
        require(balance > 0);  // Line 5
        uint amount = balance;  // Line 6
        msg.sender.call{value: amount}("");  // Line 7
        balance = 0;  // Line 8
    }
}
```

### PDG Construction:

#### Nodes:
```
N1: balance (state variable)
N2: withdraw (function)
N3: Line 5 (require statement)
N4: Line 6 (local variable assignment)
N5: amount (local variable)
N6: msg.sender (special variable)
N7: Line 7 (external call)
N8: Line 8 (state update)
```

#### Data Dependencies (Data Flow Edges):
```
N1 → N3  (balance is READ by require)
N1 → N4  (balance is READ by assignment)
N4 → N5  (assignment DEFINES amount)
N5 → N7  (amount is READ by call)
N6 → N7  (msg.sender is READ by call)
N1 → N8  (balance is WRITTEN by assignment)
```

#### Control Dependencies (Control Flow Edges):
```
N2 → N3  (require is first in function)
N3 → N4  (assignment executes if require passes)
N4 → N7  (call executes after assignment)
N7 → N8  (state update executes after call)
```

#### Visual PDG:
```
         ┌──────────┐
         │ balance  │ (N1)
         └────┬─────┘
              │ data flow
    ┌─────────┼─────────┬────────┐
    │         │         │        │
    ↓         ↓         ↓        ↓
┌────────┐ ┌──────┐ ┌──────┐ ┌──────┐
│require │ │amount│ │ call │ │= 0   │
│(Line 5)│ │(L 6) │ │(L 7) │ │(L 8) │
└────┬───┘ └───┬──┘ └───┬──┘ └──────┘
     │         │        │
     │ control │ control│ control
     ↓    flow ↓   flow ↓   flow
     └─────────┴────────┴──────────►
           (Sequential execution)
```

---

## Why PDG for Static Analysis?

### Traditional Approach (Without PDG)
```
Just parse code line by line:
  Line 1: uint balance;
  Line 2: function withdraw()
  Line 3: ...

Problem: Miss relationships!
- Which line uses which variable?
- What's the execution order?
- What depends on what?
```

### PDG Approach (With Graph)
```
Build dependency graph:
  Node: balance
  Edge: balance → external_call (data dependency)
  Edge: external_call → state_change (control dependency)

Benefit: Captures relationships!
- See data flow clearly
- Understand control flow
- Detect dependency violations
```

---

## PDG Types in Triton

### 1. Control Dependency Graph (CDG)
Shows **control flow** relationships

```solidity
if (condition) {
    doSomething();  // Control-dependent on condition
}
```

PDG:
```
[condition] ──controls──> [doSomething()]
```

### 2. Data Dependency Graph (DDG)
Shows **data flow** relationships

```solidity
uint x = 10;
uint y = x + 5;  // Data-dependent on x
```

PDG:
```
[x = 10] ──data flows──> [y = x + 5]
```

### 3. Combined PDG (What Triton Uses)
Combines **both** control and data dependencies

```solidity
if (balance > 0) {
    uint amt = balance;
    send(amt);
}
```

PDG:
```
[balance > 0] ──control──> [amt = balance]
                                  │
                            data flow
                                  ↓
[balance > 0] ──control──> [send(amt)]
```

---

## How GAT Uses PDG

### PDG as Input Format

```python
# In static_encoder.py

class StaticEncoder(nn.Module):
    def forward(self, data):
        # data is a PyTorch Geometric Data object
        x = data.x           # Node features (N × F matrix)
        edge_index = data.edge_index  # Edge connections (2 × E matrix)

        # x shape: [num_nodes, feature_dim]
        # Example: [10 nodes, 128 features each]

        # edge_index shape: [2, num_edges]
        # Example: [[0, 0, 1, 2, 3],  ← source nodes
        #           [1, 2, 2, 3, 4]]  ← target nodes
        # Means: 0→1, 0→2, 1→2, 2→3, 3→4
```

### Example: PDG to GAT Input

**PDG:**
```
balance (N0) ──reads──> call (N1) ──before──> assignment (N2)
```

**GAT Input:**
```python
# Node features (simplified)
x = torch.tensor([
    [1.0, 0.0, 0.5],  # N0: balance (state variable)
    [0.0, 1.0, 0.8],  # N1: call (external call)
    [0.0, 0.0, 1.0],  # N2: assignment (state change)
])

# Edge connections
edge_index = torch.tensor([
    [0, 1],  # Source nodes: N0, N1
    [1, 2],  # Target nodes: N1, N2
])
# This means: N0→N1 (balance→call), N1→N2 (call→assignment)
```

**GAT Processing:**
```python
# Layer 1: Each node looks at its neighbors
out1 = self.gat1(x, edge_index)
# N0 considers: itself
# N1 considers: itself + N0 (balance)
# N2 considers: itself + N1 (call)

# Layer 2: Multi-hop attention
out2 = self.gat2(out1, edge_index)
# N0 considers: itself
# N1 considers: itself + N0
# N2 considers: itself + N1 + N0 (through N1)
#               ↑ Now N2 knows about the full pattern!

# Layer 3: Global pattern recognition
out3 = self.gat3(out2, edge_index)
# N2 now encodes: "balance → call → assignment" = REENTRANCY!
```

---

## PDG Generation in Triton

### Your Code (utils/pdg_builder.py)

```python
class PDGBuilder:
    def build_pdg(self, ast_tree):
        """Build PDG from AST"""
        # Step 1: Extract nodes from AST
        nodes = self._extract_nodes(ast_tree)

        # Step 2: Build control flow edges
        cfg_edges = self._build_control_flow(nodes)

        # Step 3: Build data flow edges
        dfg_edges = self._build_data_flow(nodes)

        # Step 4: Combine into PDG
        pdg = self._combine_graphs(nodes, cfg_edges, dfg_edges)

        return pdg
```

### How It Works:

#### Step 1: Parse Solidity to AST
```
Solidity Code → Solidity Parser → Abstract Syntax Tree (AST)
```

AST Example:
```json
{
  "type": "ContractDefinition",
  "name": "Vulnerable",
  "functions": [
    {
      "type": "FunctionDefinition",
      "name": "withdraw",
      "body": {
        "statements": [
          {"type": "ExpressionStatement", "expression": "call"},
          {"type": "ExpressionStatement", "expression": "assignment"}
        ]
      }
    }
  ]
}
```

#### Step 2: Extract Nodes from AST
```python
nodes = [
    {"id": 0, "type": "variable", "name": "balance"},
    {"id": 1, "type": "call", "line": 4},
    {"id": 2, "type": "assignment", "line": 5}
]
```

#### Step 3: Build Edges (Dependencies)
```python
edges = [
    {"from": 0, "to": 1, "type": "data_read"},      # balance read by call
    {"from": 0, "to": 2, "type": "data_write"},     # balance written by assignment
    {"from": 1, "to": 2, "type": "control_flow"}    # call before assignment
]
```

#### Step 4: Create PyTorch Geometric Graph
```python
from torch_geometric.data import Data

# Convert to PyTorch Geometric format
x = torch.tensor(node_features)  # Node features
edge_index = torch.tensor([
    [0, 0, 1],  # Source nodes
    [1, 2, 2]   # Target nodes
])

pdg_data = Data(x=x, edge_index=edge_index)
```

---

## PDG Features (Node Attributes)

### What Information is in Each Node?

```python
# Example node features for "balance" variable
node_features = [
    # Basic properties
    1.0,  # is_state_variable
    0.0,  # is_local_variable
    0.0,  # is_function
    0.0,  # is_call

    # Type information
    1.0,  # is_uint
    0.0,  # is_address
    0.0,  # is_mapping

    # Usage patterns
    0.5,  # read_count (normalized)
    0.3,  # write_count (normalized)

    # Visibility
    1.0,  # is_public
    0.0,  # is_private

    # ... more features (total ~128 dimensions)
]
```

---

## Why "Static" Encoder?

### Static = Analyzes Code Structure (Not Runtime)

**Static Analysis:**
- Looks at **code** (not execution)
- Analyzes **structure** (not behavior)
- Works on **PDG** (graph representation)
- Fast (no need to run code)

**Dynamic Analysis:**
- Looks at **execution traces** (runtime)
- Analyzes **behavior** (not just structure)
- Works on **sequences** (event logs)
- Slower (needs to simulate/run code)

### Comparison:

```
Static Encoder (GAT on PDG):
Input:  Program Dependency Graph
Speed:  Fast ⚡ (just analyze structure)
Sees:   All possible paths
Misses: Runtime-only vulnerabilities

Dynamic Encoder (LSTM on Traces):
Input:  Execution traces
Speed:  Slower 🐢 (need to simulate)
Sees:   Actual behavior
Misses: Unexecuted paths
```

---

## Complete Flow in Triton

### From Code to Vulnerability Detection

```
┌─────────────────┐
│ Solidity Code   │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Parse to AST    │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Build PDG       │ ← PDG Builder (utils/pdg_builder.py)
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ GAT Processing  │ ← Static Encoder (encoders/static_encoder.py)
│ (3 layers)      │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Static Features │ (768-dim vector)
│ (structural)    │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Fusion Module   │ ← Combine with LSTM + BERT
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Vulnerability   │
│ Detection       │
└─────────────────┘
```

---

## Real Example: Reentrancy Detection

### Code:
```solidity
function withdraw() public {
    uint amt = balances[msg.sender];
    msg.sender.call{value: amt}("");
    balances[msg.sender] = 0;
}
```

### PDG:
```
Nodes:
  N0: balances (state variable)
  N1: msg.sender (special variable)
  N2: amt (local variable)
  N3: line 2 (read balances)
  N4: line 3 (external call)
  N5: line 4 (write balances)

Edges:
  N0 → N3 (data: balances read)
  N3 → N2 (data: assign to amt)
  N2 → N4 (data: amt used in call)
  N1 → N4 (data: msg.sender used in call)
  N3 → N4 (control: read before call)
  N4 → N5 (control: call before write)
  N0 → N5 (data: balances written)
```

### GAT Processing:
```
Layer 1: Learn local patterns
  N4 (call) sees: N2 (amt), N1 (msg.sender), N3 (read)
  → "External call uses balance amount"

Layer 2: Learn regional patterns
  N5 (write) sees: N4 (call), N0 (balances)
  → "State written after external call"

Layer 3: Learn global patterns
  N5 encodes full pattern: read → call → write
  → "REENTRANCY PATTERN DETECTED!" 🚨
```

### Output:
```python
static_features = [
    0.85,  # reentrancy_score (HIGH!)
    0.12,  # overflow_score
    0.03,  # access_control_score
    ...    # (768 total dimensions)
]
```

---

## Key Takeaways

### 1. PDG is Not the Encoder
```
PDG        = Input representation (graph)
GAT        = Encoder (neural network)
PDG → GAT  = Static encoding pipeline
```

### 2. Why PDG?
```
✅ Captures dependencies (data + control flow)
✅ Graph structure perfect for GAT
✅ Represents all possible execution paths
✅ Enables structural vulnerability detection
```

### 3. PDG vs Other Representations
```
AST:    Tree structure (hierarchy)
CFG:    Control flow only (execution order)
DFG:    Data flow only (variable usage)
PDG:    Combined (control + data) ← Best for vulnerability detection!
```

### 4. GAT's Role
```
Input:  PDG (graph structure)
Process: Attention-based graph convolution
Output: Static feature vector (768-dim)
Purpose: Detect structural vulnerability patterns
```

---

## Summary

**PDG (Program Dependency Graph)** is the **structured representation** of code that the static encoder (GAT) analyzes:

1. **Code** → **PDG** (via PDG Builder)
2. **PDG** → **GAT** (Static Encoder processes it)
3. **GAT** → **Features** (768-dim vector of structural patterns)

**PDG captures relationships**, **GAT learns from them**. Together they form the static analysis component of Triton!

---

## For Your Professor

**Simple:** "PDG is a graph representation of code dependencies. We feed it to GAT to learn structural vulnerability patterns."

**Detailed:** "We convert smart contracts into Program Dependency Graphs capturing both control and data dependencies. Our GAT-based static encoder uses multi-head attention to learn vulnerability signatures from this graph structure, achieving structural pattern recognition that traditional static analyzers miss."

---

**Bottom Line**: PDG is the **input format** (graph of code dependencies), GAT is the **encoder** (neural network that processes it)! 🎯
