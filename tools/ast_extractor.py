#!/usr/bin/env python3
"""
AST (Abstract Syntax Tree) Extractor for Solidity
Uses solc directly to extract AST without requiring dependency resolution.
Much more reliable than PDG extraction for contracts with external dependencies.
"""

import subprocess
import json
import tempfile
import os
import re
from typing import Dict, List, Optional, Any, Tuple
import networkx as nx
import logging

logger = logging.getLogger(__name__)


class ASTExtractor:
    """Extract AST from Solidity contracts using solc compiler."""

    def __init__(self, timeout: int = 60):
        self.timeout = timeout

        # Node types we care about for vulnerability detection
        self.important_node_types = {
            'ContractDefinition', 'FunctionDefinition', 'ModifierDefinition',
            'VariableDeclaration', 'IfStatement', 'WhileStatement', 'ForStatement',
            'FunctionCall', 'Assignment', 'Return', 'EmitStatement',
            'InlineAssembly', 'TryCatchClause', 'RevertStatement'
        }

    def _detect_solc_version(self, source_code: str) -> str:
        """Detect required Solidity version from pragma statement."""
        pragma_pattern = r'pragma\s+solidity\s+([^;]+);'
        match = re.search(pragma_pattern, source_code)

        if not match:
            return '0.6.12'  # Default

        version_spec = match.group(1).strip()

        # Extract full version number (0.8.13, ^0.8.0, etc.)
        # Try to get the exact version first
        exact_version_match = re.search(r'(\d+\.\d+\.\d+)', version_spec)
        if exact_version_match:
            exact_version = exact_version_match.group(1)
            # Return exact version if it looks reasonable
            return exact_version

        # Otherwise use major.minor
        version_match = re.search(r'(\d+\.\d+)', version_spec)
        if not version_match:
            return '0.6.12'

        major_minor = version_match.group(1)

        # Map to available versions
        version_map = {
            '0.4': '0.4.26',
            '0.5': '0.5.17',
            '0.6': '0.6.12',
            '0.7': '0.7.6',
            '0.8': '0.8.13'  # Changed to 0.8.13 (installed version)
        }

        return version_map.get(major_minor, '0.6.12')

    def _set_solc_version(self, version: str) -> bool:
        """Set solc version using solc-select."""
        try:
            result = subprocess.run(
                ['solc-select', 'use', version],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            logger.debug(f"Failed to set solc version {version}: {e}")
            return False

    def _remove_imports_and_inheritance(self, source_code: str) -> str:
        """
        Remove import statements and external inheritance from source code.
        This allows AST extraction even when dependencies are missing.
        We'll get partial AST (just the current contract) which is still useful for structure.
        """
        lines = source_code.split('\n')
        filtered_lines = []

        for line in lines:
            # Skip import statements
            if re.match(r'^\s*import\s+', line.strip()):
                # Replace with comment to preserve line numbers
                filtered_lines.append('// ' + line)
                continue

            # Remove "is BaseContract" inheritance clauses
            # Match: contract Foo is Bar, Baz {
            # Replace with: contract Foo {
            modified_line = re.sub(
                r'(\bcontract\s+\w+)\s+is\s+[^{]+(\{)',
                r'\1 \2',
                line
            )

            filtered_lines.append(modified_line)

        return '\n'.join(filtered_lines)

    def extract_ast(self, source_code: str) -> Tuple[Optional[Dict], bool]:
        """
        Extract AST from Solidity source code.
        Returns (ast_dict, success)
        """
        temp_file = None
        try:
            # Detect and set appropriate Solidity compiler version
            required_version = self._detect_solc_version(source_code)
            if required_version:
                self._set_solc_version(required_version)

            # Remove imports and inheritance to avoid dependency issues
            # This gives us partial AST (current contract only) but that's still useful
            cleaned_source = self._remove_imports_and_inheritance(source_code)

            # Write cleaned source to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False) as f:
                f.write(cleaned_source)
                temp_file = f.name

            # Run solc to get AST (compact JSON format)
            cmd = ['solc', '--ast-compact-json', temp_file]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            # Clean up temp file
            os.unlink(temp_file)
            temp_file = None

            # Parse output - AST is in stdout after "JSON AST (compact format):"
            stdout = result.stdout

            # Find JSON AST section
            if 'JSON AST (compact format):' in stdout:
                # Extract JSON part - find the first '{' after the source file marker
                # Look for "======= <filename> =======" and then parse JSON after it
                lines = stdout.split('\n')

                json_start_idx = None
                for i, line in enumerate(lines):
                    if line.startswith('=======') and i > 0:
                        # JSON starts on the next line
                        json_start_idx = i + 1
                        break

                if json_start_idx:
                    # Join lines from JSON start to end
                    json_str = '\n'.join(lines[json_start_idx:])
                    try:
                        ast = json.loads(json_str)
                        return ast, True
                    except json.JSONDecodeError as e:
                        logger.debug(f"Failed to parse AST JSON: {e}")
                        return None, False

            # If we couldn't extract AST, return None
            logger.debug(f"No AST found in solc output. stderr: {result.stderr[:200] if result.stderr else 'none'}")
            return None, False

        except subprocess.TimeoutExpired:
            logger.error(f"solc timed out after {self.timeout} seconds")
            return None, False
        except Exception as e:
            logger.debug(f"AST extraction error: {e}")
            return None, False
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass

    def ast_to_graph(self, ast: Dict) -> nx.DiGraph:
        """
        Convert AST JSON to a NetworkX graph.
        Each node represents an AST node, edges represent parent-child relationships.
        """
        graph = nx.DiGraph()

        if not ast or 'nodeType' not in ast:
            return graph

        # Counter for unique node IDs
        self.node_counter = 0

        # Recursively build graph
        self._add_ast_node(graph, ast, parent_id=None)

        return graph

    def _add_ast_node(self, graph: nx.DiGraph, ast_node: Dict, parent_id: Optional[int]) -> int:
        """Recursively add AST nodes to graph."""

        # Get node type
        node_type = ast_node.get('nodeType', 'Unknown')

        # Create unique node ID
        node_id = self.node_counter
        self.node_counter += 1

        # Extract key attributes
        attributes = {
            'nodeType': node_type,
            'id': node_id
        }

        # Add type-specific attributes
        if node_type == 'ContractDefinition':
            attributes['name'] = ast_node.get('name', '')
            attributes['contractKind'] = ast_node.get('contractKind', '')
            attributes['abstract'] = ast_node.get('abstract', False)

        elif node_type == 'FunctionDefinition':
            attributes['name'] = ast_node.get('name', '')
            attributes['visibility'] = ast_node.get('visibility', '')
            attributes['stateMutability'] = ast_node.get('stateMutability', '')
            attributes['implemented'] = ast_node.get('implemented', False)
            attributes['isConstructor'] = ast_node.get('kind', '') == 'constructor'

        elif node_type == 'ModifierDefinition':
            attributes['name'] = ast_node.get('name', '')
            attributes['visibility'] = ast_node.get('visibility', '')

        elif node_type == 'VariableDeclaration':
            attributes['name'] = ast_node.get('name', '')
            attributes['stateVariable'] = ast_node.get('stateVariable', False)
            attributes['visibility'] = ast_node.get('visibility', '')
            attributes['constant'] = ast_node.get('constant', False)

        elif node_type == 'FunctionCall':
            # Mark as important for vulnerability detection
            attributes['important'] = True

        elif node_type in ['InlineAssembly', 'DelegateCall', 'ExternalCall']:
            # These are security-sensitive
            attributes['security_sensitive'] = True

        # Add node to graph
        graph.add_node(node_id, **attributes)

        # Add edge from parent
        if parent_id is not None:
            graph.add_edge(parent_id, node_id, type='ast_child')

        # Recursively process children
        # Children can be in 'nodes', 'body', 'statements', etc.
        children_keys = ['nodes', 'body', 'statements', 'declarations',
                         'trueBody', 'falseBody', 'initializationExpression',
                         'condition', 'loopExpression']

        for key in children_keys:
            if key in ast_node:
                child_nodes = ast_node[key]
                if isinstance(child_nodes, list):
                    for child in child_nodes:
                        if isinstance(child, dict) and 'nodeType' in child:
                            self._add_ast_node(graph, child, node_id)
                elif isinstance(child_nodes, dict) and 'nodeType' in child_nodes:
                    self._add_ast_node(graph, child_nodes, node_id)

        return node_id

    def extract_ast_features(self, source_code: str) -> Dict[str, Any]:
        """
        Extract AST and convert to graph representation.
        Returns a dictionary with AST graph and statistics.
        """
        ast, success = self.extract_ast(source_code)

        if not success or ast is None:
            # Return empty graph
            return {
                'success': False,
                'ast_graph': nx.DiGraph(),
                'stats': {
                    'num_nodes': 0,
                    'num_edges': 0,
                    'num_functions': 0,
                    'num_contracts': 0,
                    'num_modifiers': 0,
                    'num_state_vars': 0,
                    'has_assembly': False,
                    'has_delegatecall': False
                }
            }

        # Convert to graph
        ast_graph = self.ast_to_graph(ast)

        # Compute statistics
        node_types = [data.get('nodeType', '') for _, data in ast_graph.nodes(data=True)]

        stats = {
            'num_nodes': ast_graph.number_of_nodes(),
            'num_edges': ast_graph.number_of_edges(),
            'num_functions': node_types.count('FunctionDefinition'),
            'num_contracts': node_types.count('ContractDefinition'),
            'num_modifiers': node_types.count('ModifierDefinition'),
            'num_state_vars': sum(1 for _, data in ast_graph.nodes(data=True)
                                 if data.get('nodeType') == 'VariableDeclaration'
                                 and data.get('stateVariable', False)),
            'has_assembly': 'InlineAssembly' in node_types,
            'has_delegatecall': any('delegatecall' in str(data).lower()
                                   for _, data in ast_graph.nodes(data=True)),
            'has_external_calls': 'FunctionCall' in node_types
        }

        logger.info(f"Extracted AST with {stats['num_nodes']} nodes, {stats['num_edges']} edges")

        return {
            'success': True,
            'ast_graph': ast_graph,
            'stats': stats
        }


def test_ast_extraction():
    """Test AST extraction on a sample contract."""
    test_contract = """
    pragma solidity ^0.6.0;

    contract Test {
        uint256 public value;

        function setValue(uint256 _value) public {
            value = _value;
        }

        function getValue() public view returns (uint256) {
            return value;
        }
    }
    """

    extractor = ASTExtractor()
    result = extractor.extract_ast_features(test_contract)

    print(f"Success: {result['success']}")
    print(f"Stats: {result['stats']}")
    print(f"Graph nodes: {result['ast_graph'].number_of_nodes()}")
    print(f"Graph edges: {result['ast_graph'].number_of_edges()}")

    # Print some nodes
    print("\nSample nodes:")
    for node_id, data in list(result['ast_graph'].nodes(data=True))[:10]:
        print(f"  Node {node_id}: {data.get('nodeType', 'Unknown')} - {data.get('name', '')}")


if __name__ == "__main__":
    test_ast_extraction()
