import subprocess
import json
import tempfile
import os
import re
from typing import Dict, List, Optional, Any
import networkx as nx
import logging

logger = logging.getLogger(__name__)


class SlitherWrapper:
    def __init__(self, timeout: int = 300):
        self.timeout = timeout
        self.detector_mapping = {
            'reentrancy-eth': 'reentrancy',
            'reentrancy-no-eth': 'reentrancy',
            'reentrancy-benign': 'reentrancy',
            'integer-overflow': 'overflow',
            'integer-underflow': 'underflow',
            'unprotected-upgrade': 'access_control',
            'suicidal': 'self_destruct',
            'unchecked-lowlevel': 'unchecked_call',
            'unchecked-send': 'unchecked_call',
            'timestamp': 'timestamp_dependency',
            'tx-origin': 'tx_origin',
            'delegatecall-loop': 'delegatecall',
            'arbitrary-send': 'access_control'
        }

    def _use_python_api(self, source_code: str) -> Optional[Dict]:
        """Use Slither's Python API to get CFG/PDG data."""
        try:
            from slither import Slither

            # Write source to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False) as f:
                f.write(source_code)
                temp_file = f.name

            # Use Slither Python API
            slither = Slither(temp_file)

            # Clean up immediately
            os.unlink(temp_file)

            return slither

        except Exception as e:
            logger.debug(f"Python API failed: {e}")
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.unlink(temp_file)
            return None

    def _detect_solc_version(self, source_code: str) -> Optional[str]:
        """Detect required Solidity version from pragma statement."""
        pragma_pattern = r'pragma\s+solidity\s+([^;]+);'
        match = re.search(pragma_pattern, source_code)

        if not match:
            return None

        version_spec = match.group(1).strip()

        # Extract version number (handles ^0.4.0, >=0.5.0, 0.6.12, etc.)
        version_match = re.search(r'(\d+\.\d+)\.?\d*', version_spec)
        if not version_match:
            return None

        major_minor = version_match.group(1)

        # Map to available versions
        version_map = {
            '0.4': '0.4.26',
            '0.5': '0.5.17',
            '0.6': '0.6.12',
            '0.7': '0.7.6',
            '0.8': '0.8.30'  # Keep 0.8 support
        }

        return version_map.get(major_minor, '0.5.17')  # Default to 0.5.17

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
            logger.warning(f"Failed to set solc version {version}: {e}")
            return False

    def analyze_contract(self, source_code: str, contract_name: Optional[str] = None) -> Dict:
        temp_file = None
        try:
            # Detect and set appropriate Solidity compiler version
            required_version = self._detect_solc_version(source_code)
            if required_version:
                if self._set_solc_version(required_version):
                    logger.debug(f"Set solc version to {required_version}")

            # Try Python API first for better PDG extraction
            slither = self._use_python_api(source_code)

            if slither:
                # Extract data using Python API
                return self._extract_from_python_api(slither, contract_name)
            else:
                # Fallback to CLI (won't have PDG but still get vulnerabilities)
                logger.warning("Python API failed, falling back to CLI")
                return self._analyze_with_cli(source_code, contract_name)

        except Exception as e:
            logger.error(f"Slither analysis error: {str(e)}")
            return {'success': False, 'error': str(e)}
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass

    def _extract_from_python_api(self, slither, contract_name: Optional[str] = None) -> Dict:
        """Extract PDG and vulnerabilities from Slither Python API."""
        try:
            pdg = nx.DiGraph()
            vulnerabilities = []

            # Get contracts
            contracts = slither.contracts

            if contract_name:
                contracts = [c for c in contracts if c.name == contract_name]

            # Build PDG from actual contract structure
            for contract in contracts:
                contract_name_str = contract.name

                # Add contract-level state variables
                for state_var in contract.state_variables:
                    var_name = f"{contract_name_str}.{state_var.name}"
                    pdg.add_node(var_name, type='state_variable', visibility=str(state_var.visibility))

                # Add functions and build control/data flow
                for function in contract.functions:
                    func_name = f"{contract_name_str}.{function.name}"
                    pdg.add_node(func_name,
                                type='function',
                                visibility=str(function.visibility),
                                is_constructor=function.is_constructor,
                                is_fallback=function.is_fallback)

                    # Add modifiers
                    for modifier in function.modifiers:
                        mod_name = f"{contract_name_str}.{modifier.name}"
                        pdg.add_node(mod_name, type='modifier')
                        pdg.add_edge(func_name, mod_name, type='uses_modifier')

                    # Add variables read/written
                    for var in function.state_variables_read:
                        var_name = f"{contract_name_str}.{var.name}"
                        if not pdg.has_node(var_name):
                            pdg.add_node(var_name, type='state_variable')
                        pdg.add_edge(func_name, var_name, type='reads')

                    for var in function.state_variables_written:
                        var_name = f"{contract_name_str}.{var.name}"
                        if not pdg.has_node(var_name):
                            pdg.add_node(var_name, type='state_variable')
                        pdg.add_edge(func_name, var_name, type='writes')

                    # Add internal calls (function calls within contract)
                    for call in function.internal_calls:
                        if hasattr(call, 'name'):
                            call_name = f"{contract_name_str}.{call.name}"
                            pdg.add_edge(func_name, call_name, type='calls')

                    # Add external calls
                    for call in function.external_calls_as_expressions:
                        # Just note that external call exists
                        pdg.add_node(f"{func_name}:external_call", type='external_call')
                        pdg.add_edge(func_name, f"{func_name}:external_call", type='makes_external_call')

            # Extract vulnerabilities from detectors (run lightweight detectors)
            # Note: We don't run all detectors as it's slow, but we get basic info
            for contract in contracts:
                for function in contract.functions:
                    # Check for some basic patterns
                    if function.can_reenter():
                        vulnerabilities.append({
                            'type': 'reentrancy',
                            'severity': 'High',
                            'confidence': 'Medium',
                            'description': f'Function {function.name} may be vulnerable to reentrancy',
                            'elements': [{'name': function.name}]
                        })

            logger.info(f"Extracted PDG with {pdg.number_of_nodes()} nodes, {pdg.number_of_edges()} edges")

            return {
                'success': True,
                'vulnerabilities': vulnerabilities,
                'pdg': pdg,
                'summary': {
                    'total_issues': len(vulnerabilities),
                    'high_severity': sum(1 for v in vulnerabilities if v['severity'] == 'High'),
                    'medium_severity': sum(1 for v in vulnerabilities if v['severity'] == 'Medium'),
                    'low_severity': sum(1 for v in vulnerabilities if v['severity'] == 'Low')
                }
            }

        except Exception as e:
            logger.error(f"Error extracting from Python API: {e}")
            return {'success': False, 'error': str(e)}

    def _analyze_with_cli(self, source_code: str, contract_name: Optional[str] = None) -> Dict:
        """Fallback: Use CLI approach (no PDG but gets vulnerabilities)."""
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False) as f:
                f.write(source_code)
                temp_file = f.name

            cmd = ['slither', temp_file, '--json', '-']
            if contract_name:
                cmd.extend(['--contract-name', contract_name])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)

            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
                temp_file = None

            if result.returncode != 0 and not result.stdout:
                stderr_lines = result.stderr.strip().split('\n') if result.stderr else []
                error_msg = next((line for line in stderr_lines if 'Error' in line or 'error' in line),
                               stderr_lines[-1] if stderr_lines else 'Unknown error')
                logger.error(f"Slither CLI analysis failed: {error_msg[:200]}")
                return {'success': False, 'error': error_msg}

            if result.returncode != 0:
                logger.warning(f"Slither returned error code but has output")

            analysis = json.loads(result.stdout) if result.stdout else {}
            return self._process_slither_output(analysis)

        except subprocess.TimeoutExpired:
            logger.error(f"Slither CLI timed out after {self.timeout} seconds")
            return {'success': False, 'error': 'Analysis timeout'}
        except Exception as e:
            logger.error(f"Slither CLI error: {str(e)}")
            return {'success': False, 'error': str(e)}
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass

    def _process_slither_output(self, analysis: Dict) -> Dict:
        vulnerabilities = []
        pdg = nx.DiGraph()

        if 'results' in analysis:
            detectors = analysis['results'].get('detectors', [])

            for detector in detectors:
                vuln_type = self.detector_mapping.get(
                    detector.get('check', ''),
                    detector.get('check', 'unknown')
                )

                vulnerability = {
                    'type': vuln_type,
                    'severity': detector.get('impact', 'unknown'),
                    'confidence': detector.get('confidence', 'unknown'),
                    'description': detector.get('description', ''),
                    'elements': []
                }

                for element in detector.get('elements', []):
                    if element.get('type') == 'function':
                        vulnerability['elements'].append({
                            'name': element.get('name', ''),
                            'source_mapping': element.get('source_mapping', {})
                        })

                vulnerabilities.append(vulnerability)

        if 'contracts' in analysis:
            pdg = self._build_program_dependence_graph(analysis['contracts'])

        return {
            'success': True,
            'vulnerabilities': vulnerabilities,
            'pdg': pdg,
            'summary': {
                'total_issues': len(vulnerabilities),
                'high_severity': sum(1 for v in vulnerabilities if v['severity'] == 'High'),
                'medium_severity': sum(1 for v in vulnerabilities if v['severity'] == 'Medium'),
                'low_severity': sum(1 for v in vulnerabilities if v['severity'] == 'Low')
            }
        }

    def _build_program_dependence_graph(self, contracts: List[Dict]) -> nx.DiGraph:
        pdg = nx.DiGraph()

        for contract in contracts:
            contract_name = contract.get('name', '')

            for function in contract.get('functions', []):
                func_name = f"{contract_name}.{function.get('name', '')}"
                pdg.add_node(func_name, type='function')

                for modifier in function.get('modifiers', []):
                    mod_name = f"{contract_name}.{modifier}"
                    pdg.add_node(mod_name, type='modifier')
                    pdg.add_edge(func_name, mod_name, type='uses_modifier')

                for var in function.get('variables_read', []):
                    var_name = f"{contract_name}.{var}"
                    pdg.add_node(var_name, type='variable')
                    pdg.add_edge(func_name, var_name, type='reads')

                for var in function.get('variables_written', []):
                    var_name = f"{contract_name}.{var}"
                    pdg.add_node(var_name, type='variable')
                    pdg.add_edge(func_name, var_name, type='writes')

                for call in function.get('internal_calls', []):
                    call_name = f"{contract_name}.{call}"
                    pdg.add_edge(func_name, call_name, type='calls')

        return pdg


def extract_static_features(source_code: str, contract_name: Optional[str] = None) -> Dict[str, Any]:
    wrapper = SlitherWrapper()
    result = wrapper.analyze_contract(source_code, contract_name)

    if not result['success']:
        return None

    features = {
        'vulnerability_count': result['summary']['total_issues'],
        'high_severity_count': result['summary']['high_severity'],
        'medium_severity_count': result['summary']['medium_severity'],
        'low_severity_count': result['summary']['low_severity'],
        'pdg': result['pdg'],
        'vulnerabilities': result['vulnerabilities']
    }

    pdg = result['pdg']
    features['pdg_stats'] = {
        'num_nodes': pdg.number_of_nodes(),
        'num_edges': pdg.number_of_edges(),
        'avg_degree': sum(dict(pdg.degree()).values()) / pdg.number_of_nodes() if pdg.number_of_nodes() > 0 else 0,
        'num_functions': sum(1 for n, d in pdg.nodes(data=True) if d.get('type') == 'function'),
        'num_variables': sum(1 for n, d in pdg.nodes(data=True) if d.get('type') == 'variable')
    }

    return features