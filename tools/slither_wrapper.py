import subprocess
import json
import tempfile
import os
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

    def analyze_contract(self, source_code: str, contract_name: Optional[str] = None) -> Dict:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False) as f:
                f.write(source_code)
                temp_file = f.name

            cmd = ['slither', temp_file, '--json', '-']

            if contract_name:
                cmd.extend(['--contract-name', contract_name])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            os.unlink(temp_file)

            if result.returncode != 0 and not result.stdout:
                logger.error(f"Slither analysis failed: {result.stderr}")
                return {'success': False, 'error': result.stderr}

            analysis = json.loads(result.stdout) if result.stdout else {}

            return self._process_slither_output(analysis)

        except subprocess.TimeoutExpired:
            logger.error(f"Slither analysis timed out after {self.timeout} seconds")
            return {'success': False, 'error': 'Analysis timeout'}
        except Exception as e:
            logger.error(f"Slither analysis error: {str(e)}")
            return {'success': False, 'error': str(e)}
        finally:
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.unlink(temp_file)

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