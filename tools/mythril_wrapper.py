import subprocess
import json
import tempfile
import os
from typing import Dict, List, Optional, Any, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class MythrilWrapper:
    def __init__(self, timeout: int = 300, max_depth: int = 22):
        self.timeout = timeout
        self.max_depth = max_depth
        self.vulnerability_mapping = {
            'Integer Overflow': 'overflow',
            'Integer Underflow': 'underflow',
            'Reentrancy': 'reentrancy',
            'Unprotected Ether Withdrawal': 'access_control',
            'Unprotected SELFDESTRUCT': 'self_destruct',
            'Unchecked Call Return Value': 'unchecked_call',
            'Dependence on predictable environment variable': 'timestamp_dependency',
            'Transaction Order Dependence': 'transaction_ordering',
            'Use of tx.origin': 'tx_origin',
            'Delegatecall to user-supplied address': 'delegatecall'
        }

    def analyze_contract(
        self,
        source_code: str,
        contract_name: Optional[str] = None,
        solc_version: str = "0.8.0"
    ) -> Dict:
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False) as f:
                f.write(source_code)
                temp_file = f.name

            cmd = [
                'myth', 'analyze',
                temp_file,
                '--solv', solc_version,
                '--execution-timeout', str(self.timeout),
                '--max-depth', str(self.max_depth),
                '-o', 'json'
            ]

            if contract_name:
                cmd.extend(['--contract', contract_name])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout + 60
            )

            # Clean up temp file immediately
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
                temp_file = None

            if result.returncode != 0 and not result.stdout:
                # Extract meaningful error from stderr
                stderr_lines = result.stderr.strip().split('\n') if result.stderr else []
                error_msg = next((line for line in stderr_lines if 'Error' in line or 'error' in line),
                               stderr_lines[-1] if stderr_lines else 'Unknown error')
                logger.error(f"Mythril analysis failed: {error_msg[:200]}")
                return {'success': False, 'error': error_msg}

            analysis = json.loads(result.stdout) if result.stdout else {}

            return self._process_mythril_output(analysis, source_code)

        except subprocess.TimeoutExpired:
            logger.error(f"Mythril analysis timed out after {self.timeout} seconds")
            return {'success': False, 'error': 'Analysis timeout'}
        except Exception as e:
            logger.error(f"Mythril analysis error: {str(e)}")
            return {'success': False, 'error': str(e)}
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass

    def _process_mythril_output(self, analysis: Dict, source_code: str) -> Dict:
        vulnerabilities = []
        execution_traces = []

        if 'issues' in analysis:
            for issue in analysis['issues']:
                vuln_type = self.vulnerability_mapping.get(
                    issue.get('title', ''),
                    'unknown'
                )

                vulnerability = {
                    'type': vuln_type,
                    'severity': issue.get('severity', 'unknown'),
                    'description': issue.get('description', ''),
                    'function': issue.get('function', ''),
                    'line_number': issue.get('lineno', -1),
                    'code': issue.get('code', ''),
                    'swc_id': issue.get('swc-id', '')
                }

                if 'tx_sequence' in issue:
                    trace = self._extract_execution_trace(issue['tx_sequence'])
                    vulnerability['execution_trace'] = trace
                    execution_traces.append(trace)

                vulnerabilities.append(vulnerability)

        return {
            'success': True,
            'vulnerabilities': vulnerabilities,
            'execution_traces': execution_traces,
            'summary': {
                'total_issues': len(vulnerabilities),
                'critical': sum(1 for v in vulnerabilities if v['severity'] == 'High'),
                'medium': sum(1 for v in vulnerabilities if v['severity'] == 'Medium'),
                'low': sum(1 for v in vulnerabilities if v['severity'] == 'Low'),
                'unique_vulnerability_types': len(set(v['type'] for v in vulnerabilities))
            }
        }

    def _extract_execution_trace(self, tx_sequence: Dict) -> Dict:
        trace = {
            'steps': [],
            'constraints': [],
            'storage_writes': [],
            'external_calls': []
        }

        if 'steps' in tx_sequence:
            for step in tx_sequence['steps']:
                trace_step = {
                    'opcode': step.get('opcode', ''),
                    'address': step.get('address', ''),
                    'input': step.get('input', ''),
                    'output': step.get('output', ''),
                    'gas': step.get('gas', 0),
                    'depth': step.get('depth', 0)
                }
                trace['steps'].append(trace_step)

                if step.get('opcode') in ['SSTORE', 'SLOAD']:
                    trace['storage_writes'].append({
                        'opcode': step['opcode'],
                        'address': step.get('address', ''),
                        'value': step.get('value', '')
                    })

                if step.get('opcode') in ['CALL', 'DELEGATECALL', 'STATICCALL']:
                    trace['external_calls'].append({
                        'opcode': step['opcode'],
                        'target': step.get('target', ''),
                        'value': step.get('value', 0)
                    })

        if 'constraints' in tx_sequence:
            trace['constraints'] = tx_sequence['constraints']

        return trace


def extract_dynamic_features(
    source_code: str,
    contract_name: Optional[str] = None
) -> Dict[str, Any]:
    wrapper = MythrilWrapper()
    result = wrapper.analyze_contract(source_code, contract_name)

    if not result['success']:
        return None

    features = {
        'vulnerability_count': result['summary']['total_issues'],
        'critical_count': result['summary']['critical'],
        'execution_traces': result['execution_traces'],
        'vulnerabilities': result['vulnerabilities']
    }

    trace_features = []
    for trace in result['execution_traces']:
        trace_feat = {
            'num_steps': len(trace['steps']),
            'num_storage_ops': len(trace['storage_writes']),
            'num_external_calls': len(trace['external_calls']),
            'max_depth': max([s['depth'] for s in trace['steps']], default=0),
            'total_gas': sum([s['gas'] for s in trace['steps']]),
            'unique_opcodes': len(set([s['opcode'] for s in trace['steps']]))
        }
        trace_features.append(trace_feat)

    features['trace_statistics'] = trace_features

    opcode_sequence = []
    for trace in result['execution_traces']:
        opcodes = [step['opcode'] for step in trace['steps']]
        opcode_sequence.extend(opcodes)
    features['opcode_sequence'] = opcode_sequence

    return features


def encode_execution_trace(trace: Dict, max_length: int = 512) -> List[int]:
    opcode_to_id = {
        'STOP': 1, 'ADD': 2, 'MUL': 3, 'SUB': 4, 'DIV': 5, 'MOD': 6,
        'EXP': 7, 'NOT': 8, 'LT': 9, 'GT': 10, 'EQ': 11, 'AND': 12,
        'OR': 13, 'XOR': 14, 'BYTE': 15, 'SHL': 16, 'SHR': 17,
        'PUSH1': 18, 'PUSH2': 19, 'PUSH32': 20, 'DUP1': 21, 'DUP16': 22,
        'SWAP1': 23, 'SWAP16': 24, 'MLOAD': 25, 'MSTORE': 26, 'SLOAD': 27,
        'SSTORE': 28, 'JUMP': 29, 'JUMPI': 30, 'PC': 31, 'MSIZE': 32,
        'GAS': 33, 'JUMPDEST': 34, 'CALL': 35, 'DELEGATECALL': 36,
        'STATICCALL': 37, 'RETURN': 38, 'REVERT': 39, 'SELFDESTRUCT': 40
    }

    encoded = []
    for step in trace.get('steps', [])[:max_length]:
        opcode = step.get('opcode', 'UNKNOWN')
        opcode_id = opcode_to_id.get(opcode, 0)
        encoded.append(opcode_id)

    while len(encoded) < max_length:
        encoded.append(0)

    return encoded[:max_length]