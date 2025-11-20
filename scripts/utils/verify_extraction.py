#!/usr/bin/env python3
"""
Contract Extraction Verification Script
Verifies PDG and AST extraction for smart contracts with detailed reporting
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.config import get_config
from tools.slither_wrapper import SlitherWrapper
from tools.ast_extractor import ASTExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExtractionVerifier:
    """Verify PDG and AST extraction for contracts"""

    def __init__(self):
        self.config = get_config()
        self.slither = SlitherWrapper(
            timeout=self.config.get('processing.slither_timeout', 60)
        )
        self.ast_extractor = ASTExtractor(
            solc_path=self.config.get('processing.solc_path', 'solc'),
            timeout=self.config.get('processing.solc_timeout', 30)
        )

        self.results = {
            'total': 0,
            'pdg_success': 0,
            'pdg_failed': 0,
            'ast_success': 0,
            'ast_failed': 0,
            'both_success': 0,
            'failed_contracts': [],
            'pdg_stats': defaultdict(int),
            'ast_stats': defaultdict(int)
        }

    def flatten_contract(self, contract_path: Path) -> str:
        """
        Flatten a Solidity contract (remove imports, combine files)
        For now, just read the file - real flattening needs truffle-flattener or similar
        """
        try:
            with open(contract_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading {contract_path}: {e}")
            return None

    def verify_pdg_extraction(self, contract_path: Path, source_code: str) -> Tuple[bool, Dict]:
        """Verify PDG extraction for a contract"""
        try:
            result = self.slither.analyze_contract(source_code)

            if result.get('success') and result.get('pdg'):
                pdg = result['pdg']
                num_nodes = pdg.number_of_nodes()
                num_edges = pdg.number_of_edges()

                self.results['pdg_stats']['total_nodes'] += num_nodes
                self.results['pdg_stats']['total_edges'] += num_edges

                return True, {
                    'nodes': num_nodes,
                    'edges': num_edges,
                    'message': f'✓ PDG: {num_nodes} nodes, {num_edges} edges'
                }
            else:
                error_msg = result.get('error', 'Unknown error')
                return False, {
                    'message': f'✗ PDG Failed: {error_msg}'
                }

        except Exception as e:
            return False, {
                'message': f'✗ PDG Exception: {str(e)}'
            }

    def verify_ast_extraction(self, contract_path: Path, source_code: str) -> Tuple[bool, Dict]:
        """Verify AST extraction for a contract"""
        try:
            result = self.ast_extractor.extract_ast(source_code, str(contract_path))

            if result.get('success') and result.get('ast'):
                ast = result['ast']

                # Count AST node types
                num_contracts = len(ast.get('contracts', {}))
                num_functions = 0
                num_state_vars = 0

                for contract_data in ast.get('contracts', {}).values():
                    if isinstance(contract_data, dict):
                        num_functions += len(contract_data.get('functions', []))
                        num_state_vars += len(contract_data.get('state_variables', []))

                self.results['ast_stats']['total_contracts'] += num_contracts
                self.results['ast_stats']['total_functions'] += num_functions
                self.results['ast_stats']['total_state_vars'] += num_state_vars

                return True, {
                    'contracts': num_contracts,
                    'functions': num_functions,
                    'state_vars': num_state_vars,
                    'message': f'✓ AST: {num_contracts} contracts, {num_functions} functions'
                }
            else:
                error_msg = result.get('error', 'Unknown error')
                return False, {
                    'message': f'✗ AST Failed: {error_msg}'
                }

        except Exception as e:
            return False, {
                'message': f'✗ AST Exception: {str(e)}'
            }

    def verify_contract(self, contract_path: Path) -> Dict:
        """Verify both PDG and AST extraction for a single contract"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Verifying: {contract_path.name}")
        logger.info(f"{'='*80}")

        self.results['total'] += 1

        # Read contract
        source_code = self.flatten_contract(contract_path)
        if not source_code:
            return {
                'path': str(contract_path),
                'pdg_success': False,
                'ast_success': False,
                'error': 'Failed to read contract'
            }

        # Verify PDG
        pdg_success, pdg_info = self.verify_pdg_extraction(contract_path, source_code)
        logger.info(f"  PDG: {pdg_info['message']}")

        # Verify AST
        ast_success, ast_info = self.verify_ast_extraction(contract_path, source_code)
        logger.info(f"  AST: {ast_info['message']}")

        # Update counts
        if pdg_success:
            self.results['pdg_success'] += 1
        else:
            self.results['pdg_failed'] += 1

        if ast_success:
            self.results['ast_success'] += 1
        else:
            self.results['ast_failed'] += 1

        if pdg_success and ast_success:
            self.results['both_success'] += 1
            logger.info(f"  ✅ BOTH SUCCESSFUL")
        else:
            self.results['failed_contracts'].append(str(contract_path))
            logger.warning(f"  ⚠️  PARTIAL/FAILED")

        return {
            'path': str(contract_path),
            'pdg_success': pdg_success,
            'ast_success': ast_success,
            'pdg_info': pdg_info,
            'ast_info': ast_info
        }

    def verify_directory(self, directory: Path, max_contracts: int = None) -> List[Dict]:
        """Verify all contracts in a directory"""
        logger.info(f"\n{'='*80}")
        logger.info(f"SCANNING DIRECTORY: {directory}")
        logger.info(f"{'='*80}\n")

        # Find all Solidity files
        sol_files = list(directory.rglob("*.sol"))
        if max_contracts:
            sol_files = sol_files[:max_contracts]

        logger.info(f"Found {len(sol_files)} Solidity contracts")
        if max_contracts:
            logger.info(f"Limiting to first {max_contracts} contracts\n")

        results = []
        for i, contract_path in enumerate(sol_files, 1):
            logger.info(f"\n[{i}/{len(sol_files)}]")
            result = self.verify_contract(contract_path)
            results.append(result)

        return results

    def print_summary(self):
        """Print verification summary"""
        logger.info(f"\n{'='*80}")
        logger.info("VERIFICATION SUMMARY")
        logger.info(f"{'='*80}\n")

        logger.info(f"Total Contracts: {self.results['total']}")
        logger.info(f"\nPDG Extraction:")
        logger.info(f"  ✓ Successful: {self.results['pdg_success']} ({self.results['pdg_success']/self.results['total']*100:.1f}%)")
        logger.info(f"  ✗ Failed:     {self.results['pdg_failed']} ({self.results['pdg_failed']/self.results['total']*100:.1f}%)")

        logger.info(f"\nAST Extraction:")
        logger.info(f"  ✓ Successful: {self.results['ast_success']} ({self.results['ast_success']/self.results['total']*100:.1f}%)")
        logger.info(f"  ✗ Failed:     {self.results['ast_failed']} ({self.results['ast_failed']/self.results['total']*100:.1f}%)")

        logger.info(f"\nBoth Successful: {self.results['both_success']} ({self.results['both_success']/self.results['total']*100:.1f}%)")

        if self.results['pdg_success'] > 0:
            avg_nodes = self.results['pdg_stats']['total_nodes'] / self.results['pdg_success']
            avg_edges = self.results['pdg_stats']['total_edges'] / self.results['pdg_success']
            logger.info(f"\nPDG Statistics (avg per successful contract):")
            logger.info(f"  Nodes: {avg_nodes:.1f}")
            logger.info(f"  Edges: {avg_edges:.1f}")

        if self.results['ast_success'] > 0:
            avg_contracts = self.results['ast_stats']['total_contracts'] / self.results['ast_success']
            avg_functions = self.results['ast_stats']['total_functions'] / self.results['ast_success']
            avg_state_vars = self.results['ast_stats']['total_state_vars'] / self.results['ast_success']
            logger.info(f"\nAST Statistics (avg per successful contract):")
            logger.info(f"  Contracts:     {avg_contracts:.1f}")
            logger.info(f"  Functions:     {avg_functions:.1f}")
            logger.info(f"  State Vars:    {avg_state_vars:.1f}")

        if self.results['failed_contracts']:
            logger.info(f"\nFailed/Partial Contracts ({len(self.results['failed_contracts'])}):")
            for contract in self.results['failed_contracts'][:10]:  # Show first 10
                logger.info(f"  - {Path(contract).name}")
            if len(self.results['failed_contracts']) > 10:
                logger.info(f"  ... and {len(self.results['failed_contracts']) - 10} more")

        logger.info(f"\n{'='*80}\n")

    def save_report(self, output_path: Path):
        """Save detailed report to file"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_contracts': self.results['total'],
                'pdg_success': self.results['pdg_success'],
                'pdg_failed': self.results['pdg_failed'],
                'ast_success': self.results['ast_success'],
                'ast_failed': self.results['ast_failed'],
                'both_success': self.results['both_success']
            },
            'pdg_stats': dict(self.results['pdg_stats']),
            'ast_stats': dict(self.results['ast_stats']),
            'failed_contracts': self.results['failed_contracts']
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"✓ Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify PDG and AST extraction for Solidity contracts"
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory containing Solidity contracts"
    )
    parser.add_argument(
        "--max-contracts",
        type=int,
        default=100,
        help="Maximum number of contracts to verify (default: 100)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for JSON report (optional)"
    )

    args = parser.parse_args()

    # Verify directory exists
    contract_dir = Path(args.dir)
    if not contract_dir.exists():
        logger.error(f"Directory not found: {contract_dir}")
        return 1

    # Run verification
    verifier = ExtractionVerifier()
    verifier.verify_directory(contract_dir, max_contracts=args.max_contracts)

    # Print summary
    verifier.print_summary()

    # Save report
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"verification_report_{timestamp}.json")

    verifier.save_report(output_path)

    # Return exit code based on success rate
    success_rate = verifier.results['both_success'] / verifier.results['total'] * 100
    if success_rate >= 80:
        logger.info(f"✅ Verification PASSED ({success_rate:.1f}% success rate)")
        return 0
    else:
        logger.warning(f"⚠️  Verification NEEDS ATTENTION ({success_rate:.1f}% success rate)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
