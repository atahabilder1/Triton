#!/usr/bin/env python3
"""
Test PDG Extraction on Filtered Implementation Contracts

Tests Slither's ability to extract PDGs from implementation contracts.
Saves results for analysis.
"""
import sys
from pathlib import Path
import json
import logging

sys.path.insert(0, '/home/anik/code/Triton')
from tools.slither_wrapper import SlitherWrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pdg_extraction():
    """Test PDG extraction on all filtered implementations"""
    wrapper = SlitherWrapper(log_failures=True, failure_log_path="logs/pdg_test_implementations.log")

    impl_dir = Path('data/datasets/forge_flattened_implementations')
    test_files = sorted(impl_dir.glob('*.sol'))

    logger.info("=" * 80)
    logger.info("PDG EXTRACTION TEST ON FILTERED IMPLEMENTATIONS")
    logger.info("=" * 80)
    logger.info(f"Testing {len(test_files)} implementation contracts")
    logger.info("")

    results = {
        'total': len(test_files),
        'success': 0,
        'failed': 0,
        'successes': [],
        'failures': []
    }

    for i, f in enumerate(test_files):
        try:
            with open(f) as file:
                code = file.read()

            result = wrapper.analyze_contract(code, None, str(f))

            # FIX: Check for PDG object, not num_nodes in result
            if result and result.get('success') and result.get('pdg'):
                pdg = result['pdg']
                num_nodes = pdg.number_of_nodes()
                num_edges = pdg.number_of_edges()

                if num_nodes > 0:
                    results['success'] += 1
                    results['successes'].append({
                        'file': f.name,
                        'nodes': num_nodes,
                        'edges': num_edges
                    })
                    logger.info(f'✅ [{i+1}/{len(test_files)}] {f.name[:60]:60} | Nodes: {num_nodes:4} | Edges: {num_edges:4}')
                else:
                    results['failed'] += 1
                    results['failures'].append(f.name)
                    logger.info(f'❌ [{i+1}/{len(test_files)}] {f.name[:60]:60} | PDG empty (0 nodes)')
            else:
                results['failed'] += 1
                results['failures'].append(f.name)
                logger.info(f'❌ [{i+1}/{len(test_files)}] {f.name[:60]:60} | No PDG extracted')

        except Exception as e:
            results['failed'] += 1
            results['failures'].append(f.name)
            logger.error(f'❌ [{i+1}/{len(test_files)}] {f.name[:60]:60} | Error: {str(e)[:40]}')

        # Progress report every 50 files
        if (i + 1) % 50 == 0:
            success_rate = 100 * results['success'] / (i + 1)
            logger.info(f"\n  Progress: {i+1}/{len(test_files)}")
            logger.info(f"  Success: {results['success']} ({success_rate:.1f}%)")
            logger.info(f"  Failed: {results['failed']}\n")

    # Final report
    logger.info("\n" + "=" * 80)
    logger.info("PDG EXTRACTION TEST COMPLETE")
    logger.info("=" * 80)

    success_rate = 100 * results['success'] / results['total'] if results['total'] > 0 else 0

    logger.info(f"\nTotal tested: {results['total']}")
    logger.info(f"✅ Successful: {results['success']} ({success_rate:.1f}%)")
    logger.info(f"❌ Failed: {results['failed']} ({100-success_rate:.1f}%)")

    # Save results
    results_file = Path('results/pdg_extraction_test_implementations.json')
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved: {results_file}")
    logger.info(f"Failure log: logs/pdg_test_implementations.log")

    return results

if __name__ == "__main__":
    results = test_pdg_extraction()

    # Exit with status code based on success rate
    success_rate = 100 * results['success'] / results['total']
    if success_rate >= 90:
        sys.exit(0)  # Success
    elif success_rate >= 50:
        sys.exit(1)  # Partial success
    else:
        sys.exit(2)  # Failure
