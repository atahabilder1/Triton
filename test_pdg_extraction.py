#!/usr/bin/env python3
"""
Test PDG extraction with improved Slither wrapper
"""
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from tools.slither_wrapper import SlitherWrapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_contracts():
    """Test PDG extraction on sample contracts"""

    test_files = [
        "data/datasets/forge_reconstructed/train/denial_of_service/QuillAudit-ritestream_Smart_Contract_Audit_Report_RitestreamNFT.sol",
        "data/datasets/forge_reconstructed/train/arithmetic/14-bnb_BNBPark.sol",
        "data/datasets/forge_reconstructed/train/unchecked_low_level_calls/harmony-busd_BUSDImplementation.sol",
        "data/datasets/forge_reconstructed/train/access_control/StrongHands 3D Smart Contract Audit Report - QuillAudits_Migrations.sol",
        "data/datasets/forge_reconstructed/train/other/SmartContract_Audit_Solidproof_PornCoin_MainToken.sol",
        "data/datasets/forge_reconstructed/train/access_control/HALO-Network-Security-Audit-Report_oracle.sol",
        "data/datasets/forge_reconstructed/train/other/Viking Finance_VIKINGToken.sol",
        "data/datasets/forge_reconstructed/train/arithmetic/PeckShield-Audit-Report-InvtAI-v1_FaucetToken.sol",
        "data/datasets/forge_reconstructed/train/other/protocol v2_ATokensAndRatesHelper.sol",
        "data/datasets/forge_reconstructed/train/unchecked_low_level_calls/BlockRewards_0x7785035610075Ec7BcD7c833B03996E866FE0072_BlockRewards.sol"
    ]

    wrapper = SlitherWrapper(log_failures=True, failure_log_path="logs/pdg_test_failures.log")

    success_count = 0
    failure_count = 0

    logger.info("="*80)
    logger.info("TESTING PDG EXTRACTION WITH IMPROVED VERSION MATCHING")
    logger.info("="*80)

    for contract_file in test_files:
        logger.info(f"\nTesting: {Path(contract_file).name}")

        try:
            with open(contract_file, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()

            result = wrapper.analyze_contract(source_code, contract_path=contract_file)

            if result.get('success') and result.get('pdg'):
                pdg = result['pdg']
                num_nodes = pdg.number_of_nodes()
                num_edges = pdg.number_of_edges()

                if num_nodes > 0:
                    logger.info(f"  ✅ SUCCESS - PDG extracted: {num_nodes} nodes, {num_edges} edges")
                    success_count += 1
                else:
                    logger.warning(f"  ⚠️  Empty PDG extracted")
                    failure_count += 1
            else:
                logger.error(f"  ❌ FAILED - {result.get('error', 'Unknown error')}")
                failure_count += 1

        except FileNotFoundError:
            logger.warning(f"  ⚠️  File not found: {contract_file}")
            continue
        except Exception as e:
            logger.error(f"  ❌ ERROR: {e}")
            failure_count += 1

    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"✅ Successful extractions: {success_count}")
    logger.info(f"❌ Failed extractions: {failure_count}")
    if success_count + failure_count > 0:
        logger.info(f"📊 Success rate: {success_count/(success_count+failure_count)*100:.1f}%")
    logger.info("="*80)

    if failure_count > 0:
        logger.info(f"\n📝 Failure log saved to: logs/pdg_test_failures.log")

if __name__ == "__main__":
    test_contracts()
