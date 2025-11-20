#!/usr/bin/env python3
"""
Re-flatten contracts with missing dependencies using Foundry
"""
import os
import sys
import subprocess
import re
from pathlib import Path
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Foundry binary
FORGE = os.path.expanduser('~/.foundry/bin/forge')

def detect_missing_imports(contract_path: str) -> List[str]:
    """Detect missing imports from a contract file."""
    missing = []

    with open(contract_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Check for inheritance without imports
    inheritance_pattern = r'contract\s+\w+\s+is\s+([\w\s,]+)\s*\{'
    matches = re.findall(inheritance_pattern, content)

    for match in matches:
        parents = [p.strip() for p in match.split(',')]
        for parent in parents:
            # Check if this parent is defined in the file
            if parent and not re.search(rf'(contract|interface|library)\s+{parent}\s+', content):
                # Check if there's an import for it
                if not re.search(rf'import.*{parent}', content):
                    missing.append(parent)

    return list(set(missing))

def needs_reflattening(contract_path: str) -> bool:
    """Check if contract needs re-flattening."""
    missing = detect_missing_imports(contract_path)
    return len(missing) > 0

def create_foundry_project(contract_path: str, work_dir: Path) -> bool:
    """Create a minimal Foundry project for flattening."""
    try:
        # Create foundry project structure
        src_dir = work_dir / 'src'
        src_dir.mkdir(parents=True, exist_ok=True)

        # Copy contract to src
        contract_name = Path(contract_path).name
        dest = src_dir / contract_name

        with open(contract_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        with open(dest, 'w', encoding='utf-8') as f:
            f.write(content)

        # Create foundry.toml with remappings for OpenZeppelin
        foundry_toml = work_dir / 'foundry.toml'
        with open(foundry_toml, 'w') as f:
            f.write('''[profile.default]
src = 'src'
out = 'out'
libs = ['lib']
remappings = [
    '@openzeppelin/=lib/openzeppelin-contracts/',
    'openzeppelin-contracts/=lib/openzeppelin-contracts/'
]

[dependencies]
openzeppelin-contracts = "4.9.0"
''')

        # Install OpenZeppelin
        logger.debug(f"Installing OpenZeppelin contracts...")
        result = subprocess.run(
            [FORGE, 'install', 'OpenZeppelin/openzeppelin-contracts@v4.9.0', '--no-commit'],
            cwd=work_dir,
            capture_output=True,
            timeout=60
        )

        if result.returncode != 0:
            logger.debug(f"OpenZeppelin install warning (may be OK): {result.stderr.decode()[:200]}")

        return True

    except Exception as e:
        logger.error(f"Error creating Foundry project: {e}")
        return False

def flatten_contract(contract_path: str, output_path: str) -> bool:
    """Flatten a contract using Foundry."""
    try:
        work_dir = Path('/tmp/foundry_flatten')
        work_dir.mkdir(parents=True, exist_ok=True)

        # Create Foundry project
        if not create_foundry_project(contract_path, work_dir):
            return False

        contract_name = Path(contract_path).name
        src_file = work_dir / 'src' / contract_name

        # Run forge flatten
        logger.info(f"  Flattening {contract_name}...")
        result = subprocess.run(
            [FORGE, 'flatten', str(src_file)],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0 and result.stdout:
            # Save flattened output
            with open(output_path, 'w', encoding='utf-8') as f:
                # Clean up the output (remove duplicate SPDX licenses)
                lines = result.stdout.split('\n')
                seen_spdx = False
                cleaned_lines = []
                for line in lines:
                    if 'SPDX-License-Identifier' in line:
                        if not seen_spdx:
                            cleaned_lines.append(line)
                            seen_spdx = True
                    else:
                        cleaned_lines.append(line)

                f.write('\n'.join(cleaned_lines))

            logger.info(f"  ✅ Successfully flattened to {output_path}")
            return True
        else:
            logger.debug(f"  Flatten failed: {result.stderr[:200]}")
            return False

    except Exception as e:
        logger.error(f"  ❌ Error flattening {contract_path}: {e}")
        return False
    finally:
        # Cleanup
        if work_dir.exists():
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)

def main():
    dataset_dir = Path('data/datasets/forge_reconstructed')
    output_dir = Path('data/datasets/forge_reconstructed_flattened')

    logger.info("="*80)
    logger.info("CONTRACT RE-FLATTENING WITH FOUNDRY")
    logger.info("="*80)
    logger.info(f"Source: {dataset_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Check if Foundry is installed
    if not Path(FORGE).exists():
        logger.error("❌ Foundry not found. Please run: curl -L https://foundry.paradigm.xyz | bash && foundryup")
        sys.exit(1)

    logger.info(f"✅ Foundry found: {FORGE}")
    logger.info("")

    # Find all contracts
    all_contracts = list(dataset_dir.rglob('*.sol'))
    logger.info(f"📁 Found {len(all_contracts)} total contracts")

    # Identify contracts needing re-flattening
    contracts_to_flatten = []
    for contract in all_contracts:
        if needs_reflattening(str(contract)):
            contracts_to_flatten.append(contract)

    logger.info(f"🔧 {len(contracts_to_flatten)} contracts need re-flattening")
    logger.info("")

    if len(contracts_to_flatten) == 0:
        logger.info("✅ No contracts need re-flattening!")
        return

    # Re-flatten each contract
    success_count = 0
    fail_count = 0

    logger.info("Starting re-flattening...")
    logger.info("="*80)

    for i, contract in enumerate(contracts_to_flatten[:20], 1):  # Limit to 20 for now
        logger.info(f"\n[{i}/{min(20, len(contracts_to_flatten))}] {contract.name}")

        # Create output path maintaining structure
        relative_path = contract.relative_to(dataset_dir)
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if flatten_contract(str(contract), str(output_path)):
            success_count += 1
        else:
            fail_count += 1
            # Copy original if flattening fails
            import shutil
            shutil.copy2(contract, output_path)
            logger.info(f"  ⚠️  Copied original (flattening failed)")

    logger.info("")
    logger.info("="*80)
    logger.info("RE-FLATTENING COMPLETE")
    logger.info("="*80)
    logger.info(f"✅ Successfully re-flattened: {success_count}")
    logger.info(f"❌ Failed to re-flatten: {fail_count}")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info("="*80)

if __name__ == '__main__':
    main()
