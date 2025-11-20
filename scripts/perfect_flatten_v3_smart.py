#!/usr/bin/env python3
"""
Perfect Flattening V3 - SMART OPTIMIZATIONS

Key improvements based on user feedback:
1. Only flatten 1,148 contracts with vulnerability labels (not all 6,616!)
2. Try flattening FIRST (fast), install dependencies ONLY if it fails (slow)
3. Install OpenZeppelin GLOBALLY once, share via remappings
4. Skip pure library files (Context.sol, SafeMath.sol, etc.)

Expected results:
- Process only necessary contracts
- 10-20x faster than V2
- 90%+ success rate
- Minimal disk usage (shared libraries)
"""
import os
import sys
import json
import subprocess
import shutil
import re
from pathlib import Path
from collections import Counter
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SmartFlattener:
    def __init__(self, forge_reconstructed_dir: str, output_dir: str):
        self.forge_reconstructed = Path(forge_reconstructed_dir)
        self.output_dir = Path(output_dir)
        self.stats = Counter()

        # Global library directory (shared across all projects)
        self.global_lib_dir = Path('/tmp/forge_global_libs')
        self.global_remappings = self.global_lib_dir / 'remappings.txt'

    def setup_global_libraries(self):
        """
        Install OpenZeppelin ONCE globally, create shared remappings
        """
        logger.info("=" * 80)
        logger.info("SETTING UP GLOBAL LIBRARIES (ONE-TIME)")
        logger.info("=" * 80)

        self.global_lib_dir.mkdir(parents=True, exist_ok=True)

        # Create package.json
        package_json = self.global_lib_dir / 'package.json'
        with open(package_json, 'w') as f:
            json.dump({
                "name": "forge-global-libs",
                "version": "1.0.0",
                "dependencies": {
                    "@openzeppelin/contracts": "^4.9.0",
                    "@openzeppelin/contracts-upgradeable": "^4.9.0",
                    "openzeppelin-solidity": "^2.5.1"
                }
            }, f)

        # Install npm packages
        logger.info("Installing OpenZeppelin globally...")
        result = subprocess.run(
            ['npm', 'install', '--silent'],
            cwd=self.global_lib_dir,
            capture_output=True,
            timeout=300
        )

        if result.returncode == 0:
            logger.info("✅ Global libraries installed successfully")
        else:
            logger.warning("⚠️ Global npm install had issues, continuing anyway...")

        # Also try forge install
        logger.info("Installing via Foundry...")
        subprocess.run(
            ['forge', 'init', '--no-commit', '--force'],
            cwd=self.global_lib_dir,
            capture_output=True
        )
        subprocess.run(
            ['forge', 'install', '--no-commit',
             'OpenZeppelin/openzeppelin-contracts@v4.9.0',
             'OpenZeppelin/openzeppelin-contracts-upgradeable@v4.9.0'],
            cwd=self.global_lib_dir,
            capture_output=True,
            timeout=300
        )

        # Create global remappings
        remappings = [
            f'@openzeppelin/={self.global_lib_dir}/node_modules/@openzeppelin/',
            f'@openzeppelin/contracts/={self.global_lib_dir}/node_modules/@openzeppelin/contracts/',
            f'@openzeppelin/contracts-upgradeable/={self.global_lib_dir}/node_modules/@openzeppelin/contracts-upgradeable/',
            f'openzeppelin-solidity/={self.global_lib_dir}/node_modules/openzeppelin-solidity/',
            f'OpenZeppelin/={self.global_lib_dir}/lib/openzeppelin-contracts/',
        ]

        with open(self.global_remappings, 'w') as f:
            f.write('\n'.join(remappings))

        logger.info(f"✅ Global remappings created: {self.global_remappings}")
        logger.info("")

    def is_library_file(self, file_path: Path) -> bool:
        """
        Detect if file is a library (Context.sol, SafeMath.sol, etc.)
        Skip these to avoid flattening unnecessary files
        """
        common_libraries = [
            'Context.sol', 'SafeMath.sol', 'IERC20.sol', 'ERC20.sol',
            'Ownable.sol', 'SafeERC20.sol', 'Address.sol', 'Strings.sol',
            'ECDSA.sol', 'ReentrancyGuard.sol', 'Pausable.sol'
        ]
        return file_path.name in common_libraries

    def detect_pragma(self, source_code: str) -> str:
        """Detect appropriate pragma from contract"""
        pragma_match = re.search(r'pragma\s+solidity\s+([^;]+);', source_code)
        if pragma_match:
            return pragma_match.group(0)

        # Syntax-based detection
        if 'unchecked' in source_code:
            return 'pragma solidity ^0.8.0;'
        elif 'constructor(' in source_code:
            return 'pragma solidity ^0.5.0;'
        else:
            return 'pragma solidity ^0.8.0;'

    def add_pragma_if_missing(self, source_code: str) -> str:
        """Add pragma statement if missing"""
        if 'pragma solidity' in source_code:
            return source_code

        pragma = self.detect_pragma(source_code)

        if 'SPDX-License-Identifier' in source_code:
            lines = source_code.split('\n')
            for i, line in enumerate(lines):
                if 'SPDX-License-Identifier' in line:
                    lines.insert(i + 1, '')
                    lines.insert(i + 2, pragma)
                    break
            return '\n'.join(lines)
        else:
            return f"{pragma}\n\n{source_code}"

    def create_project_remappings(self, project_dir: Path):
        """
        Create remappings.txt pointing to global libraries
        """
        remappings_file = project_dir / 'remappings.txt'

        # Copy global remappings
        if self.global_remappings.exists():
            shutil.copy(self.global_remappings, remappings_file)

    def flatten_with_foundry(self, contract_file: Path, project_dir: Path) -> str:
        """Flatten using Foundry (fastest, best tool)"""
        try:
            # Create remappings pointing to global libs
            self.create_project_remappings(project_dir)

            result = subprocess.run(
                ['forge', 'flatten', str(contract_file)],
                cwd=project_dir,
                capture_output=True,
                timeout=30,
                env={**os.environ, 'FOUNDRY_REMAPPINGS': str(self.global_remappings)}
            )

            if result.returncode == 0 and result.stdout:
                return result.stdout.decode('utf-8', errors='ignore')
        except:
            pass
        return None

    def install_project_dependencies(self, project_dir: Path) -> bool:
        """
        Install project-specific dependencies (ONLY called if flattening fails)
        """
        try:
            logger.info(f"      Installing project-specific dependencies...")

            # Create package.json if missing
            package_json = project_dir / 'package.json'
            if not package_json.exists():
                with open(package_json, 'w') as f:
                    json.dump({
                        "name": "flatten-project",
                        "version": "1.0.0",
                        "dependencies": {
                            "@openzeppelin/contracts": "^4.9.0",
                            "@openzeppelin/contracts-upgradeable": "^4.9.0"
                        }
                    }, f)

            # npm install
            result = subprocess.run(
                ['npm', 'install', '--silent'],
                cwd=project_dir,
                capture_output=True,
                timeout=180
            )

            return result.returncode == 0
        except:
            return False

    def flatten_contract_smart(self, contract_file: Path, project_dir: Path) -> str:
        """
        SMART FLATTENING:
        1. Try flattening first (fast)
        2. If fails, install dependencies, retry
        """
        # STEP 1: Try flattening with global libs (FAST)
        flattened = self.flatten_with_foundry(contract_file, project_dir)

        if flattened:
            flattened = self.add_pragma_if_missing(flattened)
            self.stats['success_first_try'] += 1
            return flattened

        # STEP 2: Install project dependencies, retry (SLOW)
        logger.info(f"      First attempt failed, installing dependencies...")
        if self.install_project_dependencies(project_dir):
            flattened = self.flatten_with_foundry(contract_file, project_dir)

            if flattened:
                flattened = self.add_pragma_if_missing(flattened)
                self.stats['success_after_deps'] += 1
                return flattened

        return None

    def get_all_contracts_from_forge_reconstructed(self):
        """
        Get all contracts from forge_reconstructed (train/val/test splits)
        Returns list of (contract_file, vulnerability_class)
        """
        contracts = []

        for split in ['train', 'val', 'test']:
            split_dir = self.forge_reconstructed / split
            if not split_dir.exists():
                continue

            # Iterate through vulnerability classes
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue

                vuln_class = class_dir.name

                # Find all .sol files
                for sol_file in class_dir.rglob('*.sol'):
                    # Skip library files
                    if self.is_library_file(sol_file):
                        self.stats['skipped_library'] += 1
                        continue

                    contracts.append((sol_file, vuln_class, split))

        return contracts

    def process_all_contracts(self):
        """
        Process only contracts from forge_reconstructed (1,148 labeled contracts)
        """
        logger.info("=" * 80)
        logger.info("PERFECT FLATTENING V3 - SMART OPTIMIZATIONS")
        logger.info("=" * 80)
        logger.info("Strategy:")
        logger.info("  1. Install OpenZeppelin ONCE globally")
        logger.info("  2. Only flatten 1,148 contracts with labels")
        logger.info("  3. Try flattening FIRST (fast)")
        logger.info("  4. Install dependencies ONLY if needed (slow)")
        logger.info("  5. Skip library files")
        logger.info("=" * 80)
        logger.info("")

        # Setup global libraries ONCE
        self.setup_global_libraries()

        # Get all contracts to flatten
        contracts = self.get_all_contracts_from_forge_reconstructed()
        logger.info(f"Found {len(contracts)} contracts to flatten")
        logger.info(f"Skipped {self.stats['skipped_library']} library files")
        logger.info("")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Process each contract
        for i, (contract_file, vuln_class, split) in enumerate(contracts):
            # Get project directory (parent of contract)
            project_dir = contract_file.parent
            while project_dir.parent != self.forge_reconstructed and project_dir.parent.name not in ['train', 'val', 'test']:
                project_dir = project_dir.parent

            # Output name preserves split and class
            output_name = f"{split}_{vuln_class}_{contract_file.name}"
            output_path = self.output_dir / output_name

            # Skip if already done
            if output_path.exists():
                logger.info(f"[{i+1}/{len(contracts)}] SKIP (exists): {contract_file.name}")
                self.stats['skipped_exists'] += 1
                continue

            logger.info(f"[{i+1}/{len(contracts)}] {contract_file.name} ({vuln_class})")

            # Smart flattening
            flattened = self.flatten_contract_smart(contract_file, project_dir)

            if flattened:
                with open(output_path, 'w') as f:
                    f.write(flattened)
                logger.info(f"    ✅ SUCCESS")
                self.stats['success'] += 1
            else:
                logger.warning(f"    ❌ FAILED")
                self.stats['failed'] += 1

            # Progress report every 100 contracts
            if (i + 1) % 100 == 0:
                self.print_progress()

        self.print_final_report()

    def print_progress(self):
        """Print current progress"""
        total = self.stats['success'] + self.stats['failed']
        if total == 0:
            return

        success_rate = 100 * self.stats['success'] / total
        first_try_rate = 100 * self.stats.get('success_first_try', 0) / total

        logger.info(f"\n  Progress: {total} contracts processed")
        logger.info(f"  Success: {self.stats['success']} ({success_rate:.1f}%)")
        logger.info(f"  Success on first try: {self.stats.get('success_first_try', 0)} ({first_try_rate:.1f}%)")
        logger.info(f"  Success after deps: {self.stats.get('success_after_deps', 0)}")
        logger.info(f"  Failed: {self.stats['failed']}\n")

    def print_final_report(self):
        """Print final report"""
        logger.info("\n" + "=" * 80)
        logger.info("SMART FLATTENING COMPLETE!")
        logger.info("=" * 80)

        total = self.stats['success'] + self.stats['failed']
        success_rate = 100 * self.stats['success'] / total if total > 0 else 0
        first_try_rate = 100 * self.stats.get('success_first_try', 0) / total if total > 0 else 0

        logger.info(f"\nTotal processed: {total}")
        logger.info(f"Successfully flattened: {self.stats['success']} ({success_rate:.1f}%)")
        logger.info(f"  - First try (no deps): {self.stats.get('success_first_try', 0)} ({first_try_rate:.1f}%)")
        logger.info(f"  - After installing deps: {self.stats.get('success_after_deps', 0)}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Skipped (library files): {self.stats['skipped_library']}")
        logger.info(f"Skipped (already done): {self.stats['skipped_exists']}")
        logger.info(f"\nOutput: {self.output_dir}")

        # Save stats
        stats_file = self.output_dir / 'flattening_stats_v3.json'
        with open(stats_file, 'w') as f:
            json.dump(dict(self.stats), f, indent=2)
        logger.info(f"Stats saved: {stats_file}")

if __name__ == "__main__":
    flattener = SmartFlattener(
        forge_reconstructed_dir="data/datasets/forge_reconstructed",
        output_dir="data/datasets/forge_smart_flattened"
    )
    flattener.process_all_contracts()
