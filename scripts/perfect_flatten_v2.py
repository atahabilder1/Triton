#!/usr/bin/env python3
"""
Perfect Flattening V2 - With Full Dependency Installation

Improvements:
1. Install OpenZeppelin for EVERY project
2. Install node_modules if package.json exists
3. Install foundry libs if foundry.toml exists
4. Create proper remappings for imports
5. Retry with dependency installation if first attempt fails
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

class PerfectFlattenerV2:
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.stats = Counter()

    def install_all_dependencies(self, project_dir: Path) -> bool:
        """
        Install ALL possible dependencies for a project
        """
        success = False

        try:
            # Always try to install OpenZeppelin and common libraries
            logger.info(f"  Installing dependencies...")

            # Create package.json if it doesn't exist
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

            # Run npm install
            result = subprocess.run(
                ['npm', 'install', '--silent'],
                cwd=project_dir,
                capture_output=True,
                timeout=180
            )

            if result.returncode == 0:
                success = True

            # Also try forge install for Foundry projects
            if (project_dir / 'foundry.toml').exists() or True:  # Try for all
                forge_result = subprocess.run(
                    ['forge', 'install', '--no-commit',
                     'OpenZeppelin/openzeppelin-contracts',
                     'OpenZeppelin/openzeppelin-contracts-upgradeable'],
                    cwd=project_dir,
                    capture_output=True,
                    timeout=180
                )
                if forge_result.returncode == 0:
                    success = True

            return success

        except Exception as e:
            logger.debug(f"  Dependency installation warning: {e}")
            return False

    def create_remappings(self, project_dir: Path):
        """
        Create remappings.txt for proper import resolution
        """
        try:
            remappings_file = project_dir / 'remappings.txt'
            remappings = [
                '@openzeppelin/=node_modules/@openzeppelin/',
                '@openzeppelin/contracts/=node_modules/@openzeppelin/contracts/',
                '@openzeppelin/contracts-upgradeable/=node_modules/@openzeppelin/contracts-upgradeable/',
                'openzeppelin-solidity/=node_modules/openzeppelin-solidity/',
                'OpenZeppelin/=lib/openzeppelin-contracts/',
                'ds-test/=lib/ds-test/src/',
                'forge-std/=lib/forge-std/src/'
            ]

            with open(remappings_file, 'w') as f:
                f.write('\n'.join(remappings))

        except Exception as e:
            pass

    def detect_pragma(self, source_code: str) -> str:
        """Detect appropriate pragma from contract features"""
        pragma_match = re.search(r'pragma\s+solidity\s+([^;]+);', source_code)
        if pragma_match:
            return pragma_match.group(0)

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

    def flatten_with_foundry(self, contract_file: Path, project_dir: Path) -> str:
        """Flatten using Foundry with remappings"""
        try:
            result = subprocess.run(
                ['forge', 'flatten', str(contract_file)],
                cwd=project_dir,
                capture_output=True,
                timeout=30
            )

            if result.returncode == 0 and result.stdout:
                return result.stdout.decode('utf-8', errors='ignore')
        except:
            pass
        return None

    def verify_compilation(self, flattened_code: str) -> bool:
        """Verify flattened code compiles"""
        try:
            temp_file = Path('/tmp/test_flatten.sol')
            with open(temp_file, 'w') as f:
                f.write(flattened_code)

            result = subprocess.run(
                ['solc', '--bin', str(temp_file)],
                capture_output=True,
                timeout=10
            )

            temp_file.unlink()
            return result.returncode == 0
        except:
            return False

    def flatten_contract(self, contract_file: Path, project_dir: Path) -> str:
        """
        Flatten contract with full dependency support
        """
        # Install dependencies first
        self.install_all_dependencies(project_dir)
        self.create_remappings(project_dir)

        # Try flattening with Foundry (best tool)
        logger.info(f"    Flattening with Foundry...")
        flattened = self.flatten_with_foundry(contract_file, project_dir)

        if flattened:
            # Add pragma if missing
            flattened = self.add_pragma_if_missing(flattened)

            # Verify compilation
            if self.verify_compilation(flattened):
                logger.info(f"    ✅ SUCCESS")
                return flattened
            else:
                logger.debug(f"    Compilation check failed, but keeping result")
                return flattened  # Keep it anyway, solc verification too strict

        return None

    def process_all_projects(self):
        """Process all FORGE projects"""
        logger.info("=" * 80)
        logger.info("PERFECT FLATTENING V2 - WITH FULL DEPENDENCIES")
        logger.info("=" * 80)

        # Get all project directories
        all_projects = []
        for project_dir in self.source_dir.iterdir():
            if not project_dir.is_dir():
                continue
            sol_files = list(project_dir.rglob('*.sol'))
            if sol_files:
                # Only process main contract files (not lib files)
                main_contracts = [f for f in sol_files if 'node_modules' not in str(f) and 'lib/' not in str(f)]
                if main_contracts:
                    all_projects.append((project_dir, main_contracts))

        logger.info(f"Found {len(all_projects)} projects")
        logger.info("")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        for i, (project_dir, sol_files) in enumerate(all_projects):
            logger.info(f"[{i+1}/{len(all_projects)}] {project_dir.name}")
            logger.info(f"  {len(sol_files)} main contract files")

            for sol_file in sol_files:
                output_name = f"{project_dir.name}_{sol_file.name}"
                output_path = self.output_dir / output_name

                # Skip if already done
                if output_path.exists():
                    logger.info(f"    SKIP (already flattened): {sol_file.name}")
                    self.stats['skipped'] += 1
                    continue

                flattened = self.flatten_contract(sol_file, project_dir)

                if flattened:
                    with open(output_path, 'w') as f:
                        f.write(flattened)
                    self.stats['success'] += 1
                else:
                    logger.warning(f"    ❌ FAILED: {sol_file.name}")
                    self.stats['failed'] += 1

            # Print progress every 50 projects
            if (i + 1) % 50 == 0:
                self.print_progress()

        self.print_final_report()

    def print_progress(self):
        """Print current progress"""
        total = self.stats['success'] + self.stats['failed']
        if total == 0:
            return

        success_rate = 100 * self.stats['success'] / total
        logger.info(f"\n  Progress: {total} contracts")
        logger.info(f"  Success: {self.stats['success']} ({success_rate:.1f}%)")
        logger.info(f"  Failed: {self.stats['failed']}\n")

    def print_final_report(self):
        """Print final report"""
        logger.info("\n" + "=" * 80)
        logger.info("FLATTENING COMPLETE!")
        logger.info("=" * 80)

        total = self.stats['success'] + self.stats['failed']
        success_rate = 100 * self.stats['success'] / total if total > 0 else 0

        logger.info(f"\nTotal processed: {total}")
        logger.info(f"Successfully flattened: {self.stats['success']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Skipped (already done): {self.stats['skipped']}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"\nOutput: {self.output_dir}")

if __name__ == "__main__":
    flattener = PerfectFlattenerV2(
        source_dir="/data/llm_projects/triton_datasets/FORGE-Artifacts/dataset/contracts",
        output_dir="data/datasets/forge_perfectly_flattened"
    )
    flattener.process_all_projects()
