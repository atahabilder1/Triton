#!/usr/bin/env python3
"""
Perfect Flattening Pipeline for FORGE Contracts

This script will:
1. Try multiple flattening methods (Hardhat, Foundry, truffle-flattener)
2. Install dependencies for each project (npm install, forge install)
3. Auto-detect and add missing pragma statements
4. Verify compilation with solc
5. Achieve 100% successful flattening

Strategy:
- For each project in FORGE:
  1. Check if has package.json → npm install
  2. Check if has foundry.toml → forge install
  3. Try Hardhat flatten
  4. If fails, try Foundry flatten
  5. If fails, try truffle-flattener
  6. Auto-add pragma if missing
  7. Verify with solc compilation
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

class PerfectFlattener:
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.stats = Counter()

        # Tools available
        self.has_hardhat = shutil.which('npx')
        self.has_forge = shutil.which('forge')
        self.has_truffle_flattener = Path('node_modules/.bin/truffle-flattener').exists()

        logger.info(f"Tools available:")
        logger.info(f"  Hardhat (npx): {self.has_hardhat}")
        logger.info(f"  Foundry (forge): {self.has_forge}")
        logger.info(f"  Truffle-flattener: {self.has_truffle_flattener}")

    def detect_pragma(self, source_code: str) -> str:
        """
        Detect appropriate pragma from contract features
        """
        # Look for existing pragma
        pragma_match = re.search(r'pragma\s+solidity\s+([^;]+);', source_code)
        if pragma_match:
            return pragma_match.group(0)

        # Detect Solidity version from features
        if 'unchecked' in source_code:
            return 'pragma solidity ^0.8.0;'
        elif 'constructor(' in source_code:
            return 'pragma solidity ^0.5.0;'
        elif 'function' in source_code and 'view' in source_code:
            return 'pragma solidity ^0.4.24;'
        else:
            return 'pragma solidity ^0.8.0;'  # Default to latest

    def add_pragma_if_missing(self, source_code: str) -> str:
        """
        Add pragma statement if missing
        """
        if 'pragma solidity' in source_code:
            return source_code

        pragma = self.detect_pragma(source_code)

        # Insert after SPDX if present
        if 'SPDX-License-Identifier' in source_code:
            lines = source_code.split('\n')
            for i, line in enumerate(lines):
                if 'SPDX-License-Identifier' in line:
                    lines.insert(i + 1, '')
                    lines.insert(i + 2, pragma)
                    break
            return '\n'.join(lines)
        else:
            # Insert at top
            return f"{pragma}\n\n{source_code}"

    def install_dependencies(self, project_dir: Path) -> bool:
        """
        Install dependencies for a project
        """
        try:
            # Check for package.json
            if (project_dir / 'package.json').exists():
                logger.info(f"  Installing npm dependencies...")
                result = subprocess.run(
                    ['npm', 'install'],
                    cwd=project_dir,
                    capture_output=True,
                    timeout=120
                )
                if result.returncode == 0:
                    return True

            # Check for foundry.toml
            if (project_dir / 'foundry.toml').exists():
                logger.info(f"  Installing forge dependencies...")
                result = subprocess.run(
                    ['forge', 'install'],
                    cwd=project_dir,
                    capture_output=True,
                    timeout=120
                )
                if result.returncode == 0:
                    return True

            return True  # No dependencies needed
        except Exception as e:
            logger.warning(f"  Dependency installation failed: {e}")
            return False

    def flatten_with_hardhat(self, contract_file: Path, project_dir: Path) -> str:
        """
        Flatten using Hardhat
        """
        try:
            # Create minimal hardhat config
            config_content = '''module.exports = {
  solidity: "0.8.0",
  paths: {
    sources: "."
  }
};'''
            config_file = project_dir / 'hardhat.config.js'
            with open(config_file, 'w') as f:
                f.write(config_content)

            result = subprocess.run(
                ['npx', 'hardhat', 'flatten', str(contract_file)],
                cwd=project_dir,
                capture_output=True,
                timeout=30
            )

            if result.returncode == 0 and result.stdout:
                return result.stdout.decode('utf-8', errors='ignore')
        except Exception as e:
            pass
        return None

    def flatten_with_foundry(self, contract_file: Path, project_dir: Path) -> str:
        """
        Flatten using Foundry
        """
        try:
            result = subprocess.run(
                ['forge', 'flatten', str(contract_file)],
                cwd=project_dir,
                capture_output=True,
                timeout=30
            )

            if result.returncode == 0 and result.stdout:
                return result.stdout.decode('utf-8', errors='ignore')
        except Exception as e:
            pass
        return None

    def flatten_with_truffle(self, contract_file: Path) -> str:
        """
        Flatten using truffle-flattener
        """
        try:
            result = subprocess.run(
                ['node_modules/.bin/truffle-flattener', str(contract_file)],
                capture_output=True,
                timeout=30
            )

            if result.returncode == 0 and result.stdout:
                return result.stdout.decode('utf-8', errors='ignore')
        except Exception as e:
            pass
        return None

    def verify_compilation(self, flattened_code: str) -> bool:
        """
        Verify flattened code compiles with solc
        """
        try:
            # Write to temp file
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
        Try all flattening methods until one succeeds
        """
        methods = [
            ('Hardhat', self.flatten_with_hardhat if self.has_hardhat else None),
            ('Foundry', self.flatten_with_foundry if self.has_forge else None),
            ('Truffle', self.flatten_with_truffle if self.has_truffle_flattener else None),
        ]

        for method_name, method_func in methods:
            if method_func is None:
                continue

            try:
                logger.info(f"    Trying {method_name}...")
                if method_name == 'Truffle':
                    flattened = method_func(contract_file)
                else:
                    flattened = method_func(contract_file, project_dir)

                if flattened:
                    # Add pragma if missing
                    flattened = self.add_pragma_if_missing(flattened)

                    # Verify compilation
                    if self.verify_compilation(flattened):
                        logger.info(f"    ✅ SUCCESS with {method_name}")
                        self.stats[f'success_{method_name.lower()}'] += 1
                        return flattened
            except Exception as e:
                logger.debug(f"    {method_name} failed: {e}")

        return None

    def process_all_projects(self):
        """
        Process all FORGE projects
        """
        logger.info("=" * 80)
        logger.info("PERFECT FLATTENING PIPELINE")
        logger.info("=" * 80)

        # Get all project directories
        all_projects = []
        for project_dir in self.source_dir.iterdir():
            if not project_dir.is_dir():
                continue
            # Find all .sol files in project
            sol_files = list(project_dir.rglob('*.sol'))
            if sol_files:
                all_projects.append((project_dir, sol_files))

        logger.info(f"Found {len(all_projects)} projects")
        logger.info("")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        for i, (project_dir, sol_files) in enumerate(all_projects):
            logger.info(f"[{i+1}/{len(all_projects)}] Processing: {project_dir.name}")
            logger.info(f"  Found {len(sol_files)} .sol files")

            # Install dependencies
            self.install_dependencies(project_dir)

            # Flatten each contract
            for sol_file in sol_files:
                output_name = f"{project_dir.name}_{sol_file.name}"
                output_path = self.output_dir / output_name

                flattened = self.flatten_contract(sol_file, project_dir)

                if flattened:
                    with open(output_path, 'w') as f:
                        f.write(flattened)
                    self.stats['total_success'] += 1
                else:
                    logger.warning(f"    ❌ FAILED: {sol_file.name}")
                    self.stats['total_failed'] += 1

            if (i + 1) % 50 == 0:
                self.print_stats()

        self.print_final_report()

    def print_stats(self):
        """
        Print current statistics
        """
        total = self.stats['total_success'] + self.stats['total_failed']
        if total == 0:
            return

        success_rate = 100 * self.stats['total_success'] / total
        logger.info(f"\n  Progress: {total} contracts processed")
        logger.info(f"  Success: {self.stats['total_success']} ({success_rate:.1f}%)")
        logger.info(f"  Failed: {self.stats['total_failed']}\n")

    def print_final_report(self):
        """
        Print final flattening report
        """
        logger.info("\n" + "=" * 80)
        logger.info("FLATTENING COMPLETE!")
        logger.info("=" * 80)

        total = self.stats['total_success'] + self.stats['total_failed']
        success_rate = 100 * self.stats['total_success'] / total if total > 0 else 0

        logger.info(f"\nTotal contracts: {total}")
        logger.info(f"Successfully flattened: {self.stats['total_success']}")
        logger.info(f"Failed: {self.stats['total_failed']}")
        logger.info(f"Success rate: {success_rate:.1f}%")

        logger.info(f"\nBy method:")
        for method in ['hardhat', 'foundry', 'truffle']:
            count = self.stats.get(f'success_{method}', 0)
            if count > 0:
                logger.info(f"  {method.capitalize()}: {count}")

        logger.info(f"\nOutput directory: {self.output_dir}")
        logger.info("")

if __name__ == "__main__":
    flattener = PerfectFlattener(
        source_dir="/data/llm_projects/triton_datasets/FORGE-Artifacts/dataset/contracts",
        output_dir="data/datasets/forge_perfectly_flattened"
    )

    flattener.process_all_projects()
