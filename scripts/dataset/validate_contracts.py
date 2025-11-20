#!/usr/bin/env python3
"""
Smart Contract Validation Script
Filters out problematic contracts before training
- Abstract contracts
- Contracts with missing dependencies
- Contracts that fail compilation
"""

import sys
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import argparse
from collections import defaultdict
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.slither_wrapper import SlitherWrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContractValidator:
    """Validates and filters smart contracts for training"""

    def __init__(self):
        self.slither = SlitherWrapper(timeout=30)
        self.stats = defaultdict(int)

    def is_abstract_contract(self, source_code: str) -> bool:
        """Check if contract is abstract"""
        # Check for 'abstract contract' keyword
        if re.search(r'\babstract\s+contract\b', source_code):
            return True

        # Check for unimplemented functions
        if re.search(r'function\s+\w+\([^)]*\)\s+(?:public|external|internal|private)?\s*(?:view|pure)?\s*;', source_code):
            return True

        return False

    def has_missing_dependencies(self, source_code: str) -> Tuple[bool, List[str]]:
        """Check for missing imports/dependencies"""
        missing = []

        # Find all import statements
        import_pattern = r'import\s+["\']([^"\']+)["\']'
        imports = re.findall(import_pattern, source_code)

        for imp in imports:
            # Check if it's a relative import (not node_modules/@openzeppelin etc)
            if not imp.startswith('@') and not imp.startswith('http'):
                missing.append(imp)

        return len(missing) > 0, missing

    def is_compilable(self, source_code: str) -> Tuple[bool, Optional[str]]:
        """Test if contract can be analyzed by Slither"""
        try:
            result = self.slither.analyze_contract(source_code)

            if not result['success']:
                return False, result.get('error', 'Unknown error')

            # Check if we got a valid PDG
            pdg = result.get('pdg')
            if pdg is None or pdg.number_of_nodes() == 0:
                return False, "Empty or invalid PDG"

            return True, None

        except Exception as e:
            return False, str(e)

    def is_too_simple(self, source_code: str) -> bool:
        """Check if contract is too simple to be useful"""
        # Count functions
        function_count = len(re.findall(r'\bfunction\s+\w+', source_code))

        # Count state variables
        state_var_count = len(re.findall(r'^\s*(uint|int|bool|address|string|bytes|mapping)\s+(?:public|private|internal)?\s*\w+', source_code, re.MULTILINE))

        # Too simple if < 2 functions or no state variables
        if function_count < 2 and state_var_count == 0:
            return True

        # Check code length (excluding comments)
        code_no_comments = re.sub(r'//.*$', '', source_code, flags=re.MULTILINE)
        code_no_comments = re.sub(r'/\*.*?\*/', '', code_no_comments, flags=re.DOTALL)
        lines = [l.strip() for l in code_no_comments.split('\n') if l.strip()]

        if len(lines) < 10:
            return True

        return False

    def has_syntax_errors(self, source_code: str) -> Tuple[bool, Optional[str]]:
        """Basic syntax validation"""
        # Check for balanced braces
        open_braces = source_code.count('{')
        close_braces = source_code.count('}')

        if open_braces != close_braces:
            return True, f"Unbalanced braces: {open_braces} open, {close_braces} close"

        # Check for pragma statement
        if not re.search(r'pragma\s+solidity', source_code):
            return True, "Missing pragma solidity statement"

        # Check for at least one contract definition
        if not re.search(r'\bcontract\s+\w+', source_code):
            return True, "No contract definition found"

        return False, None

    def validate_contract(self, source_code: str, file_path: str) -> Tuple[bool, str]:
        """
        Validate a contract and return (is_valid, reason)

        Returns:
            (True, "OK") if contract is valid
            (False, reason) if contract is invalid
        """
        # 1. Check syntax
        has_errors, error = self.has_syntax_errors(source_code)
        if has_errors:
            self.stats['syntax_error'] += 1
            return False, f"Syntax error: {error}"

        # 2. Check if abstract
        if self.is_abstract_contract(source_code):
            self.stats['abstract'] += 1
            return False, "Abstract contract"

        # 3. Check if too simple
        if self.is_too_simple(source_code):
            self.stats['too_simple'] += 1
            return False, "Contract too simple"

        # 4. Check for missing dependencies (warning only for external imports)
        has_missing, missing = self.has_missing_dependencies(source_code)
        if has_missing:
            # Only fail if imports look critical
            critical_missing = [m for m in missing if not any(skip in m for skip in ['@openzeppelin', 'hardhat', 'forge-std'])]
            if critical_missing:
                self.stats['missing_deps'] += 1
                logger.warning(f"{file_path}: Has imports but may be OK: {critical_missing[:2]}")

        # 5. Test compilation (most important check)
        is_compilable, compile_error = self.is_compilable(source_code)
        if not is_compilable:
            self.stats['compilation_failed'] += 1
            return False, f"Compilation failed: {compile_error}"

        # All checks passed!
        self.stats['valid'] += 1
        return True, "OK"

    def print_stats(self):
        """Print validation statistics"""
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION STATISTICS")
        logger.info("=" * 80)
        total = sum(self.stats.values())
        logger.info(f"Total contracts processed: {total}")
        logger.info(f"Valid contracts: {self.stats['valid']} ({self.stats['valid']/total*100:.1f}%)")
        logger.info(f"Abstract contracts: {self.stats['abstract']}")
        logger.info(f"Too simple: {self.stats['too_simple']}")
        logger.info(f"Syntax errors: {self.stats['syntax_error']}")
        logger.info(f"Missing dependencies: {self.stats['missing_deps']}")
        logger.info(f"Compilation failed: {self.stats['compilation_failed']}")
        logger.info("=" * 80 + "\n")


def validate_dataset(
    input_dir: str,
    output_dir: Optional[str] = None,
    copy_valid: bool = False
) -> Dict[str, int]:
    """
    Validate all contracts in a dataset

    Args:
        input_dir: Directory containing contracts
        output_dir: If provided, copy valid contracts here
        copy_valid: Whether to copy valid contracts to output_dir

    Returns:
        Statistics dictionary
    """
    validator = ContractValidator()
    input_path = Path(input_dir)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Validating contracts in: {input_dir}")
    logger.info("=" * 80)

    # Process all .sol files
    sol_files = list(input_path.rglob("*.sol"))
    logger.info(f"Found {len(sol_files)} Solidity files\n")

    valid_files = []
    invalid_files = []

    for idx, sol_file in enumerate(sol_files, 1):
        try:
            # Read contract
            with open(sol_file, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()

            # Validate
            is_valid, reason = validator.validate_contract(source_code, str(sol_file))

            if is_valid:
                valid_files.append(sol_file)
                logger.info(f"[{idx}/{len(sol_files)}] ✓ {sol_file.name} - VALID")

                # Copy if requested
                if copy_valid and output_dir:
                    # Preserve directory structure
                    rel_path = sol_file.relative_to(input_path)
                    dest_file = output_path / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(sol_file, dest_file)
            else:
                invalid_files.append((sol_file, reason))
                logger.warning(f"[{idx}/{len(sol_files)}] ✗ {sol_file.name} - {reason}")

        except Exception as e:
            logger.error(f"[{idx}/{len(sol_files)}] ERROR processing {sol_file}: {e}")
            invalid_files.append((sol_file, f"Processing error: {e}"))

        # Print progress every 50 files
        if idx % 50 == 0:
            logger.info(f"\nProgress: {idx}/{len(sol_files)} ({idx/len(sol_files)*100:.1f}%)")
            validator.print_stats()

    # Final statistics
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Valid contracts: {len(valid_files)}/{len(sol_files)} ({len(valid_files)/len(sol_files)*100:.1f}%)")
    logger.info(f"Invalid contracts: {len(invalid_files)}/{len(sol_files)} ({len(invalid_files)/len(sol_files)*100:.1f}%)")

    if output_dir and copy_valid:
        logger.info(f"\nValid contracts copied to: {output_dir}")

    validator.print_stats()

    # Save invalid files list
    if invalid_files:
        invalid_log = Path(input_dir) / "invalid_contracts.txt"
        with open(invalid_log, 'w') as f:
            f.write("Invalid Contracts Log\n")
            f.write("=" * 80 + "\n\n")
            for file_path, reason in invalid_files:
                f.write(f"{file_path}\n  Reason: {reason}\n\n")
        logger.info(f"\nInvalid contracts list saved to: {invalid_log}")

    return validator.stats


def main():
    parser = argparse.ArgumentParser(description="Validate Smart Contracts for Training")
    parser.add_argument("input_dir", help="Directory containing contracts to validate")
    parser.add_argument("--output-dir", help="Directory to copy valid contracts to")
    parser.add_argument("--copy-valid", action='store_true', help="Copy valid contracts to output directory")
    parser.add_argument("--quick-check", action='store_true', help="Skip compilation test (faster but less accurate)")

    args = parser.parse_args()

    if args.copy_valid and not args.output_dir:
        parser.error("--copy-valid requires --output-dir")

    # Run validation
    stats = validate_dataset(
        args.input_dir,
        output_dir=args.output_dir,
        copy_valid=args.copy_valid
    )

    # Exit code based on results
    if stats['valid'] == 0:
        logger.error("\nNo valid contracts found!")
        sys.exit(1)
    elif stats['valid'] < stats['compilation_failed']:
        logger.warning("\nMore invalid than valid contracts found")
        sys.exit(2)
    else:
        logger.info("\n✓ Validation successful!")
        sys.exit(0)


if __name__ == "__main__":
    main()
