#!/usr/bin/env python3
"""
Filter Flattened Contracts - Keep Only Implementations

Removes interfaces, libraries, and abstract contracts that cannot have PDGs extracted.
Only keeps contracts with actual implementation code.
"""
import os
import re
from pathlib import Path
import shutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContractFilter:
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.stats = {
            'total': 0,
            'kept': 0,
            'filtered_interface': 0,
            'filtered_library': 0,
            'filtered_abstract': 0,
            'filtered_no_code': 0
        }

    def is_interface(self, content: str) -> bool:
        """Check if contract is an interface"""
        # Check for interface keyword
        if re.search(r'\binterface\s+\w+', content, re.MULTILINE):
            return True

        # Check if filename starts with I (common interface naming)
        # and has no function implementations
        if not re.search(r'function\s+\w+\([^)]*\)\s*[^;]*\{', content):
            return True

        return False

    def is_library(self, content: str) -> bool:
        """Check if contract is a library"""
        return bool(re.search(r'\blibrary\s+\w+', content, re.MULTILINE))

    def is_abstract(self, content: str) -> bool:
        """Check if contract is abstract"""
        return bool(re.search(r'\babstract\s+contract\s+\w+', content, re.MULTILINE))

    def has_implementation(self, content: str) -> bool:
        """Check if contract has actual implementation code"""
        # Look for function implementations (not just declarations)
        function_implementations = re.findall(
            r'function\s+\w+\([^)]*\)\s*[^;]*\{',
            content
        )

        # Must have at least one function with body
        if len(function_implementations) == 0:
            return False

        # Check for actual code inside functions (not just empty braces)
        # Look for statements, variable declarations, etc.
        has_code = bool(re.search(r'\{\s*\w+', content))

        return has_code

    def should_keep(self, file_path: Path) -> tuple[bool, str]:
        """
        Determine if contract should be kept
        Returns (should_keep, reason)
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Filter out interfaces
            if self.is_interface(content):
                return False, 'interface'

            # Filter out libraries
            if self.is_library(content):
                return False, 'library'

            # Filter out abstract contracts
            if self.is_abstract(content):
                return False, 'abstract'

            # Filter out contracts with no implementation
            if not self.has_implementation(content):
                return False, 'no_code'

            return True, 'implementation'

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return False, 'error'

    def filter_all(self):
        """Filter all contracts"""
        logger.info("=" * 80)
        logger.info("FILTERING FLATTENED CONTRACTS")
        logger.info("=" * 80)
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get all .sol files
        all_files = list(self.source_dir.glob('*.sol'))
        self.stats['total'] = len(all_files)

        logger.info(f"Found {len(all_files)} flattened contracts")
        logger.info("")

        # Process each file
        for i, file_path in enumerate(all_files):
            should_keep, reason = self.should_keep(file_path)

            if should_keep:
                # Copy to output directory
                output_path = self.output_dir / file_path.name
                shutil.copy2(file_path, output_path)
                self.stats['kept'] += 1

                if (i + 1) % 100 == 0:
                    logger.info(f"[{i+1}/{len(all_files)}] ✅ KEPT: {file_path.name}")
            else:
                # Track reason for filtering
                self.stats[f'filtered_{reason}'] += 1

                if (i + 1) % 100 == 0:
                    logger.info(f"[{i+1}/{len(all_files)}] ❌ FILTERED ({reason}): {file_path.name}")

            # Progress report every 500 files
            if (i + 1) % 500 == 0:
                self.print_progress()

        self.print_final_report()

    def print_progress(self):
        """Print current progress"""
        total_processed = self.stats['kept'] + sum(
            v for k, v in self.stats.items() if k.startswith('filtered_')
        )

        if total_processed == 0:
            return

        kept_pct = 100 * self.stats['kept'] / total_processed

        logger.info(f"\n  Progress: {total_processed} / {self.stats['total']}")
        logger.info(f"  Kept: {self.stats['kept']} ({kept_pct:.1f}%)")
        logger.info(f"  Filtered: {total_processed - self.stats['kept']}\n")

    def print_final_report(self):
        """Print final report"""
        logger.info("\n" + "=" * 80)
        logger.info("FILTERING COMPLETE!")
        logger.info("=" * 80)

        total_filtered = sum(
            v for k, v in self.stats.items() if k.startswith('filtered_')
        )

        logger.info(f"\nTotal processed: {self.stats['total']}")
        logger.info(f"✅ Kept (implementations): {self.stats['kept']} ({100*self.stats['kept']/self.stats['total']:.1f}%)")
        logger.info(f"❌ Filtered: {total_filtered} ({100*total_filtered/self.stats['total']:.1f}%)")
        logger.info(f"  - Interfaces: {self.stats['filtered_interface']}")
        logger.info(f"  - Libraries: {self.stats['filtered_library']}")
        logger.info(f"  - Abstract: {self.stats['filtered_abstract']}")
        logger.info(f"  - No code: {self.stats['filtered_no_code']}")
        logger.info(f"\nOutput: {self.output_dir}")
        logger.info(f"Ready for PDG extraction!")

if __name__ == "__main__":
    filter = ContractFilter(
        source_dir="data/datasets/forge_perfectly_flattened",
        output_dir="data/datasets/forge_flattened_implementations"
    )
    filter.filter_all()
