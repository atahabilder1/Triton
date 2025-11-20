#!/usr/bin/env python3
"""
Organize Flattened FORGE Contracts by Vulnerability Class
Step 2 of Approach A: Map CWEs → classes, filter, balance, split

Process:
1. Read flattened contracts from Step 1
2. Map each to vulnerability class using audit CWE codes
3. Filter out bad quality contracts (interfaces, tiny files)
4. Balance dataset (sample per class)
5. Split into train/val/test (70/15/15)

Input: forge_flattened_all/ (from Step 1)
Output: forge_reconstructed/train|val|test/<class>/*.sol
"""

import sys
import json
import shutil
import random
from pathlib import Path
from typing import Optional, Dict, List, Set
import logging
import argparse
from collections import defaultdict
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# CWE to Vulnerability Class Mapping (COMPLETE - from FORGE audit analysis)
# Based on analysis of ALL 6,454 FORGE audit reports
CWE_TO_CLASS = {
    # Access Control (3,077 contracts in FORGE)
    'CWE-284': 'access_control',   # Improper Access Control (main)
    'CWE-1068': 'access_control',  # Incomplete Access Control Checks
    'CWE-269': 'access_control',   # Improper Privilege Management
    'CWE-287': 'access_control',   # Improper Authentication
    'CWE-863': 'access_control',   # Incorrect Authorization

    # Arithmetic (1,780 contracts in FORGE)
    'CWE-682': 'arithmetic',       # Incorrect Calculation (main)
    'CWE-190': 'arithmetic',       # Integer Overflow
    'CWE-191': 'arithmetic',       # Integer Underflow

    # Unchecked Low-Level Calls (2,551 contracts in FORGE)
    'CWE-703': 'unchecked_low_level_calls',  # Improper Error Handling (main)
    'CWE-20': 'unchecked_low_level_calls',   # Improper Input Validation
    'CWE-252': 'unchecked_low_level_calls',  # Unchecked Return Value

    # Denial of Service (1,043 contracts in FORGE)
    'CWE-691': 'denial_of_service',  # Insufficient Control Flow Management (main)
    'CWE-664': 'denial_of_service',  # Improper Control of Resource
    'CWE-400': 'denial_of_service',  # Uncontrolled Resource Consumption
    'CWE-770': 'denial_of_service',  # Allocation without Limits

    # Time Manipulation (54 contracts in FORGE)
    'CWE-829': 'time_manipulation',  # Untrusted Control Sphere (timestamp dependency)

    # Bad Randomness (1 contract in FORGE - very rare)
    'CWE-330': 'bad_randomness',     # Insufficient Randomness

    # Reentrancy (NOT in FORGE - will add from SmartBugs)
    'CWE-362': 'reentrancy',         # Race Condition
    'CWE-841': 'reentrancy',         # Improper Enforcement of Behavioral Workflow

    # Front Running (NOT in FORGE - will add from SmartBugs)
    # Uses same CWE as reentrancy but lower priority

    # Other / Low Priority (2,811 contracts in FORGE)
    'CWE-710': 'other',    # Coding Standards Violation (main)
    'CWE-435': 'other',    # Improper Interaction Between Components
    'CWE-693': 'other',    # Protection Mechanism Failure
    'CWE-697': 'other',    # Incorrect Comparison
    'CWE-707': 'other',    # Improper Neutralization
    'CWE-683': 'other',    # Function Call With Incorrect Order
    'CWE-1164': 'other',   # Irrelevant Code
    'CWE-824': 'other',    # Access of Uninitialized Pointer
    'CWE-684': 'other',    # Incorrect Provision of Functionality
}

# Priority order (higher priority = classified first if multiple CWEs)
VULNERABILITY_PRIORITY = [
    'reentrancy',
    'arithmetic',
    'bad_randomness',
    'unchecked_low_level_calls',
    'denial_of_service',
    'front_running',
    'time_manipulation',
    'access_control',
    'other'
]


class ContractOrganizer:
    """Organize flattened contracts by vulnerability class"""

    def __init__(
        self,
        flattened_dir: Path,
        forge_dir: Path,
        output_dir: Path,
        samples_per_class: Dict[str, int],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        self.flattened_dir = flattened_dir
        self.forge_dir = forge_dir
        self.results_dir = forge_dir / "dataset" / "results"
        self.output_dir = output_dir
        self.samples_per_class = samples_per_class
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Statistics
        self.stats = {
            'total_contracts': 0,
            'labeled': 0,
            'filtered_interface': 0,
            'filtered_abstract': 0,
            'filtered_small': 0,
            'filtered_no_impl': 0,
            'no_audit_found': 0,
            'organized': 0,
            'by_class': defaultdict(int)
        }

    def find_audit_json(self, project_name: str) -> Optional[Path]:
        """Find audit JSON for a project"""
        # Extract project name from flattened filename: project_contract.sol
        parts = project_name.split('_')
        if len(parts) >= 2:
            project_base = '_'.join(parts[:-1])  # Remove contract name
        else:
            project_base = project_name

        # Try exact match
        audit_file = self.results_dir / f"{project_base}.pdf.json"
        if audit_file.exists():
            return audit_file

        # Try variations
        for json_file in self.results_dir.glob("*.json"):
            if project_base.lower() in json_file.stem.lower():
                return json_file

        return None

    def extract_cwes_from_audit(self, audit_path: Path) -> List[str]:
        """Extract all CWE codes from audit JSON"""
        try:
            with open(audit_path, 'r') as f:
                data = json.load(f)

            cwes = []
            findings = data.get('findings', [])

            for finding in findings:
                category = finding.get('category', {})
                # CWEs are in category["1"] as list
                cwe_list = category.get('1', [])
                cwes.extend(cwe_list)

            return list(set(cwes))  # Unique CWEs

        except Exception as e:
            logger.debug(f"Error reading audit {audit_path}: {e}")
            return []

    def map_to_vulnerability_class(self, cwes: List[str]) -> str:
        """Map CWE codes to vulnerability class using priority"""
        if not cwes:
            return 'other'

        # Find highest priority vulnerability
        for priority_vuln in VULNERABILITY_PRIORITY:
            for cwe in cwes:
                if CWE_TO_CLASS.get(cwe) == priority_vuln:
                    return priority_vuln

        # Default to 'other'
        return 'other'

    def is_interface(self, content: str) -> bool:
        """Check if contract is an interface"""
        # Remove comments
        content_no_comments = re.sub(r'//.*', '', content)
        content_no_comments = re.sub(r'/\*.*?\*/', '', content_no_comments, flags=re.DOTALL)

        # Check for interface keyword
        if re.search(r'\binterface\s+\w+', content_no_comments):
            return True

        return False

    def is_abstract_no_impl(self, content: str) -> bool:
        """Check if contract is abstract with no implementations"""
        # Remove comments
        content_no_comments = re.sub(r'//.*', '', content)
        content_no_comments = re.sub(r'/\*.*?\*/', '', content_no_comments, flags=re.DOTALL)

        # Check for abstract keyword
        is_abstract = re.search(r'\babstract\s+contract', content_no_comments)

        if is_abstract:
            # Check if has any function implementations
            has_impl = re.search(r'function\s+\w+[^;]*\{', content_no_comments)
            return not has_impl

        return False

    def is_too_small(self, content: str) -> bool:
        """Check if contract is too small (<10 lines of code)"""
        # Count non-empty, non-comment lines
        lines = content.split('\n')
        code_lines = 0

        for line in lines:
            line = line.strip()
            if line and not line.startswith('//') and not line.startswith('/*') and not line.startswith('*'):
                code_lines += 1

        return code_lines < 10

    def has_no_implementations(self, content: str) -> bool:
        """Check if contract has no function implementations"""
        # Remove comments
        content_no_comments = re.sub(r'//.*', '', content)
        content_no_comments = re.sub(r'/\*.*?\*/', '', content_no_comments, flags=re.DOTALL)

        # Check for any function implementations (function ... { ... })
        has_impl = re.search(r'function\s+\w+[^;]*\{', content_no_comments)

        return not has_impl

    def filter_contract(self, contract_path: Path) -> Optional[str]:
        """
        Filter contract based on quality checks

        Returns:
            None if contract should be filtered out
            Filter reason string if filtered
        """
        try:
            with open(contract_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if self.is_interface(content):
                return 'interface'

            if self.is_abstract_no_impl(content):
                return 'abstract_no_impl'

            if self.is_too_small(content):
                return 'too_small'

            if self.has_no_implementations(content):
                return 'no_implementations'

            return None  # Pass filter

        except Exception as e:
            logger.debug(f"Error filtering {contract_path}: {e}")
            return 'read_error'

    def process_contract(self, contract_path: Path) -> Optional[Dict]:
        """
        Process a single contract

        Returns:
            Dict with contract info if valid, None if filtered/failed
        """
        self.stats['total_contracts'] += 1

        # Extract project name from filename
        filename = contract_path.stem  # Remove .sol

        # Filter quality
        filter_reason = self.filter_contract(contract_path)
        if filter_reason:
            if filter_reason == 'interface':
                self.stats['filtered_interface'] += 1
            elif filter_reason == 'abstract_no_impl':
                self.stats['filtered_abstract'] += 1
            elif filter_reason == 'too_small':
                self.stats['filtered_small'] += 1
            elif filter_reason == 'no_implementations':
                self.stats['filtered_no_impl'] += 1
            return None

        # Find audit JSON
        audit_json = self.find_audit_json(filename)
        if not audit_json:
            self.stats['no_audit_found'] += 1
            return None

        # Extract CWEs
        cwes = self.extract_cwes_from_audit(audit_json)

        # Map to vulnerability class
        vuln_class = self.map_to_vulnerability_class(cwes)

        self.stats['labeled'] += 1
        self.stats['by_class'][vuln_class] += 1

        return {
            'path': contract_path,
            'class': vuln_class,
            'cwes': cwes
        }

    def organize_dataset(self):
        """Main organization workflow"""
        logger.info("="*80)
        logger.info("FORGE DATASET ORGANIZATION - Step 2 of Approach A")
        logger.info("="*80)
        logger.info(f"Input: {self.flattened_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Samples per class: {self.samples_per_class}")
        logger.info("="*80 + "\n")

        # Step 1: Process all contracts
        logger.info("Step 1: Processing and labeling contracts...")

        contracts_by_class = defaultdict(list)
        all_contracts = list(self.flattened_dir.glob("*.sol"))

        for idx, contract_path in enumerate(all_contracts, 1):
            if idx % 500 == 0:
                logger.info(f"  Processed {idx}/{len(all_contracts)} contracts...")

            contract_info = self.process_contract(contract_path)
            if contract_info:
                contracts_by_class[contract_info['class']].append(contract_info)

        logger.info(f"\n✓ Processed {self.stats['total_contracts']} contracts")
        logger.info(f"✓ Labeled: {self.stats['labeled']}")
        logger.info(f"✗ Filtered: {self.stats['filtered_interface'] + self.stats['filtered_abstract'] + self.stats['filtered_small'] + self.stats['filtered_no_impl']}")
        logger.info(f"  - Interfaces: {self.stats['filtered_interface']}")
        logger.info(f"  - Abstract (no impl): {self.stats['filtered_abstract']}")
        logger.info(f"  - Too small: {self.stats['filtered_small']}")
        logger.info(f"  - No implementations: {self.stats['filtered_no_impl']}")
        logger.info(f"✗ No audit found: {self.stats['no_audit_found']}\n")

        # Step 2: Balance dataset
        logger.info("Step 2: Balancing dataset...")
        logger.info("\nContracts per class (before balancing):")
        for vuln_class in sorted(contracts_by_class.keys()):
            count = len(contracts_by_class[vuln_class])
            max_samples = self.samples_per_class.get(vuln_class, 1000)
            logger.info(f"  {vuln_class:30s}: {count:5d} (will sample {min(count, max_samples)})")

        # Sample per class
        balanced_contracts = {}
        for vuln_class, contracts in contracts_by_class.items():
            max_samples = self.samples_per_class.get(vuln_class, 1000)

            if len(contracts) > max_samples:
                # Random sample
                sampled = random.sample(contracts, max_samples)
            else:
                # Take all
                sampled = contracts

            balanced_contracts[vuln_class] = sampled

        logger.info("\n✓ Balanced dataset:")
        total_balanced = 0
        for vuln_class in sorted(balanced_contracts.keys()):
            count = len(balanced_contracts[vuln_class])
            total_balanced += count
            logger.info(f"  {vuln_class:30s}: {count:5d}")
        logger.info(f"\nTotal balanced: {total_balanced}\n")

        # Step 3: Split into train/val/test
        logger.info("Step 3: Splitting into train/val/test...")

        for vuln_class, contracts in balanced_contracts.items():
            # Shuffle
            random.shuffle(contracts)

            # Calculate split sizes
            total = len(contracts)
            train_size = int(total * self.train_ratio)
            val_size = int(total * self.val_ratio)

            train_contracts = contracts[:train_size]
            val_contracts = contracts[train_size:train_size + val_size]
            test_contracts = contracts[train_size + val_size:]

            # Copy to output directories
            for split_name, split_contracts in [
                ('train', train_contracts),
                ('val', val_contracts),
                ('test', test_contracts)
            ]:
                if not split_contracts:
                    continue

                output_class_dir = self.output_dir / split_name / vuln_class
                output_class_dir.mkdir(parents=True, exist_ok=True)

                for contract_info in split_contracts:
                    src = contract_info['path']
                    dst = output_class_dir / src.name
                    shutil.copy2(src, dst)
                    self.stats['organized'] += 1

        logger.info("✓ Dataset organized!\n")

        # Step 4: Summary
        logger.info("="*80)
        logger.info("ORGANIZATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Total contracts processed: {self.stats['total_contracts']}")
        logger.info(f"Labeled: {self.stats['labeled']}")
        logger.info(f"Organized: {self.stats['organized']}")
        logger.info(f"\nOutput structure:")

        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            if split_dir.exists():
                logger.info(f"\n{split}/")
                for class_dir in sorted(split_dir.iterdir()):
                    if class_dir.is_dir():
                        count = len(list(class_dir.glob("*.sol")))
                        logger.info(f"  {class_dir.name:30s}: {count:5d} contracts")

        logger.info(f"\n✓ Dataset saved to: {self.output_dir}")
        logger.info("="*80 + "\n")

        # Save statistics
        stats_file = self.output_dir / "organization_stats.json"
        with open(stats_file, 'w') as f:
            json.dump({
                **self.stats,
                'by_class': dict(self.stats['by_class']),
                'samples_per_class': self.samples_per_class,
                'train_ratio': self.train_ratio,
                'val_ratio': self.val_ratio,
                'test_ratio': self.test_ratio
            }, f, indent=2)

        logger.info(f"✓ Statistics saved to: {stats_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Organize flattened FORGE contracts - Step 2 of Approach A"
    )
    parser.add_argument(
        "--flattened-dir",
        type=str,
        default="data/datasets/forge_flattened_all",
        help="Input directory with flattened contracts (from Step 1)"
    )
    parser.add_argument(
        "--forge-dir",
        type=str,
        default="data/datasets/FORGE-Artifacts",
        help="Path to FORGE-Artifacts directory (for audit JSONs)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/datasets/forge_reconstructed",
        help="Output directory for organized dataset"
    )
    parser.add_argument(
        "--samples-per-class",
        type=str,
        default="reentrancy:800,arithmetic:1000,access_control:1000,unchecked_low_level_calls:1000,denial_of_service:500,bad_randomness:300,time_manipulation:300,front_running:300,other:1000",
        help="Samples per class (format: class1:count1,class2:count2,...)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio (default: 0.70)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15)"
    )

    args = parser.parse_args()

    # Parse samples per class
    samples_per_class = {}
    for pair in args.samples_per_class.split(','):
        class_name, count = pair.split(':')
        samples_per_class[class_name.strip()] = int(count.strip())

    flattened_dir = Path(args.flattened_dir)
    forge_dir = Path(args.forge_dir)
    output_dir = Path(args.output_dir)

    if not flattened_dir.exists():
        logger.error(f"Flattened directory not found: {flattened_dir}")
        return 1

    if not forge_dir.exists():
        logger.error(f"FORGE directory not found: {forge_dir}")
        return 1

    organizer = ContractOrganizer(
        flattened_dir=flattened_dir,
        forge_dir=forge_dir,
        output_dir=output_dir,
        samples_per_class=samples_per_class,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )

    organizer.organize_dataset()

    return 0


if __name__ == "__main__":
    random.seed(42)  # Reproducible sampling
    sys.exit(main())
