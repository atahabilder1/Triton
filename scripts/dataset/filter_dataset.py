#!/usr/bin/env python3
"""
Filter Smart Contract Dataset
Removes low-quality contracts that produce empty/sparse PDGs:
- Interface files (no implementation)
- Abstract contracts (no implementation)
- Library files (limited functionality)
- Very small contracts (< 10 lines of actual code)
"""

import os
import re
from pathlib import Path
from typing import List, Tuple
import shutil
from collections import defaultdict


def is_low_quality_contract(file_path: Path) -> Tuple[bool, str]:
    """
    Check if a contract is low quality (interface, abstract, library, or too small)
    Returns (is_low_quality, reason)
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Count non-empty, non-comment lines
        lines = content.split('\n')
        code_lines = 0
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('//') and stripped != '{' and stripped != '}':
                code_lines += 1

        # Check if it's an interface
        if re.search(r'^\s*interface\s+\w+', content, re.MULTILINE):
            return True, "interface"

        # Check if it's abstract with no implementation
        if re.search(r'^\s*abstract\s+contract', content, re.MULTILINE):
            # Check if it has any function implementations
            if not re.search(r'function\s+\w+[^;]*\{', content):
                return True, "abstract_no_impl"

        # Check if it's only a library
        if re.search(r'^\s*library\s+\w+', content, re.MULTILINE):
            # Libraries are OK if they have substantial code
            if code_lines < 20:
                return True, "small_library"

        # Check if contract is too small (likely just a stub/interface)
        if code_lines < 10:
            return True, "too_small"

        # Check if it has no function implementations (only declarations)
        function_decls = len(re.findall(r'function\s+\w+', content))
        function_impls = len(re.findall(r'function\s+\w+[^;]*\{', content))

        if function_decls > 0 and function_impls == 0:
            return True, "no_implementations"

        return False, "ok"

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return True, "error"


def filter_dataset_directory(input_dir: Path, output_dir: Path, dry_run: bool = False):
    """Filter a dataset directory (train/val/test)"""

    print(f"\n{'='*80}")
    print(f"Filtering: {input_dir}")
    print(f"Output:    {output_dir}")
    print(f"{'='*80}\n")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Track statistics
    stats = defaultdict(lambda: {'total': 0, 'kept': 0, 'removed': defaultdict(int)})

    # Process each vulnerability type directory
    for vuln_dir in sorted(input_dir.iterdir()):
        if not vuln_dir.is_dir():
            continue

        vuln_type = vuln_dir.name
        print(f"Processing {vuln_type}...")

        if not dry_run:
            output_vuln_dir = output_dir / vuln_type
            output_vuln_dir.mkdir(parents=True, exist_ok=True)

        # Process each contract file
        for contract_file in vuln_dir.glob("*.sol"):
            stats[vuln_type]['total'] += 1

            is_low_quality, reason = is_low_quality_contract(contract_file)

            if is_low_quality:
                stats[vuln_type]['removed'][reason] += 1
            else:
                stats[vuln_type]['kept'] += 1
                if not dry_run:
                    shutil.copy2(contract_file, output_vuln_dir / contract_file.name)

    # Print statistics
    print(f"\n{'='*80}")
    print(f"FILTERING RESULTS - {input_dir.name}")
    print(f"{'='*80}")
    print(f"{'Vulnerability Type':<25} {'Total':>8} {'Kept':>8} {'Removed':>8} {'% Kept':>8}")
    print(f"{'-'*80}")

    total_all = 0
    kept_all = 0

    for vuln_type in sorted(stats.keys()):
        total = stats[vuln_type]['total']
        kept = stats[vuln_type]['kept']
        removed = total - kept
        pct_kept = (kept / total * 100) if total > 0 else 0

        total_all += total
        kept_all += kept

        print(f"{vuln_type:<25} {total:>8} {kept:>8} {removed:>8} {pct_kept:>7.1f}%")

    print(f"{'-'*80}")
    removed_all = total_all - kept_all
    pct_kept_all = (kept_all / total_all * 100) if total_all > 0 else 0
    print(f"{'TOTAL':<25} {total_all:>8} {kept_all:>8} {removed_all:>8} {pct_kept_all:>7.1f}%")
    print(f"{'='*80}\n")

    # Print removal reasons summary
    print("Removal Reasons:")
    all_reasons = defaultdict(int)
    for vuln_stats in stats.values():
        for reason, count in vuln_stats['removed'].items():
            all_reasons[reason] += count

    for reason, count in sorted(all_reasons.items(), key=lambda x: x[1], reverse=True):
        pct = (count / removed_all * 100) if removed_all > 0 else 0
        print(f"  {reason:<25} {count:>6} ({pct:>5.1f}%)")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Filter smart contract dataset")
    parser.add_argument("--input-dir", required=True, help="Input dataset root directory")
    parser.add_argument("--output-dir", required=True, help="Output filtered dataset directory")
    parser.add_argument("--dry-run", action="store_true", help="Show stats without copying files")

    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if not input_root.exists():
        print(f"Error: Input directory {input_root} does not exist")
        return

    print(f"\n{'='*80}")
    print(f"SMART CONTRACT DATASET FILTERING")
    print(f"{'='*80}")
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print(f"Mode:   {'DRY RUN (no files will be copied)' if args.dry_run else 'LIVE (files will be copied)'}")
    print(f"{'='*80}\n")

    # Filter train/val/test directories
    for subset in ['train', 'val', 'test']:
        input_dir = input_root / subset
        if input_dir.exists():
            output_dir = output_root / subset
            filter_dataset_directory(input_dir, output_dir, args.dry_run)
        else:
            print(f"Warning: {input_dir} not found, skipping")

    if args.dry_run:
        print("\n" + "="*80)
        print("DRY RUN COMPLETE - No files were copied")
        print("Run without --dry-run to perform actual filtering")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("FILTERING COMPLETE!")
        print(f"Filtered dataset saved to: {output_root}")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
