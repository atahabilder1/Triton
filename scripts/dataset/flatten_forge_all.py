#!/usr/bin/env python3
"""
Flatten All FORGE Projects
Step 1 of Approach A: Flatten first, organize later

Process:
1. Read each project folder in FORGE-Artifacts/dataset/contracts/
2. Find the main contract file (from audit JSON)
3. Flatten using forge/truffle/simple method
4. Save to output directory with project name

Output: Single folder with all flattened contracts
"""

import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging
import argparse
from collections import defaultdict
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ForgeFlattener:
    """Flatten all FORGE projects"""

    def __init__(self, forge_dir: Path, output_dir: Path, tool: str = 'simple'):
        self.forge_dir = forge_dir
        self.contracts_dir = forge_dir / "dataset" / "contracts"
        self.results_dir = forge_dir / "dataset" / "results"
        self.output_dir = output_dir
        self.tool = tool

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            'total_projects': 0,
            'flattened': 0,
            'copied_original': 0,
            'failed': 0,
            'skipped_no_sol': 0,
            'skipped_no_audit': 0
        }

    def find_audit_json(self, project_name: str) -> Optional[Path]:
        """Find audit JSON for a project"""
        # Try exact match
        audit_file = self.results_dir / f"{project_name}.pdf.json"
        if audit_file.exists():
            return audit_file

        # Try variations
        for json_file in self.results_dir.glob("*.json"):
            if project_name.lower() in json_file.stem.lower():
                return json_file

        return None

    def find_main_contract(self, project_path: Path, audit_json: Optional[Path]) -> Optional[Path]:
        """Find the main contract file in project folder"""
        # Strategy 1: Use audit JSON project_path
        if audit_json and audit_json.exists():
            try:
                with open(audit_json, 'r') as f:
                    data = json.load(f)
                    project_paths = data.get('project_info', {}).get('project_path', {})

                    if project_paths:
                        # Get first contract name
                        contract_name = list(project_paths.keys())[0]
                        # Try to find file
                        for sol_file in project_path.rglob(f"{contract_name}.sol"):
                            return sol_file
            except:
                pass

        # Strategy 2: Find .sol file matching project folder name
        project_name = project_path.name
        for sol_file in project_path.rglob("*.sol"):
            if project_name.lower() in sol_file.stem.lower():
                return sol_file

        # Strategy 3: Return first .sol file found
        sol_files = list(project_path.rglob("*.sol"))
        if sol_files:
            # Prefer files in root of project, not subdirs
            root_files = [f for f in sol_files if f.parent == project_path]
            if root_files:
                return root_files[0]
            return sol_files[0]

        return None

    def flatten_with_forge(self, contract_path: Path, output_path: Path) -> Tuple[bool, str]:
        """Flatten using Foundry's forge"""
        try:
            # Change to contract directory for imports to work
            working_dir = contract_path.parent

            result = subprocess.run(
                ['forge', 'flatten', str(contract_path)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=working_dir
            )

            if result.returncode == 0 and result.stdout:
                # Save flattened output
                with open(output_path, 'w') as f:
                    f.write(result.stdout)
                return True, "Success"
            else:
                return False, result.stderr or "No output"

        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except FileNotFoundError:
            return False, "Forge not installed"
        except Exception as e:
            return False, str(e)

    def flatten_with_truffle(self, contract_path: Path, output_path: Path) -> Tuple[bool, str]:
        """Flatten using truffle-flattener"""
        try:
            working_dir = contract_path.parent

            result = subprocess.run(
                ['truffle-flattener', str(contract_path)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=working_dir
            )

            if result.returncode == 0 and result.stdout:
                with open(output_path, 'w') as f:
                    f.write(result.stdout)
                return True, "Success"
            else:
                return False, result.stderr or "No output"

        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except FileNotFoundError:
            return False, "Truffle-flattener not installed"
        except Exception as e:
            return False, str(e)

    def flatten_simple(self, contract_path: Path, output_path: Path) -> Tuple[bool, str]:
        """Simple flattening by resolving local imports"""
        try:
            processed_files = set()

            def resolve_imports(file_path: Path) -> str:
                """Recursively resolve imports"""
                if file_path in processed_files:
                    return ""

                processed_files.add(file_path)

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                except:
                    return ""

                # Find and resolve imports
                import_lines = []
                other_lines = []

                for line in content.split('\n'):
                    if line.strip().startswith('import'):
                        import_lines.append(line)
                    else:
                        other_lines.append(line)

                # Resolve each import
                imported_code = []
                for import_line in import_lines:
                    # Extract import path: import "./SafeMath.sol"; or import "path/to/File.sol";
                    parts = import_line.split('"')
                    if len(parts) >= 2:
                        import_path = parts[1]

                        # Only resolve relative imports (./  or ../)
                        if import_path.startswith('./') or import_path.startswith('../'):
                            full_import_path = (file_path.parent / import_path).resolve()

                            if full_import_path.exists():
                                imported_content = resolve_imports(full_import_path)
                                if imported_content:
                                    imported_code.append(f"\n// ========== From {full_import_path.name} ==========\n")
                                    imported_code.append(imported_content)

                # Combine
                result = '\n'.join(imported_code) + '\n' + '\n'.join(other_lines)
                return result

            # Start resolution
            flattened = resolve_imports(contract_path)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(flattened)

            return True, "Success"

        except Exception as e:
            return False, str(e)

    def flatten_contract(self, contract_path: Path, output_path: Path) -> Tuple[bool, str]:
        """Flatten a contract using specified tool"""

        # Try specified tool first
        if self.tool == 'forge':
            success, msg = self.flatten_with_forge(contract_path, output_path)
            if success:
                return True, msg
        elif self.tool == 'truffle':
            success, msg = self.flatten_with_truffle(contract_path, output_path)
            if success:
                return True, msg
        elif self.tool == 'simple':
            success, msg = self.flatten_simple(contract_path, output_path)
            if success:
                return True, msg

        # Fallback: try other tools
        if self.tool != 'forge':
            success, msg = self.flatten_with_forge(contract_path, output_path)
            if success:
                return True, "forge (fallback)"

        if self.tool != 'truffle':
            success, msg = self.flatten_with_truffle(contract_path, output_path)
            if success:
                return True, "truffle (fallback)"

        if self.tool != 'simple':
            success, msg = self.flatten_simple(contract_path, output_path)
            if success:
                return True, "simple (fallback)"

        # All failed - copy original
        try:
            shutil.copy2(contract_path, output_path)
            return False, "Copied original (flattening failed)"
        except Exception as e:
            return False, f"Copy failed: {e}"

    def process_project(self, project_path: Path) -> Tuple[bool, str]:
        """Process a single project"""
        project_name = project_path.name

        # Find audit JSON
        audit_json = self.find_audit_json(project_name)
        if not audit_json:
            return False, "No audit JSON"

        # Find main contract
        main_contract = self.find_main_contract(project_path, audit_json)
        if not main_contract:
            return False, "No .sol file found"

        # Output filename: projectname_contractname.sol
        contract_name = main_contract.stem
        output_filename = f"{project_name}_{contract_name}.sol"
        output_path = self.output_dir / output_filename

        # Skip if already processed
        if output_path.exists():
            return True, "Already exists (skipped)"

        # Flatten
        success, msg = self.flatten_contract(main_contract, output_path)

        return success, msg

    def process_all(self, max_projects: Optional[int] = None):
        """Process all projects in FORGE"""
        logger.info("="*80)
        logger.info("FORGE PROJECT FLATTENING - Step 1 of Approach A")
        logger.info("="*80)
        logger.info(f"Source: {self.contracts_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Tool: {self.tool}")
        logger.info("="*80 + "\n")

        # Get all project folders
        project_folders = sorted([p for p in self.contracts_dir.iterdir() if p.is_dir()])

        if max_projects:
            project_folders = project_folders[:max_projects]
            logger.info(f"⚠️  Limited to first {max_projects} projects for testing\n")

        total = len(project_folders)
        logger.info(f"Found {total} project folders\n")

        start_time = time.time()

        for idx, project_path in enumerate(project_folders, 1):
            self.stats['total_projects'] += 1
            project_name = project_path.name

            try:
                success, msg = self.process_project(project_path)

                if success:
                    self.stats['flattened'] += 1
                    logger.info(f"[{idx}/{total}] ✓ {project_name[:50]:<50} | {msg}")
                elif "Copied original" in msg:
                    self.stats['copied_original'] += 1
                    logger.warning(f"[{idx}/{total}] ⚠ {project_name[:50]:<50} | {msg}")
                elif "No audit" in msg:
                    self.stats['skipped_no_audit'] += 1
                    logger.debug(f"[{idx}/{total}] ⊘ {project_name[:50]:<50} | {msg}")
                elif "No .sol" in msg:
                    self.stats['skipped_no_sol'] += 1
                    logger.debug(f"[{idx}/{total}] ⊘ {project_name[:50]:<50} | {msg}")
                else:
                    self.stats['failed'] += 1
                    logger.error(f"[{idx}/{total}] ✗ {project_name[:50]:<50} | {msg}")

            except Exception as e:
                self.stats['failed'] += 1
                logger.error(f"[{idx}/{total}] ✗ {project_name[:50]:<50} | Exception: {e}")

            # Progress report every 100 projects
            if idx % 100 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0
                remaining = (total - idx) / rate if rate > 0 else 0

                logger.info(f"\n{'='*80}")
                logger.info(f"Progress: {idx}/{total} ({idx/total*100:.1f}%)")
                logger.info(f"Speed: {rate:.1f} projects/sec")
                logger.info(f"ETA: {int(remaining//60)}m {int(remaining%60)}s")
                logger.info(f"Success: {self.stats['flattened']}, Copied: {self.stats['copied_original']}, Failed: {self.stats['failed']}")
                logger.info(f"{'='*80}\n")

        # Final summary
        elapsed = time.time() - start_time

        logger.info("\n" + "="*80)
        logger.info("FLATTENING COMPLETE")
        logger.info("="*80)
        logger.info(f"Total projects processed: {self.stats['total_projects']}")
        logger.info(f"✓ Successfully flattened: {self.stats['flattened']} ({self.stats['flattened']/total*100:.1f}%)")
        logger.info(f"⚠ Copied original: {self.stats['copied_original']} ({self.stats['copied_original']/total*100:.1f}%)")
        logger.info(f"⊘ Skipped (no audit): {self.stats['skipped_no_audit']}")
        logger.info(f"⊘ Skipped (no .sol): {self.stats['skipped_no_sol']}")
        logger.info(f"✗ Failed: {self.stats['failed']} ({self.stats['failed']/total*100:.1f}%)")
        logger.info(f"\nTotal time: {int(elapsed//60)}m {int(elapsed%60)}s")
        logger.info(f"Average: {total/elapsed:.1f} projects/sec")
        logger.info(f"\nOutput directory: {self.output_dir}")
        logger.info(f"Total files created: {len(list(self.output_dir.glob('*.sol')))}")
        logger.info("="*80 + "\n")

        # Save statistics
        stats_file = self.output_dir / "flattening_stats.json"
        with open(stats_file, 'w') as f:
            json.dump({
                **self.stats,
                'total_time_seconds': elapsed,
                'tool_used': self.tool
            }, f, indent=2)

        logger.info(f"✓ Statistics saved to: {stats_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Flatten all FORGE projects - Step 1 of Approach A"
    )
    parser.add_argument(
        "--forge-dir",
        type=str,
        default="data/datasets/FORGE-Artifacts",
        help="Path to FORGE-Artifacts directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/datasets/forge_flattened_all",
        help="Output directory for flattened contracts"
    )
    parser.add_argument(
        "--tool",
        choices=['forge', 'truffle', 'simple'],
        default='simple',
        help="Flattening tool to use (default: simple)"
    )
    parser.add_argument(
        "--max-projects",
        type=int,
        default=None,
        help="Maximum number of projects to process (for testing)"
    )

    args = parser.parse_args()

    forge_dir = Path(args.forge_dir)
    output_dir = Path(args.output_dir)

    if not forge_dir.exists():
        logger.error(f"FORGE directory not found: {forge_dir}")
        return 1

    flattener = ForgeFlattener(
        forge_dir=forge_dir,
        output_dir=output_dir,
        tool=args.tool
    )

    flattener.process_all(max_projects=args.max_projects)

    return 0


if __name__ == "__main__":
    sys.exit(main())
