#!/usr/bin/env python3
"""
Solidity Contract Flattener
Combines contract with all dependencies into single file
Eliminates import/dependency issues for training
"""

import sys
import subprocess
import os
from pathlib import Path
from typing import Optional, Tuple
import logging
import argparse
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SolidityFlattener:
    """Flatten Solidity contracts using various tools"""

    def __init__(self):
        self.available_tools = self._check_available_tools()
        logger.info(f"Available flattening tools: {', '.join(self.available_tools) if self.available_tools else 'None'}")

    def _check_available_tools(self) -> list:
        """Check which flattening tools are installed"""
        tools = []

        # Check for hardhat
        if shutil.which('npx'):
            try:
                result = subprocess.run(
                    ['npx', 'hardhat', '--version'],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    tools.append('hardhat')
            except:
                pass

        # Check for truffle
        if shutil.which('truffle'):
            tools.append('truffle')

        # Check for forge (foundry)
        if shutil.which('forge'):
            tools.append('forge')

        # Check for sol-merger (Python package)
        try:
            import sol_merger
            tools.append('sol-merger')
        except ImportError:
            pass

        return tools

    def flatten_with_forge(self, contract_path: str, output_path: Optional[str] = None) -> Tuple[bool, str]:
        """Flatten using Foundry's forge"""
        try:
            cmd = ['forge', 'flatten', contract_path]

            if output_path:
                cmd.extend(['-o', output_path])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                if output_path:
                    return True, f"Flattened to {output_path}"
                else:
                    return True, result.stdout
            else:
                return False, result.stderr

        except subprocess.TimeoutExpired:
            return False, "Forge flatten timed out"
        except Exception as e:
            return False, f"Forge error: {e}"

    def flatten_with_hardhat(self, contract_path: str, output_path: Optional[str] = None) -> Tuple[bool, str]:
        """Flatten using Hardhat"""
        try:
            cmd = ['npx', 'hardhat', 'flatten', contract_path]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                flattened_code = result.stdout

                if output_path:
                    with open(output_path, 'w') as f:
                        f.write(flattened_code)
                    return True, f"Flattened to {output_path}"
                else:
                    return True, flattened_code
            else:
                return False, result.stderr

        except subprocess.TimeoutExpired:
            return False, "Hardhat flatten timed out"
        except Exception as e:
            return False, f"Hardhat error: {e}"

    def flatten_with_truffle(self, contract_path: str, output_path: Optional[str] = None) -> Tuple[bool, str]:
        """Flatten using Truffle"""
        try:
            cmd = ['truffle-flattener', contract_path]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                flattened_code = result.stdout

                if output_path:
                    with open(output_path, 'w') as f:
                        f.write(flattened_code)
                    return True, f"Flattened to {output_path}"
                else:
                    return True, flattened_code
            else:
                return False, result.stderr

        except subprocess.TimeoutExpired:
            return False, "Truffle flatten timed out"
        except Exception as e:
            return False, f"Truffle error: {e}"

    def flatten_with_sol_merger(self, contract_path: str, output_path: Optional[str] = None) -> Tuple[bool, str]:
        """Flatten using sol-merger (Python package)"""
        try:
            from sol_merger.merger import merge_files

            # Get directory for import resolution
            contract_dir = os.path.dirname(os.path.abspath(contract_path))

            # Merge
            merged_code = merge_files(contract_path, root_path=contract_dir)

            if output_path:
                with open(output_path, 'w') as f:
                    f.write(merged_code)
                return True, f"Flattened to {output_path}"
            else:
                return True, merged_code

        except Exception as e:
            return False, f"sol-merger error: {e}"

    def flatten_simple(self, contract_path: str, output_path: Optional[str] = None) -> Tuple[bool, str]:
        """
        Simple flattening by recursively resolving imports
        Fallback method when no tools are available
        """
        try:
            contract_dir = Path(contract_path).parent
            processed_files = set()

            def resolve_imports(file_path: Path) -> str:
                """Recursively resolve imports"""
                if file_path in processed_files:
                    return ""

                processed_files.add(file_path)

                with open(file_path, 'r') as f:
                    content = f.read()

                # Find imports
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
                    # Extract import path
                    import_match = import_line.split('"')
                    if len(import_match) >= 2:
                        import_path = import_match[1]

                        # Resolve relative path
                        if import_path.startswith('./') or import_path.startswith('../'):
                            full_import_path = (file_path.parent / import_path).resolve()

                            if full_import_path.exists():
                                imported_content = resolve_imports(full_import_path)
                                if imported_content:
                                    imported_code.append(f"\n// ========== {full_import_path.name} ==========\n")
                                    imported_code.append(imported_content)

                # Combine
                result = '\n'.join(imported_code) + '\n' + '\n'.join(other_lines)
                return result

            # Start resolution
            flattened = resolve_imports(Path(contract_path))

            if output_path:
                with open(output_path, 'w') as f:
                    f.write(flattened)
                return True, f"Flattened to {output_path}"
            else:
                return True, flattened

        except Exception as e:
            return False, f"Simple flatten error: {e}"

    def flatten(self, contract_path: str, output_path: Optional[str] = None, tool: Optional[str] = None) -> Tuple[bool, str]:
        """
        Flatten a contract using available tools

        Args:
            contract_path: Path to contract file
            output_path: Optional output file path
            tool: Preferred tool ('forge', 'hardhat', 'truffle', 'sol-merger', 'simple')

        Returns:
            (success, result_or_error)
        """
        if not os.path.exists(contract_path):
            return False, f"File not found: {contract_path}"

        # Try preferred tool first
        if tool:
            if tool == 'forge' and 'forge' in self.available_tools:
                return self.flatten_with_forge(contract_path, output_path)
            elif tool == 'hardhat' and 'hardhat' in self.available_tools:
                return self.flatten_with_hardhat(contract_path, output_path)
            elif tool == 'truffle' and 'truffle' in self.available_tools:
                return self.flatten_with_truffle(contract_path, output_path)
            elif tool == 'sol-merger' and 'sol-merger' in self.available_tools:
                return self.flatten_with_sol_merger(contract_path, output_path)
            elif tool == 'simple':
                return self.flatten_simple(contract_path, output_path)

        # Try available tools in order of preference
        if 'forge' in self.available_tools:
            success, result = self.flatten_with_forge(contract_path, output_path)
            if success:
                return success, result

        if 'hardhat' in self.available_tools:
            success, result = self.flatten_with_hardhat(contract_path, output_path)
            if success:
                return success, result

        if 'truffle' in self.available_tools:
            success, result = self.flatten_with_truffle(contract_path, output_path)
            if success:
                return success, result

        if 'sol-merger' in self.available_tools:
            success, result = self.flatten_with_sol_merger(contract_path, output_path)
            if success:
                return success, result

        # Fallback to simple flattening
        logger.warning("No external tools available, using simple flattening")
        return self.flatten_simple(contract_path, output_path)


def flatten_dataset(input_dir: str, output_dir: str, tool: Optional[str] = None):
    """Flatten all contracts in a dataset"""
    flattener = SolidityFlattener()

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Flattening contracts from: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)

    sol_files = list(input_path.rglob("*.sol"))
    logger.info(f"Found {len(sol_files)} Solidity files\n")

    success_count = 0
    fail_count = 0

    for idx, sol_file in enumerate(sol_files, 1):
        try:
            # Preserve directory structure
            rel_path = sol_file.relative_to(input_path)
            output_file = output_path / rel_path

            # Create output directory
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Flatten
            success, result = flattener.flatten(str(sol_file), str(output_file), tool)

            if success:
                logger.info(f"[{idx}/{len(sol_files)}] ✓ {sol_file.name} -> {output_file.name}")
                success_count += 1
            else:
                logger.warning(f"[{idx}/{len(sol_files)}] ✗ {sol_file.name} - {result}")
                fail_count += 1

                # Copy original if flattening fails
                shutil.copy2(sol_file, output_file)
                logger.info(f"    Copied original instead")

        except Exception as e:
            logger.error(f"[{idx}/{len(sol_files)}] ERROR: {sol_file.name} - {e}")
            fail_count += 1

        # Progress update
        if idx % 50 == 0:
            logger.info(f"\nProgress: {idx}/{len(sol_files)} ({idx/len(sol_files)*100:.1f}%)")
            logger.info(f"Success: {success_count}, Failed: {fail_count}\n")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("FLATTENING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total files: {len(sol_files)}")
    logger.info(f"Successfully flattened: {success_count} ({success_count/len(sol_files)*100:.1f}%)")
    logger.info(f"Failed: {fail_count} ({fail_count/len(sol_files)*100:.1f}%)")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Flatten Solidity Contracts")
    parser.add_argument("input", help="Input contract file or directory")
    parser.add_argument("--output", "-o", help="Output file or directory")
    parser.add_argument("--tool", choices=['forge', 'hardhat', 'truffle', 'sol-merger', 'simple'],
                       help="Preferred flattening tool")
    parser.add_argument("--batch", action='store_true',
                       help="Process entire directory (requires --output)")

    args = parser.parse_args()

    if args.batch:
        if not args.output:
            parser.error("--batch requires --output directory")

        flatten_dataset(args.input, args.output, args.tool)
    else:
        # Single file
        flattener = SolidityFlattener()
        success, result = flattener.flatten(args.input, args.output, args.tool)

        if success:
            if args.output:
                logger.info(f"✓ {result}")
            else:
                print(result)  # Print flattened code to stdout
        else:
            logger.error(f"✗ Failed: {result}")
            sys.exit(1)


if __name__ == "__main__":
    main()
