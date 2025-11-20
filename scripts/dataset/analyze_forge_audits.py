#!/usr/bin/env python3
"""
Analyze FORGE Audit Reports
Extract insights from audit JSONs about CWE codes, severity, and vulnerabilities
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ForgeAuditAnalyzer:
    """Analyze FORGE audit reports"""

    def __init__(self, forge_dir: Path):
        self.forge_dir = forge_dir
        self.results_dir = forge_dir / "dataset" / "results"
        self.contracts_dir = forge_dir / "dataset" / "contracts"

        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")

        self.audit_files = list(self.results_dir.glob("*.json"))
        print(f"Found {len(self.audit_files)} audit reports\n")

    def analyze_cwe_distribution(self) -> Counter:
        """Analyze CWE code distribution"""
        print("="*80)
        print("CWE CODE DISTRIBUTION")
        print("="*80)

        cwe_counts = Counter()

        for audit_file in self.audit_files:
            try:
                with open(audit_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for finding in data.get("findings", []):
                    category = finding.get("category", {})
                    # CWE codes are in category["1"]
                    cwes = category.get("1", [])
                    for cwe in cwes:
                        cwe_counts[cwe] += 1
            except Exception as e:
                print(f"Error reading {audit_file.name}: {e}")

        print(f"\nTotal unique CWE codes: {len(cwe_counts)}")
        print(f"Total CWE occurrences: {sum(cwe_counts.values())}\n")

        print("Top 30 Most Common CWEs:")
        print("-"*80)
        print(f"{'CWE Code':<20} {'Count':>10} {'Percentage':>12}")
        print("-"*80)

        total = sum(cwe_counts.values())
        for cwe, count in cwe_counts.most_common(30):
            pct = (count / total * 100) if total > 0 else 0
            print(f"{cwe:<20} {count:>10} {pct:>11.2f}%")

        print("="*80 + "\n")
        return cwe_counts

    def analyze_severity_distribution(self) -> Counter:
        """Analyze severity distribution"""
        print("="*80)
        print("SEVERITY DISTRIBUTION")
        print("="*80)

        severity_counts = Counter()
        cwe_severity_map = {}  # Map CWE to its severity levels

        for audit_file in self.audit_files:
            try:
                with open(audit_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for finding in data.get("findings", []):
                    severity = finding.get("severity", "unknown").lower()
                    severity_counts[severity] += 1

                    # Map CWE codes to severity
                    cwes = finding.get("category", {}).get("1", [])
                    for cwe in cwes:
                        if cwe not in cwe_severity_map:
                            cwe_severity_map[cwe] = Counter()
                        cwe_severity_map[cwe][severity] += 1

            except Exception as e:
                print(f"Error reading {audit_file.name}: {e}")

        print(f"\nTotal findings: {sum(severity_counts.values())}\n")

        print("Severity Distribution:")
        print("-"*80)
        print(f"{'Severity':<20} {'Count':>10} {'Percentage':>12}")
        print("-"*80)

        total = sum(severity_counts.values())
        for severity, count in severity_counts.most_common():
            pct = (count / total * 100) if total > 0 else 0
            print(f"{severity:<20} {count:>10} {pct:>11.2f}%")

        print("="*80 + "\n")
        return severity_counts

    def analyze_multi_label_contracts(self) -> List[Dict]:
        """Find contracts with multiple vulnerabilities"""
        print("="*80)
        print("MULTI-VULNERABILITY CONTRACTS")
        print("="*80)

        multi_vuln = []
        vuln_count_dist = Counter()

        for audit_file in self.audit_files:
            try:
                with open(audit_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Collect all unique CWEs for this contract
                all_cwes = set()
                for finding in data.get("findings", []):
                    cwes = finding.get("category", {}).get("1", [])
                    all_cwes.update(cwes)

                num_cwes = len(all_cwes)
                vuln_count_dist[num_cwes] += 1

                if num_cwes > 1:
                    project_name = audit_file.stem.replace('.pdf', '')
                    multi_vuln.append({
                        'project': project_name,
                        'cwes': sorted(list(all_cwes)),
                        'count': num_cwes
                    })

            except Exception as e:
                print(f"Error reading {audit_file.name}: {e}")

        print(f"\nTotal contracts analyzed: {len(self.audit_files)}")
        print(f"Contracts with multiple CWEs: {len(multi_vuln)}\n")

        print("Distribution of CWE count per contract:")
        print("-"*80)
        print(f"{'# of CWEs':<15} {'# of Contracts':>15} {'Percentage':>12}")
        print("-"*80)

        total = sum(vuln_count_dist.values())
        for count in sorted(vuln_count_dist.keys()):
            num_contracts = vuln_count_dist[count]
            pct = (num_contracts / total * 100) if total > 0 else 0
            print(f"{count:<15} {num_contracts:>15} {pct:>11.2f}%")

        print("\n" + "="*80)

        # Show examples
        if multi_vuln:
            print("\nExample contracts with multiple CWEs (top 10):")
            print("-"*80)
            sorted_multi = sorted(multi_vuln, key=lambda x: x['count'], reverse=True)[:10]
            for item in sorted_multi:
                print(f"\n{item['project']} ({item['count']} CWEs):")
                for cwe in item['cwes'][:5]:  # Show first 5 CWEs
                    print(f"  - {cwe}")
                if len(item['cwes']) > 5:
                    print(f"  ... and {len(item['cwes']) - 5} more")

        print("\n" + "="*80 + "\n")
        return multi_vuln

    def analyze_cwe_cooccurrence(self, top_n: int = 10) -> Dict:
        """Analyze which CWEs appear together"""
        print("="*80)
        print("CWE CO-OCCURRENCE ANALYSIS")
        print("="*80)

        # Track which CWEs appear together
        cooccurrence = defaultdict(Counter)

        for audit_file in self.audit_files:
            try:
                with open(audit_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Collect all CWEs in this contract
                all_cwes = set()
                for finding in data.get("findings", []):
                    cwes = finding.get("category", {}).get("1", [])
                    all_cwes.update(cwes)

                # Record co-occurrences
                all_cwes_list = sorted(list(all_cwes))
                for i, cwe1 in enumerate(all_cwes_list):
                    for cwe2 in all_cwes_list[i+1:]:
                        cooccurrence[cwe1][cwe2] += 1
                        cooccurrence[cwe2][cwe1] += 1

            except Exception as e:
                print(f"Error reading {audit_file.name}: {e}")

        # Find top co-occurring pairs
        pairs = []
        seen = set()
        for cwe1, partners in cooccurrence.items():
            for cwe2, count in partners.items():
                pair = tuple(sorted([cwe1, cwe2]))
                if pair not in seen:
                    pairs.append((pair, count))
                    seen.add(pair)

        pairs.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop {top_n} CWE pairs that appear together:")
        print("-"*80)
        print(f"{'CWE 1':<25} {'CWE 2':<25} {'Co-occurrence':>15}")
        print("-"*80)

        for (cwe1, cwe2), count in pairs[:top_n]:
            print(f"{cwe1:<25} {cwe2:<25} {count:>15}")

        print("="*80 + "\n")
        return dict(cooccurrence)

    def analyze_compiler_versions(self) -> Counter:
        """Analyze Solidity compiler version distribution"""
        print("="*80)
        print("COMPILER VERSION DISTRIBUTION")
        print("="*80)

        version_counts = Counter()
        version_vulns = defaultdict(list)

        for audit_file in self.audit_files:
            try:
                with open(audit_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                versions = data.get('project_info', {}).get('compiler_version', [])
                num_findings = len(data.get('findings', []))

                for version in versions:
                    # Extract major.minor version
                    try:
                        # Format: v0.8.4+commit.c7e474f2
                        clean_version = version.split('+')[0].strip('v')
                        major_minor = '.'.join(clean_version.split('.')[:2])
                        version_counts[major_minor] += 1
                        version_vulns[major_minor].append(num_findings)
                    except:
                        version_counts['unknown'] += 1

            except Exception as e:
                print(f"Error reading {audit_file.name}: {e}")

        print(f"\nTotal compiler versions found: {len(version_counts)}\n")

        print("Top 20 Compiler Versions:")
        print("-"*80)
        print(f"{'Version':<15} {'Count':>10} {'Avg Vulns':>12}")
        print("-"*80)

        for version, count in version_counts.most_common(20):
            avg_vulns = sum(version_vulns[version]) / len(version_vulns[version]) if version_vulns[version] else 0
            print(f"{version:<15} {count:>10} {avg_vulns:>12.2f}")

        print("="*80 + "\n")
        return version_counts

    def analyze_blockchain_distribution(self) -> Counter:
        """Analyze blockchain chain distribution"""
        print("="*80)
        print("BLOCKCHAIN DISTRIBUTION")
        print("="*80)

        chain_counts = Counter()
        chain_vulns = defaultdict(Counter)

        for audit_file in self.audit_files:
            try:
                with open(audit_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                chain = data.get('project_info', {}).get('chain', 'unknown')
                chain_counts[chain] += 1

                # Count vulnerabilities per chain
                for finding in data.get('findings', []):
                    cwes = finding.get("category", {}).get("1", [])
                    for cwe in cwes:
                        chain_vulns[chain][cwe] += 1

            except Exception as e:
                print(f"Error reading {audit_file.name}: {e}")

        print(f"\nBlockchains represented: {len(chain_counts)}\n")

        print("Chain Distribution:")
        print("-"*80)
        print(f"{'Chain':<20} {'Contracts':>12} {'Percentage':>12}")
        print("-"*80)

        total = sum(chain_counts.values())
        for chain, count in chain_counts.most_common():
            pct = (count / total * 100) if total > 0 else 0
            print(f"{chain:<20} {count:>12} {pct:>11.2f}%")

        print("="*80 + "\n")
        return chain_counts

    def generate_full_report(self, output_file: Path = None):
        """Generate comprehensive report"""
        print("\n" + "="*80)
        print("FORGE AUDIT ANALYSIS - COMPREHENSIVE REPORT")
        print("="*80 + "\n")

        # Run all analyses
        cwe_dist = self.analyze_cwe_distribution()
        severity_dist = self.analyze_severity_distribution()
        multi_vuln = self.analyze_multi_label_contracts()
        cooccurrence = self.analyze_cwe_cooccurrence(top_n=15)
        compiler_versions = self.analyze_compiler_versions()
        chain_dist = self.analyze_blockchain_distribution()

        # Save to JSON if requested
        if output_file:
            report = {
                'total_audits': len(self.audit_files),
                'cwe_distribution': dict(cwe_dist),
                'severity_distribution': dict(severity_dist),
                'multi_vulnerability_contracts': len(multi_vuln),
                'compiler_versions': dict(compiler_versions),
                'blockchain_distribution': dict(chain_dist),
                'top_cwes': dict(cwe_dist.most_common(30)),
            }

            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)

            print(f"\n✓ Full report saved to: {output_file}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze FORGE audit reports for insights"
    )
    parser.add_argument(
        "--forge-dir",
        type=str,
        default="data/datasets/FORGE-Artifacts",
        help="Path to FORGE-Artifacts directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for full report (optional)"
    )

    args = parser.parse_args()

    # Initialize analyzer
    forge_dir = Path(args.forge_dir)
    if not forge_dir.exists():
        print(f"Error: FORGE directory not found: {forge_dir}")
        return 1

    analyzer = ForgeAuditAnalyzer(forge_dir)

    # Generate report
    output_file = Path(args.output) if args.output else Path("forge_audit_analysis.json")
    analyzer.generate_full_report(output_file)

    print("="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nKey Insights:")
    print("1. Check CWE distribution to see which vulnerabilities are most common")
    print("2. Review multi-label contracts - many have multiple vulnerabilities")
    print("3. Consider multi-label classification instead of single-label")
    print("4. Use CWE co-occurrence to understand vulnerability patterns")
    print("5. Compiler version analysis shows which versions need attention")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
