#!/bin/bash
################################################################################
# Install ALL Solidity Compiler Versions
# This maximizes PDG extraction success rate
################################################################################

cd /home/anik/code/Triton
source triton_env/bin/activate

echo "Installing comprehensive Solidity compiler versions..."
echo "This will take 5-10 minutes..."
echo ""

# Array of all important versions
versions=(
  0.4.11 0.4.18 0.4.19 0.4.21 0.4.22 0.4.23 0.4.25
  0.5.0 0.5.1 0.5.2 0.5.3 0.5.4 0.5.5 0.5.6 0.5.7 0.5.8 0.5.9 0.5.10 0.5.11 0.5.12 0.5.13 0.5.14 0.5.15
  0.6.0 0.6.1 0.6.2 0.6.3 0.6.4 0.6.5 0.6.6 0.6.7 0.6.8 0.6.9 0.6.10
  0.7.0 0.7.1 0.7.2 0.7.3 0.7.4 0.7.5
  0.8.1 0.8.2 0.8.3 0.8.5 0.8.6 0.8.7 0.8.8 0.8.10 0.8.11 0.8.12 0.8.13 0.8.14 0.8.15 0.8.16 0.8.18 0.8.19 0.8.20 0.8.21 0.8.22 0.8.23 0.8.24 0.8.25 0.8.26 0.8.27 0.8.28
)

installed=0
failed=0

for v in "${versions[@]}"; do
  echo -n "Installing solc $v... "
  if solc-select install "$v" >/dev/null 2>&1; then
    echo "✓"
    ((installed++))
  else
    echo "✗ (may not exist)"
    ((failed++))
  fi
done

echo ""
echo "================================================================================"
echo "Installation complete!"
echo "================================================================================"
echo "Successfully installed: $installed versions"
echo "Failed/skipped: $failed versions"
echo ""
echo "Total available versions:"
solc-select versions | wc -l
echo ""
