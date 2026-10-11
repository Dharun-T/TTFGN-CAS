#!/usr/bin/env bash
# T-TFGN Master Execution Script
set -e

echo "=========================================================="
echo " Starting T-TFGN Clinical Actionability Pipeline          "
echo "=========================================================="

# Check Python version
python3 -c "import sys; assert sys.version_info >= (3, 9), 'Python 3.9+ required'"

# Run full research pipeline
python3 scripts/run_all.py

# Run all unit tests
python3 -m unittest discover tests -v

echo "=========================================================="
echo " All T-TFGN experiments and tests completed successfully! "
echo "=========================================================="
