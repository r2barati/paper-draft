#!/bin/bash
# Run this script after MacTeX install finishes to copy remaining code
# Usage: bash copy_remaining_code.sh

set -e
BASE="/Users/sanamimani/Desktop/NetworkInvLog-BenchmarkTest"
DEST="$BASE/paper-draft-fresh/code"

echo "Copying scripts..."
rsync -av --exclude='__pycache__' --exclude='*.pyc' --exclude='_legacy' \
    "$BASE/scripts/" "$DEST/scripts/"

echo "Copying tests..."
rsync -av --exclude='__pycache__' --exclude='*.pyc' \
    "$BASE/tests/" "$DEST/tests/"

echo "Copying data/results (CSV files only)..."
mkdir -p "$DEST/data/results"
find "$BASE/data/results" -name "*.csv" -exec cp {} "$DEST/data/results/" \;

echo "Copying environment charts..."
mkdir -p "$DEST/../figures"
cp "$BASE/docs/charts_environment/E1_demand_profile_gallery.png" "$DEST/../figures/" 2>/dev/null || true
cp "$BASE/docs/charts_environment/E4_reward_decomposition.png" "$DEST/../figures/" 2>/dev/null || true

echo "Done! Now commit and push:"
echo "  cd $BASE/paper-draft-fresh"
echo "  git add -A && git commit -m 'Add remaining scripts, tests, and data' && git push"
