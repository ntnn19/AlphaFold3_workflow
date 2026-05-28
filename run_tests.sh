#!/usr/bin/env bash
# run_tests.sh — run the full AF3 workflow dry-run test suite locally.
#
# Usage:
#   bash run_tests.sh [--verbose]
#
# Requirements:
#   - Snakemake >= 8.0 installed and on PATH (activate the venv first)
#   - Run from the repository root
#
# What it does:
#   1. Lints the workflow against the config schema (advisory warnings are
#      printed but do not fail the suite; only parse errors fail)
#   2. Runs --dry-run for all 18 test cases (4 mode tests + 14 entry-point
#      combination tests covering all 15 valid combinations of the 4 entry
#      points)
#
# No GPU, container, or real data is required — all tests use bundled fixtures
# under .test/.

set -uo pipefail

SNAKEFILE="workflow/Snakefile"
VERBOSE=false
LOG_FILE="run_tests.log"

for arg in "$@"; do
  case "$arg" in
    --verbose|-v) VERBOSE=true ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────

run_snakemake() {
  local config_dir="$1"
  shift
  local cmd=(snakemake
    --snakefile "$SNAKEFILE"
    --configfile "$config_dir/config.yaml"
    --directory  "$config_dir"
    "$@"
  )
  echo "CMD: ${cmd[*]}" >> "$LOG_FILE"
  if $VERBOSE; then
    "${cmd[@]}"
  else
    "${cmd[@]}" --quiet all 2>/dev/null
  fi
}

# ── Initialise log ───────────────────────────────────────────────────────────

: > "$LOG_FILE"
echo "# run_tests.sh — $(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# ── Step 1: Lint ─────────────────────────────────────────────────────────────
# `snakemake --lint` exits 1 when it finds style warnings (not just parse
# errors), so we capture output and only fail on actual WorkflowError lines.

echo "=== Step 1: Lint ==="
LINT_CMD=(snakemake --snakefile "$SNAKEFILE" --configfile .test/config/custom/config.yaml --lint)
echo "CMD: ${LINT_CMD[*]}" >> "$LOG_FILE"
LINT_OUT=$("${LINT_CMD[@]}" 2>&1 || true)

if echo "$LINT_OUT" | grep -q "WorkflowError\|SyntaxError\|Error in"; then
  echo "FAIL  lint (parse/schema error detected)"
  echo "$LINT_OUT"
  exit 1
else
  echo "PASS  lint (advisory warnings only)"
  if $VERBOSE; then
    echo "$LINT_OUT"
  fi
fi
echo ""

# ── Step 2: Dry-run matrix ───────────────────────────────────────────────────

echo "=== Step 2: Dry-run matrix (18 cases) ==="

# Ordered list of "name:dir" pairs (associative arrays have undefined order in
# bash 4, so we use parallel arrays for deterministic output).
NAMES=(
  "custom"
  "pulldown"
  "stoichio-screen"
  "vds"
  "ep/data_pipeline_ready"
  "ep/inference_ready"
  "ep/merge_ready"
  "ep/dp+ir"
  "ep/dp+mr"
  "ep/dp+raw"
  "ep/ir+raw"
  "ep/mr+ir"
  "ep/mr+raw"
  "ep/ir+mr+dp"
  "ep/ir+raw+dp"
  "ep/ir+raw+mr"
  "ep/mr+dp+raw"
  "ep/all"
)

DIRS=(
  ".test/config/custom"
  ".test/config/pulldown"
  ".test/config/stoichio-screen"
  ".test/config/vds"
  ".test/config/entry_points/data_pipeline_ready"
  ".test/config/entry_points/inference_ready"
  ".test/config/entry_points/merge_ready"
  ".test/config/entry_points/data_pipeline_ready_plus_inference_ready"
  ".test/config/entry_points/data_pipeline_ready_plus_merge_ready"
  ".test/config/entry_points/data_pipeline_ready_plus_raw_data"
  ".test/config/entry_points/inference_ready_plus_raw_data"
  ".test/config/entry_points/merge_ready_plus_inference_ready"
  ".test/config/entry_points/merge_ready_plus_raw_data"
  ".test/config/entry_points/inference_ready_plus_merge_ready_plus_data_pipeline_ready"
  ".test/config/entry_points/inference_ready_plus_raw_data_plus_data_pipeline_ready"
  ".test/config/entry_points/inference_ready_plus_raw_data_plus_merge_ready"
  ".test/config/entry_points/merge_ready_plus_data_pipeline_ready_plus_raw_data"
  ".test/config/entry_points/all"
)

PASS=0
FAIL=0
FAILED_CASES=()

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  dir="${DIRS[$i]}"
  echo "# test: $name" >> "$LOG_FILE"
  if run_snakemake "$dir" --dry-run --cores 1; then
    printf "  PASS  %s\n" "$name"
    ((PASS++)) || true
  else
    printf "  FAIL  %s\n" "$name"
    ((FAIL++)) || true
    FAILED_CASES+=("$name")
  fi
done

echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="

if [ "${#FAILED_CASES[@]}" -gt 0 ]; then
  echo ""
  echo "Failed cases:"
  for c in "${FAILED_CASES[@]}"; do
    echo "  - $c"
  done
  echo ""
  echo "Re-run with --verbose to see full Snakemake output for failing cases."
  exit 1
fi

echo "All tests passed."
