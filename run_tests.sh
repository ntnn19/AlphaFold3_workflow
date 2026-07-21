#!/usr/bin/env bash
# run_tests.sh — run the full AF3 workflow dry-run test suite locally.
#
# Usage:
#   bash run_tests.sh [--verbose]
#
# Requirements:
#   - Snakemake >= 8.0 installed and on PATH (activate the venv first)
#   - GNU `parallel` on PATH (needed by the MUTATE rule; not exercised in the
#     dry-run matrix but required for a full run)
#   - Run from the repository root
#
# What it does:
#   1. Lints the workflow against the config schema (advisory warnings are
#      printed but do not fail the suite; only parse/schema errors fail).
#   2. POSITIVE MATRIX — runs `--dry-run` for every bundled fixture that is
#      expected to succeed (mode tests + all entry-point single/combination
#      tests + mutation/seedless edge cases). Each must exit 0.
#   3. NEGATIVE TEST — runs the canonical-name collision fixture and asserts it
#      FAILS with the expected WorkflowError (a pass here means it errored).
#   4. IDEMPOTENCY TEST — dry-runs a multi-seed inference_ready fixture twice and
#      asserts the exploded JSONs under normalized_inputs/ are NOT rewritten
#      (mtimes unchanged), proving load-time normalization is idempotent.
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

# Run a dry-run for a fixture directory. Returns snakemake's exit code.
run_dry() {
  local config_dir="$1"
  local cmd=(snakemake
    --snakefile "$SNAKEFILE"
    --configfile "$config_dir/config.yaml"
    --directory  "$config_dir"
    --dry-run --cores 1
  )
  echo "CMD: ${cmd[*]}" >> "$LOG_FILE"
  if $VERBOSE; then
    "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
    return "${PIPESTATUS[0]}"
  else
    "${cmd[@]}" >> "$LOG_FILE" 2>&1
  fi
}

# ── Initialise log ───────────────────────────────────────────────────────────

: > "$LOG_FILE"
echo "# run_tests.sh — $(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

OVERALL_FAIL=0

# ── Step 1: Lint ─────────────────────────────────────────────────────────────
# `snakemake --lint` exits 1 when it finds style warnings (not just parse
# errors), so we capture output and only fail on actual parse/schema errors.

echo "=== Step 1: Lint ==="
LINT_CMD=(snakemake --snakefile "$SNAKEFILE" --configfile .test/config/custom/config.yaml --directory .test/config/custom --lint)
echo "CMD: ${LINT_CMD[*]}" >> "$LOG_FILE"
LINT_OUT=$("${LINT_CMD[@]}" 2>&1 || true)
echo "$LINT_OUT" >> "$LOG_FILE"

if echo "$LINT_OUT" | grep -qE "WorkflowError|SyntaxError|Error in rule|KeyError|Exception"; then
  echo "FAIL  lint (parse/schema error detected)"
  echo "$LINT_OUT"
  OVERALL_FAIL=1
else
  echo "PASS  lint (advisory warnings only)"
  if $VERBOSE; then
    echo "$LINT_OUT"
  fi
fi
echo ""

# ── Step 2: Positive dry-run matrix ──────────────────────────────────────────
# Every fixture here is expected to build a valid DAG (exit 0). Checkpoint-driven
# fixtures (raw_data / screen modes) legitimately stop at the PREPROCESSING
# checkpoint in a dry-run; that still exits 0.

echo "=== Step 2: Positive dry-run matrix ==="

# Parallel arrays "name" / "dir" for deterministic ordering.
POS_NAMES=(
  # ── run modes (raw_data entry point) ──
  "mode/custom"
  "mode/pulldown"
  "mode/stoichio-screen"
  "mode/virtual-drug-screen"
  # ── single entry points ──
  "ep/data_pipeline_ready"
  "ep/inference_ready"
  "ep/merge_ready"
  # ── two-entry-point combinations ──
  "ep/data_pipeline_ready+inference_ready"
  "ep/data_pipeline_ready+merge_ready"
  "ep/data_pipeline_ready+raw_data"
  "ep/inference_ready+raw_data"
  "ep/merge_ready+inference_ready"
  "ep/merge_ready+raw_data"
  # ── three-entry-point combinations ──
  "ep/inference_ready+merge_ready+data_pipeline_ready"
  "ep/inference_ready+raw_data+data_pipeline_ready"
  "ep/inference_ready+raw_data+merge_ready"
  "ep/merge_ready+data_pipeline_ready+raw_data"
  # ── all four entry points ──
  "ep/all"
  # ── mutation + normalization edge cases ──
  "edge/raw_data_mutation"
  "edge/inference_ready_mutation"
  "edge/inference_ready_seedless"
  # bare / seedless sample NAME (no seed-<N> token) is a VALID input: the seed
  # comes from the JSON modelSeeds (default [1]), not the filename.
  "edge/inference_ready_seedless_no_seed"
)

POS_DIRS=(
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
  ".test/config/raw_data_mutation"
  ".test/config/entry_points/inference_ready_mutation"
  ".test/config/entry_points/inference_ready_seedless"
  ".test/config/entry_points/inference_ready_seedless_no_seed"
)

PASS=0
FAIL=0
FAILED_CASES=()

for i in "${!POS_NAMES[@]}"; do
  name="${POS_NAMES[$i]}"
  dir="${POS_DIRS[$i]}"
  echo "# positive test: $name  ($dir)" >> "$LOG_FILE"
  if [ ! -f "$dir/config.yaml" ] || [ ! -s "$dir/config.yaml" ]; then
    printf "  FAIL  %s (missing/empty config: %s)\n" "$name" "$dir"
    ((FAIL++)) || true
    FAILED_CASES+=("$name (missing config)")
    continue
  fi
  if run_dry "$dir"; then
    printf "  PASS  %s\n" "$name"
    ((PASS++)) || true
  else
    printf "  FAIL  %s\n" "$name"
    ((FAIL++)) || true
    FAILED_CASES+=("$name")
  fi
done

echo ""
echo "  Positive matrix: ${PASS} passed, ${FAIL} failed"
if [ "$FAIL" -gt 0 ]; then
  OVERALL_FAIL=1
fi
echo ""

# ── Step 3: Negative tests (fixtures that MUST fail loudly) ───────────────────
# Each negative test wires an invalid configuration that the workflow is
# expected to reject with a clear WorkflowError. A "PASS" here means the dry-run
# correctly failed (non-zero exit) AND the error message matched the expected
# pattern — i.e. the workflow fails fast with an actionable message instead of
# crashing cryptically or silently producing wrong output.

echo "=== Step 3: Negative tests (must fail loudly) ==="

# expect_fail <label> <config_dir> <grep-pattern>
expect_fail() {
  local label="$1" dir="$2" pattern="$3"
  echo "# negative test: $label  ($dir)" >> "$LOG_FILE"
  local out rc
  out=$(snakemake --snakefile "$SNAKEFILE" \
    --configfile "$dir/config.yaml" \
    --directory "$dir" \
    --dry-run --cores 1 2>&1)
  rc=$?
  echo "$out" >> "$LOG_FILE"
  # A genuine AttributeError means a guard regressed into a cryptic crash.
  if echo "$out" | grep -q "AttributeError"; then
    echo "  FAIL  $label raised a cryptic AttributeError (guard regressed)"
    echo "$out" | grep -A2 "AttributeError" | head
    OVERALL_FAIL=1
    return
  fi
  if [ "$rc" -ne 0 ] && echo "$out" | grep -qiE "$pattern"; then
    echo "  PASS  $label correctly rejected (exit=$rc, matched /$pattern/)"
    if $VERBOSE; then
      echo "$out" | grep -iE "$pattern" | head
    fi
  else
    echo "  FAIL  $label NOT rejected as expected (exit=$rc, pattern /$pattern/ not found)"
    echo "$out" | tail -20
    OVERALL_FAIL=1
  fi
}

# 3a. Canonical-name collision: the SAME stem from two distinct entry points
#     (data_pipeline_ready + inference_ready) must be detected at load time.
#     (Seedless / bare sample names are NOT an error case any more — the seed is
#     resolved from the JSON modelSeeds, so those fixtures live in the positive
#     matrix above.)
expect_fail "collision" \
  ".test/config/entry_points/collision_dp_vs_inference_ready" \
  "collision"

echo ""

# ── Step 4: Idempotency test ─────────────────────────────────────────────────
# Load-time normalization writes exploded one-seed-per-file JSONs to
# normalized_inputs/. Because common.smk runs on every invocation (incl.
# --dry-run), writes MUST be content-addressed: a second identical run must not
# touch mtimes, otherwise AF3_INFERENCE would needlessly rerun.

echo "=== Step 4: Idempotency test (normalized_inputs not rewritten) ==="
IDEM_DIR=".test/config/entry_points/inference_ready_seedless"
IDEM_NORM="$IDEM_DIR/results/normalized_inputs/inference_ready"
echo "# idempotency test  ($IDEM_DIR)" >> "$LOG_FILE"

# Clean slate so run 1 is the initial write.
rm -rf "$IDEM_DIR/results" "$IDEM_DIR/.snakemake"

snakemake --snakefile "$SNAKEFILE" --configfile "$IDEM_DIR/config.yaml" \
  --directory "$IDEM_DIR" --dry-run --cores 1 >> "$LOG_FILE" 2>&1
IDEM_RC1=$?

if [ "$IDEM_RC1" -ne 0 ] || [ ! -d "$IDEM_NORM" ]; then
  echo "  FAIL  idempotency setup failed (run 1 exit=$IDEM_RC1, normalized dir present=$([ -d "$IDEM_NORM" ] && echo yes || echo no))"
  OVERALL_FAIL=1
else
  N_JSON=$(find "$IDEM_NORM" -name '*.json' | wc -l | tr -d ' ')
  MTIMES1=$(stat -c '%Y' "$IDEM_NORM"/*.json 2>/dev/null | sort | md5sum)
  sleep 1.2  # ensure any rewrite would land on a different second-resolution mtime
  snakemake --snakefile "$SNAKEFILE" --configfile "$IDEM_DIR/config.yaml" \
    --directory "$IDEM_DIR" --dry-run --cores 1 >> "$LOG_FILE" 2>&1
  IDEM_RC2=$?
  MTIMES2=$(stat -c '%Y' "$IDEM_NORM"/*.json 2>/dev/null | sort | md5sum)

  if [ "$IDEM_RC2" -eq 0 ] && [ "$MTIMES1" = "$MTIMES2" ]; then
    echo "  PASS  ${N_JSON} exploded JSON(s) unchanged across two dry-runs (idempotent)"
  else
    echo "  FAIL  normalized_inputs mtimes changed on rerun (run2 exit=$IDEM_RC2) — normalization not idempotent"
    OVERALL_FAIL=1
  fi
fi
# Clean up idempotency artifacts so the repo stays pristine.
rm -rf "$IDEM_DIR/results" "$IDEM_DIR/.snakemake"
echo ""

# ── Summary ──────────────────────────────────────────────────────────────────

echo "=================================================="
if [ "$OVERALL_FAIL" -eq 0 ]; then
  echo "ALL TESTS PASSED"
  echo "=================================================="
  exit 0
else
  echo "SOME TESTS FAILED"
  if [ "${#FAILED_CASES[@]}" -gt 0 ]; then
    echo ""
    echo "Failed positive cases:"
    for c in "${FAILED_CASES[@]}"; do
      echo "  - $c"
    done
  fi
  echo ""
  echo "See $LOG_FILE for full Snakemake output. Re-run with --verbose for live output."
  echo "=================================================="
  exit 1
fi
