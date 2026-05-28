#!/usr/bin/env bash
# run_workflow.sh — convenience wrapper for the AF3 Snakemake workflow.
#
# Usage:
#   ./run_workflow.sh [CONFIG] [PROFILE] [EXTRA_SNAKEMAKE_ARGS...]
#
# Defaults:
#   CONFIG  = config/config.yaml
#   PROFILE = profiles/local
#
# Examples:
#   ./run_workflow.sh
#   ./run_workflow.sh config/my_run.yaml profiles/slurm/standard
#   ./run_workflow.sh config/config.yaml profiles/slurm/exclusive --dry-run
#
# Required environment variables (for container execution):
#   AF3_CONTAINER   — path to the AlphaFold3 Singularity/Apptainer .sif image
#   AF3_MODELS_DIR  — path to AF3 model weights directory
#   AF3_DB_DIR      — path to genetic databases directory (~2 TB)

set -euo pipefail

CONFIG="${1:-config/config.yaml}"
PROFILE="${2:-profiles/local}"
shift 2 || true

snakemake \
    --workflow-profile "${PROFILE}" \
    --configfile "${CONFIG}" \
    "$@"
