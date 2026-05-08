#!/bin/bash
# gpu_lock_run.sh — acquire an exclusive GPU slot then run a command.
#
# Usage:
#   gpu_lock_run.sh <lock_dir> <command> [args...]
#
# Arguments:
#   lock_dir   Directory for GPU lock files. Must be on a filesystem shared
#              across all concurrent container instances (e.g. a bind-mounted
#              host path). The directory must already exist.
#   command    The command to run once a GPU slot is acquired.
#   args       Any additional arguments forwarded verbatim to <command>.
#
# Behaviour:
#   - Queries the number of available GPUs via nvidia-smi at runtime.
#   - Iterates over GPU indices, attempting a non-blocking flock on each.
#   - On success: sets CUDA_VISIBLE_DEVICES and execs <command> [args...].
#   - On failure (all GPUs busy): sleeps 30 s and retries indefinitely.
#   - The flock is held for the lifetime of the process and released
#     automatically on exit, crash, or SIGKILL (OS-level guarantee).
#
# Example (from a Snakemake shell directive):
#   bash /app/scripts/gpu_lock_run.sh /root/af_output/.gpu_locks \
#     python /app/alphafold/run_alphafold.py --json_path=... [other flags]

set -euo pipefail

LOCK_DIR="${1:?gpu_lock_run.sh: lock_dir argument is required}"
shift  # remaining args are the command to run

if [[ $# -eq 0 ]]; then
    echo "gpu_lock_run.sh: no command specified" >&2
    exit 1
fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [[ "$NUM_GPUS" -eq 0 ]]; then
    echo "gpu_lock_run.sh: no GPUs detected by nvidia-smi" >&2
    exit 1
fi

while true; do
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        exec {lock_fd}>>"${LOCK_DIR}/gpu_${gpu}.lock"
        if flock --nonblock ${lock_fd}; then
            export CUDA_VISIBLE_DEVICES=${gpu}
            exec "$@"   # replaces this shell; lock_fd held until process exits
        else
            exec {lock_fd}>&-
        fi
    done
    sleep 30
done
