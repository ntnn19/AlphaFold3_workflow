#!/bin/bash
# gpu_lock.sh — acquire an exclusive GPU slot then exec a command.
#
# Usage:
#   gpu_lock.sh <lock_dir> <command> [args...]
#
# Arguments:
#   lock_dir  Directory for GPU lock files. Must be on a filesystem shared
#             across all concurrent container instances (e.g. a bind-mounted
#             host path). Created if it does not exist.
#   command   The command to run once a GPU slot is acquired.
#   args      Any additional arguments forwarded verbatim to <command>.
#
# Behaviour:
#   - Discovers real GPU indices via `nvidia-smi --query-gpu=index` (respects
#     any pre-existing CUDA_VISIBLE_DEVICES / MIG masking the parent set).
#   - Makes one non-blocking flock sweep over all GPUs (fast path: grab a free
#     GPU instantly with zero added latency).
#   - If every GPU is busy, blocks on a single flock with a timeout instead of
#     polling — the OS wakes this process the instant any GPU is released, so
#     idle GPU time between jobs is eliminated (no fixed 30 s sleep penalty).
#   - On success: sets CUDA_VISIBLE_DEVICES to the *physical UUID* of the GPU
#     and execs <command> [args...] (replaces this shell; PID identity kept for
#     schedulers/cgroups).
#   - The flock is held for the lifetime of the process and released
#     automatically on exit, crash, or SIGKILL (OS-level guarantee).
#
# Tunables (environment variables):
#   GPU_LOCK_WAIT_TIMEOUT  Seconds to block per blocking-wait attempt (default 300).
#                          Bounds the wait so a dead lock holder can't wedge us
#                          forever; on timeout we re-sweep and retry.
#
# Example (from a Snakemake shell directive):
#   bash /app/scripts/gpu_lock.sh /root/af_output/.gpu_locks \
#       python /app/alphafold/run_alphafold.py --json_path=... [other flags]

set -euo pipefail

LOCK_DIR="${1:?gpu_lock.sh: lock_dir argument is required}"
shift  # remaining args are the command to run

if [[ $# -eq 0 ]]; then
    echo "gpu_lock.sh: no command specified" >&2
    exit 1
fi

command -v nvidia-smi >/dev/null 2>&1 || {
    echo "gpu_lock.sh: nvidia-smi not found on PATH" >&2
    exit 1
}

WAIT_TIMEOUT="${GPU_LOCK_WAIT_TIMEOUT:-300}"

# Discover the GPUs actually visible to this process. Query index + UUID so we
# can (a) name lock files by stable UUID rather than an ordinal that can shift,
# and (b) pin the child to a specific physical device by UUID.
mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d ' ')

if [[ "${#GPU_UUIDS[@]}" -eq 0 ]]; then
    echo "gpu_lock.sh: no GPUs detected by nvidia-smi" >&2
    exit 1
fi

mkdir -p "$LOCK_DIR"

# Try to grab a GPU. $1 = flock flag string ("-n" for non-blocking sweep,
# "-w <secs>" for a bounded blocking wait). Execs the command on success;
# returns non-zero if this particular GPU could not be locked.
try_lock_gpu() {
    local uuid="$1"; shift
    local flock_flags="$1"; shift
    local lock_fd
    # Lock filename keyed by UUID (sanitised: keep alnum, dash, dot).
    local lock_file="${LOCK_DIR}/gpu_${uuid//[^A-Za-z0-9._-]/_}.lock"

    exec {lock_fd}>>"$lock_file"
    # shellcheck disable=SC2086  # flock_flags is intentionally word-split
    if flock $flock_flags "$lock_fd"; then
        export CUDA_VISIBLE_DEVICES="$uuid"
        exec "$@"  # replaces this shell; lock_fd stays open (and held) for its lifetime
    fi
    exec {lock_fd}>&-  # release the fd on failure so retries don't leak descriptors
    return 1
}

while true; do
    # Fast path: one non-blocking sweep. Grabs an idle GPU with zero extra latency.
    for uuid in "${GPU_UUIDS[@]}"; do
        try_lock_gpu "$uuid" "-n" "$@" || true
    done

    # Slow path: all GPUs busy. Block (with a timeout) on each in turn instead of
    # spin-sleeping. flock returns the moment a holder releases, so we reclaim a
    # freed GPU immediately rather than up to 30 s later.
    for uuid in "${GPU_UUIDS[@]}"; do
        try_lock_gpu "$uuid" "-w $WAIT_TIMEOUT" "$@" || true
    done
    # If we fall through here, every blocking wait timed out; loop and re-sweep.
done