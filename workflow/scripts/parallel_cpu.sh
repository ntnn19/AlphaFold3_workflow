# Determine number of CPUs on the node we ended up on.
JOB_LIST=$1
NUM_CPUS=$(lscpu | awk '/^CPU\(s\):/ {print $2}')
echo "host=$(hostname), number_of_cpus=${NUM_CPUS}"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "SLURM_JOB_CPUS_PER_NODE=$SLURM_JOB_CPUS_PER_NODE"
echo "nproc=$(nproc)"
< $JOB_LIST parallel -j $NUM_CPUS '{}'
