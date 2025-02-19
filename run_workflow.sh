#!/bin/bash
ENV_DIR=$(pwd)/venv
mamba activate $ENV_DIR
AF3_CONTAINER=$1
AF3_WEIGHTS_DIR=$2
INPUT_DIR=$3
OUTPUT_DIR=$4
DB_DIR=$5
TMP_DIR=$6
mkdir -p "$OUTPUT_DIR"
mkdir -p logs
mkdir -p $TMP_DIR
cmd="snakemake --use-singularity --config af3_container=$AF3_CONTAINER dataset=$INPUT_DIR output_dir=$OUTPUT_DIR --singularity-args '--nv -B ${AF3_WEIGHTS_DIR}:/root/models -B ${INPUT_DIR}:/root/af_input -B ${OUTPUT_DIR}:/root/af_output -B ${DB_DIR}:/root/public_databases -B ${TMP_DIR}:/tmp --env XLA_CLIENT_MEM_FRACTION=3.2' -j unlimited -c all --executor slurm --set-resources RUN_AF3_INFERENCE:slurm_partition=vds RUN_AF3_INFERENCE:mem_mb=16000 RUN_AF3_DATA:cpus_per_task=16 RUN_AF3_DATA:slurm_partition=vds --groups RUN_AF3_DATA=group0 --group-components group0=3 -p"
#cmd="snakemake --use-singularity --config af3_container=$AF3_CONTAINER dataset=$INPUT_DIR output_dir=$OUTPUT_DIR --singularity-args '--nv -B ${AF3_WEIGHTS_DIR}:/root/models -B ${INPUT_DIR}:/root/af_input -B ${OUTPUT_DIR}:/root/af_output -B ${DB_DIR}:/root/public_databases -B ${TMP_DIR}:/tmp --env XLA_CLIENT_MEM_FRACTION=3.2' -j unlimited -c all --executor slurm --set-resources RUN_AF3_INFERENCE:slurm_partition=vds RUN_AF3_INFERENCE:mem_mb=16000 RUN_AF3_DATA:cpus_per_task=16 RUN_AF3_DATA:slurm_partition=vds --groups RUN_AF3_DATA=group0 RUN_AF3_INFERENCE=group1 --group-components group0=3 group1=2 -p"
echo "$cmd"
echo "$0 $AF3_CONTAINER $AF3_WEIGHTS_DIR $INPUT_DIR $OUTPUT_DIR $OUTPUT_DIR" >> logs/workflow_invocations_log.txt
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >> logs/workflow_invocations_log.txt
echo "$cmd" >> logs/workflow_invocations_log.txt
echo "########################################" >> logs/workflow_invocations_log.txt
eval $cmd
