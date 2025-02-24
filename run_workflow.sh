#!/bin/bash
ENV_DIR=$(pwd)/venv
micromamba activate $ENV_DIR
AF3_CONTAINER=$1
AF3_WEIGHTS_DIR=$2
INPUT_DIR=$3
OUTPUT_DIR=$4
DB_DIR=$5
TMP_DIR=$6
SNK_FILE=$7
mkdir -p "$OUTPUT_DIR"
mkdir -p logs
mkdir -p $TMP_DIR
cmd="snakemake -s $SNK_FILE --use-singularity --config af3_container=$AF3_CONTAINER dataset=$INPUT_DIR output_dir=$OUTPUT_DIR --singularity-args '--nv -B ${AF3_WEIGHTS_DIR}:/root/models -B $(realpath ${INPUT_DIR}):/root/af_input -B $(realpath ${OUTPUT_DIR}):/root/af_output -B ${DB_DIR}:/root/public_databases -B $(realpath ${TMP_DIR}):/tmp --env XLA_CLIENT_MEM_FRACTION=3.2' -j unlimited -c all --executor slurm --set-resources RUN_AF3_INFERENCE:slurm_partition=vds RUN_AF3_INFERENCE:mem_mb=16000 RUN_AF3_DATA:cpus_per_task=8 RUN_AF3_DATA:slurm_partition=vds --groups RUN_AF3_DATA=group0 --group-components group0=12 -p -k -w 30 --rerun-triggers mtime"
echo "$cmd"
echo "$0 $AF3_CONTAINER $AF3_WEIGHTS_DIR $INPUT_DIR $OUTPUT_DIR $DB_DIR $TMP_DIR $SNK_FILE" >> logs/workflow_invocations_log.txt
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >> logs/workflow_invocations_log.txt
echo "$cmd" >> logs/workflow_invocations_log.txt
echo "########################################" >> logs/workflow_invocations_log.txt
eval $cmd
