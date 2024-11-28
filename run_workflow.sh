#!/bin/bash
source /etc/profile.d/modules.sh
source /gpfs/cssb/software/envrc
module reset
module load maxwell mamba
. mamba-init
ENV_DIR=../../../../colabfold_structure_prediction_workflow_maxwell
mamba env create --prefix $ENV_DIR/env -f $ENV_DIR/environment.yml
mamba activate $ENV_DIR/env
AF3_CONTAINER=$1
AF3_WEIGHTS_DIR=$2
INPUT_DIR=$3
OUTPUT_DIR=$4
DB_DIR=$5
mkdir -p "$OUTPUT_DIR"
mkdir -p logs
cmd="snakemake --use-singularity --config af3_container=$AF3_CONTAINER --singularity-args '--nv -B ${AF3_WEIGHTS_DIR}:/root/models -B ${INPUT_DIR}:/root/af_input -B ${OUTPUT_DIR}:/root/af_output -B ${DB_DIR}:/root/public_databases' -j 99 --set-resources RUN_AF3:runtime=20000min RUN_AF3:slurm_partition=topfgpu RUN_AF3:cpus_per_task=128 --executor slurm" 
#cmd="snakemake --use-singularity --config af3_container=$AF3_CONTAINER --singularity-args '--nv -B ${AF3_WEIGHTS_DIR}:/root/models -B ${INPUT_DIR}:/root/af_input -B ${OUTPUT_DIR}:/root/af_output -B ${DB_DIR}:/root/public_databases' -j 99 --set-resources RUN_AF3:runtime=20000min RUN_AF3:slurm_partition=cssbgpu RUN_AF3:slurm_nodelist=max-cssbg018,max-cssbg019,max-cssbg020,max-cssbg021,max-cssbg022,max-cssbg023 RUN_AF3:cpus_per_task=128 --executor slurm"
echo "$cmd"
echo "$0 $AF3_CONTAINER $AF3_WEIGHTS_DIR $INPUT_DIR $OUTPUT_DIR $OUTPUT_DIR" >> logs/workflow_invocations_log.txt
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >> logs/workflow_invocations_log.txt
echo "$cmd" >> logs/workflow_invocations_log.txt
echo "########################################" >> logs/workflow_invocations_log.txt
eval $cmd
