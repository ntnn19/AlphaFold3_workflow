#

#        --singularity-args '--env DB_DIR=/data/public_databases --env MMCIF_DIR=/data/public_databases/mmcif_files --env MMSEQS_DB_DIR=/data/mmseqs_databases     --bind /scratch:/scratch --bind /gpfs/cssb/group/cssb-topf/natan/alphafast_db:/data/public_databases --bind /gpfs/cssb/group/cssb-topf/natan/alphafast_db/mmseqs:/data/mmseqs_databases'
DB_DIR="/gpfs/cssb/group/cssb-topf/natan/alphafast_db"
MMSEQS_DB_DIR="${DB_DIR}/mmseqs"
MODEL_DIR="/gpfs/cssb/group/cssb-topf/natan/singularity_containers/alphafold3/weights"
TEMP_DIR="/scratch"
LOG="log/log_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "log"
(
set -euo pipefail
set -x
snakemake \
		--directory .test/config/entry_points/data_pipeline_ready \
		--configfile .test/config/entry_points/data_pipeline_ready/config.yaml \
		--config batch_size=5 alphafast_container=/gpfs/cssb/group/cssb-topf/natan/tools/alphafast/alphafast.sif \
		--workflow-profile profiles/slurm/exclusive \
		--sdm apptainer \
		-w 50 \
		--rerun-trigger mtime \
		--trust-io-cache \
		-c all \
		--singularity-args  '--env MODEL_DIR=/gpfs/cssb/group/cssb-topf/natan/singularity_containers/alphafold3/weights --env DB_DIR=/gpfs/cssb/group/cssb-topf/natan/alphafast_db --env MMSEQS_DB_DIR=/gpfs/cssb/group/cssb-topf/natan/alphafast_db/mmseqs -B /gpfs/cssb/group/cssb-topf/vds/workflows/AlphaFold3_workflow_v8.14.0_v0.8.0/workflow/scripts:/app/alphafold/scripts'
) > $LOG 2>&1		

snakemake --directory .test/config/entry_points/data_pipeline_ready \
		--configfile .test/config/entry_points/data_pipeline_ready/config.yaml \
		--config batch_size=5 alphafast_container=/gpfs/cssb/group/cssb-topf/natan/tools/alphafast/alphafast.sif \
		--report report.zip -c1