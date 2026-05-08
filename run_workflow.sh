#!/bin/bash

# Configurable wrapper script for launching the AlphaFold3 workflow
# All paths are passed explicitly as arguments - no environment variables required

# Parse arguments
output_dir=$1
configfile=$2
models_path=$3
databases_path=$4
tmp_path=$5
extra_flgs="$6"

# Check if mandatory arguments are provided
if [ -z "$output_dir" ] || [ -z "$configfile" ] || [ -z "$models_path" ] || [ -z "$databases_path" ] || [ -z "$tmp_path" ]; then
    echo "Error: Missing mandatory arguments."
    echo "Usage: $0 <output_dir> <config_file> <models_path> <databases_path> <tmp_path> [<extra_flgs>]"
    echo ""
    echo "Mandatory arguments:"
    echo "  output_dir           - Where all outputs will be written"
    echo "  config_file          - Path to your workflow configuration file"
    echo "  models_path          - Path to AlphaFold 3 model weights directory"
    echo "  databases_path       - Path to genetic databases directory"
    echo "  tmp_path             - Path to temporary directory"
    echo ""
    echo "Optional arguments:"
    echo "  extra_flgs           - Additional flags to pass to snakemake (e.g., '--dry-run')"
    echo ""
    echo "Example:"
    echo "  $0 results/custom config/my_config.yaml /path/to/models /path/to/databases /path/to/tmp"
    echo ""
    echo "Example with extra flags:"
    echo "  $0 results/custom config/my_config.yaml /path/to/models /path/to/databases /path/to/tmp '--dry-run'"
    exit 1
fi

# Create necessary directories
mkdir -p "$output_dir"
mkdir -p "logs"
mkdir -p "$tmp_path"

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Step 1: Prepare workflow
echo "Preparing workflow..."
python "$SCRIPT_DIR/workflow/scripts/prepare_workflow.py" "$configfile" -o $(pwd)

# Check if preparation was successful
if [ $? -ne 0 ]; then
    echo "Error: Workflow preparation failed."
    exit 1
fi


# Step 2: Execute workflow with Snakemake
echo "Running workflow..."

# Build and execute the snakemake command
cmd="snakemake -s $SCRIPT_DIR/workflow/Snakefile \
  --configfile "$configfile" \
  --directory "$PWD" \
  --use-singularity \
  --singularity-args '\
    --nv \
    -B "$models_path":/root/models \
    -B "$databases_path":/root/public_databases \
    -B "$tmp_path"/:/tmp \
    -B "$(realpath "$output_dir")":/root/af_output \
    -B "$SCRIPT_DIR"/workflow/scripts:/app/scripts' \
  $extra_flgs"


echo "Executing command:"
echo "$cmd"
echo ""

# Log the command for reference
echo "$0 $output_dir $configfile $models_path $databases_path $tmp_path $extra_flgs" >> logs/workflow_invocations_log.txt
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >> logs/workflow_invocations_log.txt
echo "$cmd" >> logs/workflow_invocations_log.txt
echo "########################################" >> logs/workflow_invocations_log.txt

# Execute the command
eval "$cmd"

# Check if workflow execution was successful
if [ $? -eq 0 ]; then
    echo "Workflow completed successfully!"
else
    echo "Error: Workflow execution failed."
    exit 1
