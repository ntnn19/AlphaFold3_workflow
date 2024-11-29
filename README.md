# AlphaFold3 Workflow

This workflow supports separate execution of the **CPU** and **GPU** steps. It also distributes inference runs across multiple GPU devices using **GNU parallel**.

## Steps to setup & execute

### 1. Build the Singularity container

Run the following command to build the Singularity container that supports parallel inference runs:

```bash
singularity build alphafold3_parallel.sif docker://ntnn19/alphafold3:latest_parallel_a100_40gb
```

**Notes**
- Set <number_of_inference_job_lists> to 1 for local runs.
- For SLURM runs, set <number_of_inference_job_lists> to n, where n is the number of nodes with GPUs.
- Make sure to download the required [AlphaFold3 databases](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md#obtaining-genetic-databases) and [weights](https://forms.gle/svvpY4u2jsHEwWYS6) before proceeding.

### 2. Clone This repository

Clone this repository into your project directory. After cloning, your project structure should look like this:

```bash
.  <-- This represents your current location
├── dataset_1
│   ├── af_input
│   ├── data_pipeline
│       └── <your_input_json_file>
├── example
│   └── example.json
├── README.md
└── workflow
    ├── scripts
    │   ├── create_job_list.py
    │   ├── parallel.sh
    │   └── split_json_and_create_job_list.py
    ├── Snakefile
```
An example JSON file is available in the example/ directory:
example/example.json

### 3. Create & activate the Snakemake environment

Install mamba or micromamba if not already installed. Then, set up and activate the environment using the following commands:
```bash 
mamba create -p $(pwd)/env -f environment.yml
```
```bash
mamba activate $(pwd)/env
```

### 4. Run the workflow
**Dry run (local)**
```bash
snakemake --use-singularity \
    --config af3_container=<path_to_your_alphafold3_container> \
    --singularity-args '--nv -B <alphafold3_weights_dir>:/root/models -B $(pwd)/<dataset_directory>/af_input:/root/af_input -B $(pwd)/<dataset_directory>/af_output:/root/af_output -B <path_to_alphafold3_db_directory>:/root/public_databases' \
    -c all \
    --set-scatter split=<number_of_inference_job_lists> -n
```
**Dry run (slurm)**
```bash
snakemake --use-singularity \
    --config af3_container=<path_to_your_alphafold3_container> \
    --singularity-args '--nv -B <alphafold3_weights_dir>:/root/models -B $(pwd)/<dataset_directory>/af_input:/root/af_input -B $(pwd)/<dataset_directory>/af_output:/root/af_output -B <path_to_alphafold3_db_directory>:/root/public_databases' \
    -j 99 \
    --executor slurm \
    --set-scatter split=<number_of_inference_job_lists> -n
```
**Local run**
```bash
snakemake --use-singularity \
    --config af3_container=<path_to_your_alphafold3_container> \
    --singularity-args '--nv -B <alphafold3_weights_dir>:/root/models -B $(pwd)/<dataset_directory>/af_input:/root/af_input -B $(pwd)/<dataset_directory>/af_output:/root/af_output -B <path_to_alphafold3_db_directory>:/root/public_databases' \
    -c all \
    --set-scatter split=<number_of_inference_job_lists>
```

**slurm run**
```bash
snakemake --use-singularity \
    --config af3_container=<path_to_your_alphafold3_container> \
    --singularity-args '--nv -B <alphafold3_weights_dir>:/root/models -B $(pwd)/<dataset_directory>/af_input:/root/af_input -B $(pwd)/<dataset_directory>/af_output:/root/af_output -B <path_to_alphafold3_db_directory>:/root/public_databases' \
    -j 99 \
    --executor slurm \
    --set-scatter split=<number_of_inference_job_lists> -n
```
