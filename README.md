# AlphaFold3 Workflow

This workflow supports separate execution of the **CPU** and **GPU** steps. It also distributes inference runs across multiple GPU devices using **GNU parallel**.


### TO DO
- Add a config file to allow running using different alphafold3 configurations & encode the configuration in the output file/directory names.
- Add a json preparation step for different analyses such as all-vs-all (e.g. for binary PPI) , baits-vs-targets (e.g. for drug screens), assembly, etc.
- Add steps for downstream analyses such as relaxation, assembly, binding site prediction, scoring etc. 
- Add a reporting step to the workflow in a form of a table that describe each predicted structure.


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

Clone this repository this repository.

```bash
git clone https://github.com/ntnn19/alphafold3_workflow.git
```

Go to the repository location
```bash
cd alphafold3_workflow
```

An example JSON and CSV files is available in the example directory:
example/example.json
example/all_vs_all.csv
example/pulldown.csv
example/virtual_drug_screen.csv

### 3. Create & activate the Snakemake environment

Install [mamba](https://github.com/conda-forge/miniforge) or [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) if not already installed.

Then, set up and activate the environment using the following commands:

```bash
mamba env create -p $(pwd)/venv -f environment.yml
mamba activate $(pwd)/venv
```

Or if using micromamba

```bash
micromamba env create -p $(pwd)/venv -f environment.yml
eval "$(micromamba shell hook --shell=bash)"
micromamba activate $(pwd)/venv
```
### 4. Configure the workflow

### 5. Run the workflow
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
