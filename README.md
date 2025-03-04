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

### Notes

Make sure to download the required [AlphaFold3 databases](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md#obtaining-genetic-databases) and [weights](https://forms.gle/svvpY4u2jsHEwWYS6) before proceeding.

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
Open config/config.yaml with your favourite text editor.
Edit the values to your needs.
#### Mandatory workflow flags:
##### This workflow adapts the input preparation logic from [AlphaFold3-GUI](https://github.com/Hanziwww/AlphaFold3-GUI).

See the following input examples: 
-   **output_dir:** <path_to_your_output_directory> # Stores the outputs of this workflow
-   **af3_flags:** # configures AlphaFold 3
     -   **af3_container:** <path_to_your_alphafold3_container> 
- **input_csv:** <path_to_your_csv_table> 

**Examples for supported input_csv files:**

**default:** example/default.csv

| job_name  | type    | id | sequence        |
|-----------|--------|----|----------------|
| TestJob1  | protein | A  | MVLSPADKTNVKAAW |
| TestJob1  | protein | B  | MVLSPADKTNVKAAW |
| TestJob2  | protein | C  | MVLSPADKTNVKAAW |

##### Explanation:
For explanation and full list of optional columns, see  [AlphaFold3-GUI api tutorial](https://alphafold3-gui.readthedocs.io/en/latest/tutorial.html)

##### all-vs-all: example/all_vs_all.csv 

| id  | type    | sequence          |
|-----|--------|------------------|
| p1  | rna    | AUGGCA           |
| p2  | protein | MKPSFDR          |
| p3  | protein | MVLSPADKTNVKAAW  |

##### Explanation:
- **id**: A unique identifier for each entry.
- **type**: The biological macromolecule type (protein, dna, or rna).
- **sequence**: The nucleotide or amino acid sequence.


virtual drug screen: example/virtual_drug_screen.csv

##### pulldown: example/pulldown.csv

| id  | type    | sequence          | bait_or_target | target_id | bait_id |
|-----|--------|------------------|----------------|-----------|---------|
| p2  | protein | MASEQASDTTVCIK   | target         | t2        |         |
| b1  | protein | MASEQASDTTVCIK   | bait           |           | b1      |
| b2  | protein | MASEQASDTTVCIK   | bait           |           | b2      |
| p1  | protein | MHIKPEERF        | target         | t1        |         |
| p3  | protein | ANHIREQDS        | target         | t2        |         |
| b3  | protein | ANHIREQDS        | bait           |           | b1      |

##### Explanation:
- **id**: A unique identifier for each entry.
- **type**: The biological macromolecule type (protein, dna, or rna).
- **sequence**: The nucleotide or amino acid sequence.
- **bait_or_target**: Indicates whether the protein is a "bait" or "target" in the experiment.
- **Optional columns: 
  - **target_id**: The identifier for the target.
  - **bait_id**: The identifier for the bait.

**The optional columns can be used to pulldown oligomeric targets with monomeric baits, oligomeric targets with oligomeric baits, or monomeric targets with oligomeric baits.**

#### Optional workflow flags:

By default the workflow will run in a default mode, to which a csv table such as the following is required:


#### Optional AlphaFold3 flags:
Include the optional flags within the scope of the af3_flags. 
The optional flags are:
<details>

--buckets

--conformer_max_iterations

--flash_attention_implementation

--gpu_device

--hmmalign_binary_path

--hmmbuild_binary_path

--hmmsearch_binary_path

--jackhmmer_binary_path

--jackhmmer_n_cpu

--jax_compilation_cache_dir

--max_template_date

--mgnify_database_path

--nhmmer_binary_path

--nhmmer_n_cpu

--ntrna_database_path

--num_diffusion_samples

--num_recycles

--num_seeds

--pdb_database_path

--rfam_database_path

--rna_central_database_path

--save_embeddings

--seqres_database_path

--small_bfd_database_path

--uniprot_cluster_annot_database_path

--uniref90_database_path
</details>

### 5. Configure the profile (Optional)

For running this workflow on HPC using slurm, you can modify profile/config.yaml to make it compatible with your HPC setting.
More details can be found [here](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html#using-profiles).

### 6. Run the workflow
Information on snakemake flags can be found [here](https://snakemake.readthedocs.io/en/stable/executing/cli.html#)

**Dry run (local)**
```bash
snakemake -s workflow/Snakefile \
--use-singularity --singularity-args  \
'--nv -B <your_alphafold3_weights_dir>:/root/models -B <your_output_dir>/PREPROCESSING:/root/af_input -B <your_output_dir>:/root/af_output -B <your_alphafold3_databases_dir>:/root/public_databases -B <your_alphafold3_tmp_dir>/tmp:/tmp --env XLA_CLIENT_MEM_FRACTION=3.2' \
-j unlimited -c all --executor slurm --groups \
RUN_AF3_DATA=group0 --group-components group0=12 \
-p -k -w 30 --rerun-triggers mtime -n
```
**Dry run (slurm)**
```bash
snakemake -s workflow/Snakefile \
--use-singularity --singularity-args  \
'--nv -B <your_alphafold3_weights_dir>:/root/models -B <your_output_dir>/PREPROCESSING:/root/af_input -B <your_output_dir>:/root/af_output -B <your_alphafold3_databases_dir>:/root/public_databases -B <your_alphafold3_tmp_dir>/tmp:/tmp --env XLA_CLIENT_MEM_FRACTION=3.2' \
-j unlimited -c all --executor slurm --groups \
RUN_AF3_DATA=group0 --group-components group0=12 \
-p -k -w 30 --rerun-triggers mtime --workflow-profile profile -n
```
**Local run**
```bash
snakemake -s workflow/Snakefile \
--use-singularity --singularity-args  \
'--nv -B <your_alphafold3_weights_dir>:/root/models -B <your_output_dir>/PREPROCESSING:/root/af_input -B <your_output_dir>:/root/af_output -B <your_alphafold3_databases_dir>:/root/public_databases -B <your_alphafold3_tmp_dir>/tmp:/tmp --env XLA_CLIENT_MEM_FRACTION=3.2' \
-j unlimited -c all --executor slurm --groups \
RUN_AF3_DATA=group0 --group-components group0=12 \
-p -k -w 30 --rerun-triggers mtime
```

**slurm run**
```bash
snakemake -s workflow/Snakefile \
--use-singularity --singularity-args  \
'--nv -B <your_alphafold3_weights_dir>:/root/models -B <your_output_dir>/PREPROCESSING:/root/af_input -B <your_output_dir>:/root/af_output -B <your_alphafold3_databases_dir>:/root/public_databases -B <your_alphafold3_tmp_dir>/tmp:/tmp --env XLA_CLIENT_MEM_FRACTION=3.2' \
-j unlimited -c all --executor slurm --groups \
RUN_AF3_DATA=group0 --group-components group0=12 \
-p -k -w 30 --rerun-triggers mtime --workflow-profile profile
```
