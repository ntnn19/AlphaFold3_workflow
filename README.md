# Snakemake workflow: `AlphaFold 3 workflow`

A Snakemake workflow for high-throughput AlphaFold 3 structure predictions.
This workflow has the following advantages over standard AlphaFold 3 runs:

1. **Separated data and inference pipelines** for better resource utilization.
2. **Implements the assemble-from-monomers technique** described in the official AlphaFold 3 documentation for predicting multimers. See [here](https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md#:~:text=Pre%2Dcomputing%20and%20reusing%20MSA%20and%20templates) for details.
3. **Treats each random seed as a separate job when multiple seeds are used**, substantially increasing throughput for large-scale sampling campaigns.
4. **Efficient on HPC systems with multi-GPU nodes** that lack support for consumable resources (i.e., nodes are allocated exclusively to a single user). See the configuration documentation at  ```config/README.md``` for details.



### 🚀 What’s new?

    📖 Better documentation to make setup & usage smoother
    🔄 Support for different running modes, including:
        🧲 Pulldown
        💊 Virtual screening
        🔬 All-vs-all pairwise interactions
        ⚖️  Stoichiometry screen
        🎲  Massive sampling


- [Snakemake workflow: `AlphaFold 3 workflow`](#snakemake-workflow-name)
  - [Usage](#usage)
  - [Deployment options](#deployment-options)
  - [Workflow profiles](#workflow-profiles)
  - [Authors](#authors)
  - [References](#references)
  - [TODO](#todo)

## Usage

Detailed information about input data and workflow configuration can also be found in the [`config/README.md`](config/README.md).

If you use this workflow in a paper, don't forget to give credits to the authors by citing the URL of this repository or its DOI.

## Deployment options

To run the workflow from command line, change the working directory.

```bash
git clone https://github.com/ntnn19/AlphaFold3_workflow.git
cd path/to/AlphaFold3_workflow
```

## Build the Singularity container (Optional)
**Use this option if you prefer to have a copy of the sif file in a specific directory instead of letting snakemake automatically build it. Otherwise, you can skip it.**
Install singularity. See [here](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md#install-singularity) or [here](https://docs.sylabs.io/guides/3.3/user-guide/installation.html) for instructions.
Run the following command to build the Singularity container that supports parallel inference runs:

```bash
singularity build alphafold3_parallel.sif docker://ntnn19/alphafold3:latest_parallel_a100_40gb
```

Or

```bash
singularity build alphafold3_parallel.sif docker://ntnn19/alphafold3:latest_parallel_a100_80gb
```

## Create & activate the Snakemake environment

Install [mamba](https://github.com/conda-forge/miniforge) or [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) if not already installed.

Then, set up and activate the environment using the following commands:

```bash
mamba env create -p $(pwd)/venv -f environment.yml
mamba activate $(pwd)/venv
```
For Maxwell users

```bash
module load maxwell mamba
. mamba-init
mamba env create -p $(pwd)/venv -f environment.yml
mamba activate $(pwd)/venv
```

Or if using micromamba

```bash
micromamba env create -p $(pwd)/venv -f environment.yml
eval "$(micromamba shell hook --shell=<YOUR SHELL>)"
micromamba activate $(pwd)/venv
```

## Notes

Make sure to download the required [AlphaFold3 databases](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md#obtaining-genetic-databases) and [weights](https://forms.gle/svvpY4u2jsHEwWYS6) before proceeding.


## Run the workflow

Adjust options in the default config file `config/config.yaml`.
Before running the complete workflow, you can perform a dry run using:

```bash
snakemake --dry-run
```


To run the workflow with test files using **singularity**, add a link to a container registry in the `config.yaml`, for example `af3_container: "oras://ghcr.io/<user>/<repository>:<version>"` for Github's container registry.
Run the workflow with:

```bash
snakemake --cores 2 --use-singularity --singularity-args "-B <LOCAL_AF3_SEQUENCE_DATABASE>:/root/public_databases -B <YOUR_AF3_PARAMETERS>:/root/models" --directory .test/config/custom
```

Other example JSON, TSV and configuration files (YAML format) files for testing other configurations are available in ```bash .test```

## Workflow profiles

The `profiles/` directory can contain any number of [workflow-specific profiles](https://snakemake.readthedocs.io/en/stable/executing/cli.html#profiles) that users can choose from.
The [profiles `README.md`](profiles/README.md) provides more details.

## Authors

- Nathan Nagar @ CSSB/LIV

## References

> Köster, J., Mölder, F., Jablonski, K. P., Letcher, B., Hall, M. B., Tomkins-Tinch, C. H., Sochat, V., Forster, J., Lee, S., Twardziok, S. O., Kanitz, A., Wilm, A., Holtgrewe, M., Rahmann, S., & Nahnsen, S. _Sustainable data analysis with Snakemake_. F1000Research, 10:33, 10, 33, **2021**. https://doi.org/10.12688/f1000research.29032.2.

## TODO

- Add configuration-specific `config/README.md` file.
- Add scoring report. 
- Update snakemake version to latest in environment.yml. 

If you find this useful, please consider giving it a star! ⭐
