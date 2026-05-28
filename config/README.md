# Configuration Reference

This document describes every parameter accepted by `config/config.yaml`.
Run `snakemake --lint` at any time to validate your configuration against the
JSON Schema (`workflow/schemas/config.schema.yaml`).

---

## Minimal working configuration

```yaml
sample_sheets:
  raw_data: path/to/samples.tsv   # at least one entry point required

mode: custom
msa_option: auto
n_samples: 5

output_dir: results

af3_flags:
  af3_container: /path/to/alphafold3.sif   # required
  models_dir:    /path/to/af3_weights      # required
  databases_dir: /path/to/af3_databases    # required (~2 TB)
```

---

## Parameters

### Top-level

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mode` | string | — **(required)** | Run mode: `custom`, `all-vs-all`, `pulldown`, `virtual-drug-screen`, `stoichio-screen`. See [docs/input.md](../docs/input.md). |
| `output_dir` | string | `results` | Root directory for all workflow outputs. |
| `tmp_dir` | string | `/tmp/af3` | Scratch directory for AlphaFold3 temporary files. |
| `n_seeds` | integer ≥ 1 | — | Number of random seeds per job. Overrides the `model_seeds` column in the sample sheet. |
| `n_samples` | integer ≥ 1 | `5` | Number of diffusion samples per seed. |
| `msa_option` | string | `auto` | MSA strategy: `auto` (run MSA), `none` (skip MSA), `upload` (use pre-computed MSAs). |
| `exclusive_lock` | boolean | `false` | Request exclusive node allocation for inference jobs (SLURM). Use on clusters that allocate whole nodes. |
| `run_data_pipeline_locally` | boolean | `false` | Run the CPU-bound data pipeline on the submission node instead of via SLURM. |
| `n_node_splits` | integer ≥ 1 | `1` | Number of node splits for parallel inference dispatch (exclusive-lock mode). |
| `run_ost_scoring` | boolean | `false` | Run OpenStructure scoring after inference. |
| `ground_truth_dir` | string | `""` | Directory containing ground-truth structures for OST scoring. |
| `n_scoring_splits` | integer ≥ 1 | `4` | Number of parallel splits for OST scoring. |
| `predict_individual_components` | boolean | `false` | Also predict individual monomer components of each multimer job. |

### `sample_sheets`

Any combination of the four entry points may be provided simultaneously.
Each stream is processed independently and all results are collected into a
single final target. At least one entry point must be non-empty.

| Key | Description |
|---|---|
| `raw_data` | TSV with full job specification. Triggers the complete pipeline (MSA → merge → inference). See [docs/input.md](../docs/input.md) for column definitions. |
| `data_pipeline_ready` | TSV pointing to pre-computed monomer JSONs (`sample_id`, `file`). Skips MSA generation. |
| `merge_ready` | TSV pointing to pre-merged multimer JSONs (`sample_id`, `multimer_file`, `monomer_chain_id`, `monomer_file`). Skips MSA and merge steps. |
| `inference_ready` | TSV pointing to final AF3 input JSONs (`sample_id`, `file`). Runs inference only. |

**Entry-point pipeline stages:**

```
raw_data ──► MSA (data pipeline) ──► merge ──► inference
                                        ▲
data_pipeline_ready ────────────────────┘
                                                  ▲
merge_ready ──────────────────────────────────────┘
                                                  ▲
inference_ready ──────────────────────────────────┘
```

### `af3_flags`

| Key | Required | Description |
|---|---|---|
| `af3_container` | **yes** | Path to the AlphaFold3 Singularity/Apptainer image (`.sif`). |
| `models_dir` | **yes** | Path to AlphaFold3 model weights directory. |
| `databases_dir` | **yes** | Path to genetic databases directory (~2 TB). |
| `tmp_dir` | no | Scratch space for AF3 temp files inside the container. |
| `extra_af3_flags` | no | Additional flags passed verbatim to `run_alphafold.py`. Useful for specifying sharded database paths or tuning CPU counts for jackhmmer/nhmmer. |

---

## HPC profiles

The workflow ships with SLURM and local profiles under `workflow/profiles/`.
Select a profile with `--workflow-profile workflow/profiles/slurm` (or `local`).

For clusters that allocate whole nodes, set `exclusive_lock: true` and
`n_node_splits` to the number of GPUs per node. The workflow will batch
inference jobs and dispatch them with GNU parallel.

---

## Further reading

- [docs/input.md](../docs/input.md) — sample sheet column reference and mode descriptions
- [docs/output.md](../docs/output.md) — output directory structure
- [workflow/schemas/config.schema.yaml](../workflow/schemas/config.schema.yaml) — JSON Schema for this file
