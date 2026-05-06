# AlphaFold3 Workflow — Outputs


---

## Overview

All outputs are written under `output_dir` (set in `config.yaml`, default `"results"`). Each Snakemake rule writes to its own subdirectory prefixed `rule_<RULENAME>/`. The pipeline stages are:

```
PREPROCESSING
    └── AF3_DATA_SPEEDY_PIPELINE  (data pipeline / MSA generation)
            └── MERGE_MONO_AND_MULTI_JSON  (inject MSAs into multimer JSONs)
                    └── AF3_INFERENCE  (structure prediction)
                            └── (optional) OST scoring
```

---

## Directory Structure

```
<output_dir>/
├── rule_PREPROCESSING/
│   ├── monomers/
│   │   └── <job_name>.json
│   ├── multimers/
│   │   └── <job_name>_seed-<N>.json
│   └── metadata/
│       ├── duplicate_job_summary.json
│       ├── data_pipeline_samples.tsv
│       ├── inference_samples.tsv
│       ├── inference_to_data_pipeline_map.tsv
│       └── stoichio_screen.csv              # stoichio-screen mode only
│
├── rule_AF3_DATA_PIPELINE/
│   └── <mono_job_name>/
│       └── <mono_job_name>_data.json
│
├── rule_MERGE_MONOMERS_TO_MULTIMERS/
│   └── <multimer_job_name>_data.json
│
├── rule_CREATE_AF3_INFERENCE_JOBS/          # exclusive_lock mode only
│   └── <job_name>_af3_inference_job.txt
│
├── rule_AGG_AF3_INFERENCE_JOBS/             # exclusive_lock mode only
│   └── af3_inference_jobs.txt
│
├── rule_SPLIT_INFERENCE_JOB_LIST/           # exclusive_lock mode only
│   └── af3_inference_jobs_<N>-of-<SPLIT_TOTAL>.txt
│
├── rule_AF3_INFERENCE/
│   ├── done_flags/
│   │   └── af3_inference_jobs_<N>-of-<SPLIT_TOTAL>.done.txt   # exclusive_lock mode
│   └── <job_name>/
        └── <job_name>_model.cif                                # non-exclusive mode
```

---

## Rule-by-Rule Output Reference

### `PREPROCESSING` (checkpoint)

**Output directory:** `<output_dir>/rule_PREPROCESSING/`

Produced by `workflow/scripts/preprocessing.py`. Converts the raw sample sheet into AlphaFold 3 fold-input JSON files and three metadata TSVs.

#### Fold-input JSONs

| Path | Description |
|------|-------------|
| `rule_PREPROCESSING/monomers/<job_name>.json` | Single-chain fold-input JSON for each monomer (protein or RNA chain extracted from a multimeric job, or an independent monomer). Fed into the data pipeline. |
| `rule_PREPROCESSING/multimers/<job_name>_seed-<N>.json` | Multi-chain fold-input JSON for each multimeric job, one file per seed. Used as the structural template for merging. |

Naming conventions:
- Multimer files: `<job_name>_seed-<N>.json`
- Independent monomer files (when `predict_individual_components` is set or job has no multimer partner): `<job_name>_seed-<N>_chain-<id>.json`
- Monomer chain files derived from multimers: `<job_name>_chain-<id>.json`

Job names are sanitised: lowercased, spaces replaced with `_`, only `[a-z0-9_-.]` retained.

#### Metadata Files

| File | Description |
|------|-------------|
| `metadata/duplicate_job_summary.json` | Summary of duplicate jobs detected and removed. Fields: `total_jobs`, `unique_jobs`, `duplicate_jobs`, `duplicate_groups`, `group_size_distribution`, `largest_groups`, `sample_duplicates`. A `_full_mapping.txt.gz` companion is written when >100 duplicates are found. |
| `metadata/data_pipeline_samples.tsv` | Sample sheet for the `AF3_DATA_SPEEDY_PIPELINE` rule. Columns: `file` (path to monomer JSON in `rule_PREPROCESSING/monomers/`), `sample_id` (stem of the file), `expected_output` (expected `_data.json` path in `rule_AF3_DATA_PIPELINE/`). |
| `metadata/inference_samples.tsv` | Sample sheet for the `AF3_INFERENCE` rule. Columns: `sample_id`, `file` (path to merged multimer `_data.json` in `rule_MERGE_MONOMERS_TO_MULTIMERS/`), `expected_output` (expected CIF path in `rule_AF3_INFERENCE/`). Rows are expanded: one row per (job × seed × sample) combination. |
| `metadata/inference_to_data_pipeline_map.tsv` | Mapping from multimer inference files to their constituent monomer data-pipeline files. Columns: `multimer_file`, `monomer_chain_id`, `monomer_file`, `sample_id`. Used by `MERGE_MONO_AND_MULTI_JSON`. |
| `metadata/stoichio_screen.csv` | *(stoichio-screen mode only)* Summary of all stoichiometry combinations generated. Columns: `job_name`, `parent_job`, `monomer_1`, `monomer_2`, ..., `monomer_N`, `monomer_1_prefix`, ... |

---

### `AF3_DATA_SPEEDY_PIPELINE`

**Output:** `<output_dir>/rule_AF3_DATA_PIPELINE/<mono_job_name>/<mono_job_name>_data.json`

Runs `run_alphafold.py` with `--run_data_pipeline=true --run_inference=false` inside the AF3 Singularity container. Produces a monomer fold-input JSON enriched with MSA and template data. One output directory per monomer chain.

The input is either:
- `rule_PREPROCESSING/monomers/<mono>.json` (from `raw_data` entry point), or
- The file listed in `data_pipeline_ready` sample sheet (skip-ahead entry point).

---

### `MERGE_MONO_AND_MULTI_JSON`

**Output:** `<output_dir>/rule_MERGE_MONOMERS_TO_MULTIMERS/<multimer_job_name>_data.json`

Runs `workflow/scripts/merge_mono_and_multi_jsons.py`. Injects the per-chain MSA data from the monomer `_data.json` files into the multimer template JSON, producing a complete multimer fold-input ready for inference.

One output file per multimer job (across all seeds, since seeds are encoded in the job name at this stage).

---

### `CREATE_AF3_INFERENCE_JOBS` *(exclusive_lock mode only)*

**Output:** `<output_dir>/rule_CREATE_AF3_INFERENCE_JOBS/<job_name>_af3_inference_job.txt`

A plain-text file containing the `run_alphafold.py` command for this job. Used to batch all inference commands before parallel dispatch.

---

### `AGG_AF3_INFERENCE_JOBS` *(exclusive_lock mode only)*

**Output:** `<output_dir>/rule_AGG_AF3_INFERENCE_JOBS/af3_inference_jobs.txt`

Concatenation of all `*_af3_inference_job.txt` files into a single job list.

---

### `SPLIT_INFERENCE_JOB_LIST` *(exclusive_lock mode only)*

**Output:** `<output_dir>/rule_SPLIT_INFERENCE_JOB_LIST/af3_inference_jobs_<N>-of-<SPLIT_TOTAL>.txt`

The aggregated job list split into `n_splits` chunks for parallel dispatch across multi-GPU nodes.

---

### `AF3_INFERENCE`

**Two output modes depending on `exclusive_lock`:**

| Mode | Output |
|------|--------|
| `exclusive_lock: false` | `<output_dir>/rule_AF3_INFERENCE/<job_name>/<job_name>_model.cif` — one CIF structure per job |
| `exclusive_lock: true` | `<output_dir>/rule_AF3_INFERENCE/done_flags/af3_inference_jobs_<N>-of-<SPLIT_TOTAL>.done.txt` — touch file per split; actual CIF files are written by AF3 to `/root/af_output/rule_AF3_INFERENCE/` inside the container |

Runs `run_alphafold.py` with `--run_data_pipeline=false --run_inference=true`. Automatically detects GPU compute capability and disables flash attention for pre-Ampere GPUs (`CC < 8`). In `exclusive_lock` mode, jobs within a split are dispatched in parallel using GNU `parallel`, one job per GPU.

---


## Output File Naming Conventions

| Pattern | Meaning |
|---------|---------|
| `<job_name>` | Sanitised job name: lowercase, `[a-z0-9_-.]` only |
| `_seed-<N>` | Seed index (integer, 1-based by default) |
| `_sample-<M>` | Sample index within a seed (1-based) |
| `_chain-<id>` | Chain letter (lowercase) for per-chain monomer files |
| `_data.json` | Fold-input JSON enriched with MSA/template data (post data-pipeline) |
| `_model.cif` | Predicted structure in mmCIF format |
| `_af3_inference_job.txt` | Shell command file for one inference job |
| `.done.txt` | Touch file indicating rule completion |

---
