# AlphaFold3 Workflow — Outputs

---

## Overview

All outputs are written under `output_dir` (set in `config.yaml`, default `"results"`). Each Snakemake rule writes to its own subdirectory prefixed `rule_<RULENAME>/`. The pipeline stages are:

```
PREPROCESSING
    └── AF3_DATA_SPEEDY_PIPELINE  (data pipeline / MSA generation)
            └── MERGE_MONO_AND_MULTI_JSON  (inject MSAs into multimer JSONs)
                    └── AF3_INFERENCE  (structure prediction)
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
├── rule_AF3_INFERENCE/
    └── <job_name>/
        ├── <job_name>_model.cif
        └── ... # all other standard AlphaFold3 outputs                                
```

---

#### Metadata Files

| File | Description |
|------|-------------|
| `metadata/duplicate_job_summary.json` | Summary of duplicate jobs detected and removed. Fields: `total_jobs`, `unique_jobs`, `duplicate_jobs`, `duplicate_groups`, `group_size_distribution`, `largest_groups`, `sample_duplicates`. A `_full_mapping.txt.gz` companion is written when >100 duplicates are found. |
| `metadata/data_pipeline_samples.tsv` | Sample sheet for the `AF3_DATA_SPEEDY_PIPELINE` rule. Columns: `file` (path to monomer JSON in `rule_PREPROCESSING/monomers/`), `sample_id` (stem of the file), `expected_output` (expected `_data.json` path in `rule_AF3_DATA_PIPELINE/`). |
| `metadata/inference_samples.tsv` | Sample sheet for the `AF3_INFERENCE` rule. Columns: `sample_id`, `file` (path to merged multimer `_data.json` in `rule_MERGE_MONOMERS_TO_MULTIMERS/`), `expected_output` (expected CIF path in `rule_AF3_INFERENCE/`). Rows are expanded: one row per (job × seed × sample) combination. |
| `metadata/inference_to_data_pipeline_map.tsv` | Mapping from multimer inference files to their constituent monomer data-pipeline files. Columns: `multimer_file`, `monomer_chain_id`, `monomer_file`, `sample_id`. Used by `MERGE_MONO_AND_MULTI_JSON`. |
| `metadata/stoichio_screen.csv` | *(stoichio-screen mode only)* Summary of all stoichiometry combinations generated. Columns: `job_name`, `parent_job`, `monomer_1`, `monomer_2`, ..., `monomer_N`, `monomer_1_prefix`, ... |

## Output File Naming Conventions

| Pattern | Meaning |
|---------|---------|
| `<job_name>` | Sanitised job name: lowercase, `[a-z0-9_-.]` only |
| `_seed-<N>` | Seed index (integer, 1-based by default) |
| `_sample-<M>` | Sample index within a seed (1-based) |
| `_chain-<id>` | Chain letter (lowercase) for per-chain monomer files |
| `_data.json` | Fold-input JSON enriched with MSA/template data (post data-pipeline) |
| `_model.cif` | Predicted structure in mmCIF format |

---
