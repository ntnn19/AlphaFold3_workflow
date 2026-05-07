# AlphaFold3 Workflow — Inputs & Configuration

---

## Overview

The workflow accepts two kinds of inputs:

1. **A sample sheet** — a TSV file describing the molecular entities to predict.
2. **A configuration file** (`config/config.yaml`) — controls pipeline mode, resources, and paths.

Depending on how far along the pipeline you are, you can enter at one of four stages by providing the corresponding sample sheet key under `sample_sheets:` in the config.

---

## 1. Configuration File (`config.yaml`)

All parameters are read via `config.get(...)` in the Snakefile. A minimal working config:

```yaml
sample_sheets:
  raw_data: example/custom.tsv   # path to your sample sheet

output_dir: results              # where all outputs are written
mode: custom                     # run mode (see §2)
msa_option: auto                 # auto | none | upload
n_seeds: 3                       # number of random seeds per job

af3_flags:
  --af3_container: /path/to/alphafold3.sif   # required
```

### 1.1 Sample Sheet Keys

```yaml
sample_sheets:
  raw_data:            <path>   # Entry point 1: raw molecular entities (TSV)
  data_pipeline_ready: <path>   # Entry point 2: pre-computed MSA/data-pipeline outputs (TSV)
  inference_ready:     <path>   # Entry point 3: merged multimer JSONs ready for inference (TSV)
  merge_ready:         <path>   # Entry point 4: multimer+monomer mapping for merging step (TSV)
```

Only one entry point needs to be provided. The workflow detects which sheets are present and skips upstream rules accordingly.

### 1.2 Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | string | `"results"` | Root directory for all rule outputs |
| `tmp_dir` | string | — | Directory for AlphaFold 3 temporary files |
| `mode` | string | `"custom"` | Run mode (see §2) |
| `msa_option` | string | `"auto"` | Global MSA strategy: `auto`, `none`, or `upload` |
| `n_seeds` | integer | `null` | Number of random seeds. Overrides `model_seeds` column in sample sheet when set |
| `n_samples` | integer | `null` | Number of models per seed (used for massive sampling) |
| `n_splits` | integer | `1` | Number of parallel inference job splits (set to number of multi-GPU nodes) |
| `exclusive_lock` | bool | `false` | When `true`, jobs are batched and dispatched via a job-list file rather than one-per-rule |
| `predict_individual_components` | bool | `false` | Also predict each monomer chain individually from multimeric jobs |
| `run_data_pipeline_locally` | bool | `false` | Run `AF3_DATA_SPEEDY_PIPELINE` as a local rule (no cluster submission) |
| `run_inference_locally` | bool | `false` | Run `AF3_INFERENCE` as a local rule |




### 1.3 AlphaFold 3 Container & Flags

```yaml
af3_flags:
  --af3_container: /path/to/alphafold3.sif   # Required: Singularity image for AF3
  --extra_af3_flags: ""   # Optional: extra flags passed verbatim to run_alphafold.py
```

`--af3_container` is **required**. It is used as the `container:` directive for the `AF3_DATA_SPEEDY_PIPELINE` and `AF3_INFERENCE` rules.

### 1.4 Seed Resolution Logic

The Snakefile resolves seeds in this priority order:

1. `n_seeds` in config → passed as `--n-seeds=<N>` to `preprocessing.py`, overrides sample sheet
2. `model_seeds` column present and non-empty in sample sheet → sample sheet drives seeds, flag not passed
3. Neither → falls back to `--n-seeds=1`

---

## 2. Run Modes

Set via `mode:` in `config.yaml`. Determines how `preprocessing.py` combines entities into jobs. For per-mode config examples and run commands, see the [README](../README.md#usage).

| Mode | Description |
|------|-------------|
| `custom` | Predict each job as defined in the sample sheet. Multimers and monomers are handled separately. |
| `all-vs-all` | Every pair of jobs (including self-pairs) is combined into a multimeric prediction using `itertools.combinations_with_replacement`. |
| `pulldown` | Cross-group pairings only: jobs labelled `bait` are paired with jobs labelled `target`. Requires `bait_or_target` column. |
| `virtual-drug-screen` | Ligand-centric screen. Uses a compact VDS sample sheet format (see §3.4). Entities are expanded from `count` column. |
| `stoichio-screen` | Stoichiometry screen. The `count` column accepts a range `"start,end"` and all Cartesian combinations of stoichiometries are generated. |

---

## 3. Sample Sheet Formats

All sample sheets are **tab-separated** (TSV). The schema for each entry point is defined in `SAMPLE_SHEET_SCHEMAS` in the Snakefile.

### 3.1 `raw_data` — Full Entity Sheet

Used with `mode: custom`, `all-vs-all`, `pulldown`, `stoichio-screen`.

**Required columns:**

| Column | Type | Description |
|--------|------|-------------|
| `job_name` | string | Unique job identifier. Sanitised to lowercase alphanumeric + `_-.` |
| `type` | string | Entity type: `protein`, `rna`, `dna`, or `ligand` |
| `id` | string | Chain identifier (e.g. `A`, `B`, `C`). Must be unique within a job |
| `sequence` | string | Amino acid / nucleotide sequence (one-letter code). Leave empty for ligands |

**Optional columns:**

| Column | Type | Description |
|--------|------|-------------|
| `modifications` | JSON string | List of modification dicts. For proteins: `[{"ptmType": "HY3", "ptmPosition": 1}]`. For DNA/RNA: `[{"modificationType": "6OG", "basePosition": 1}]` |
| `ccd_codes` | string | CCD code(s) for ligands, comma-separated (e.g. `ATP` or `NAG,FUC`). Used when `type=ligand` and value does not contain SMILES characters |
| `smiles` | string | SMILES string for ligands (e.g. `CC(=O)OC1C[NH+]2CCC1CC2`). Used when `type=ligand` and value contains `=`, `#`, `(`, `)`, or digits |
| `msa_option` | string | Per-entity MSA strategy: `auto` (default), `none`, or `upload` |
| `unpaired_msa` | string | Path to unpaired MSA file (A3M format). Required when `msa_option=upload` |
| `paired_msa` | string | Path to paired MSA file. Used when `msa_option=upload` |
| `templates` | string or JSON | Template specification. `null`/omitted = auto search; `[]` = template-free; JSON list of template dicts = custom templates; `"path/to/file.cif,CHAIN"` = path+chain for `prepare_af3_templates` |
| `model_seeds` | string | Comma-separated integer seeds (e.g. `"10,42"`). Ignored when `n_seeds` is set in config |
| `bonded_atom_pairs` | JSON string | List of bonded atom pair definitions, e.g. `[[[\"A\",1,\"CA\"],[\"G\",1,\"CHA\"]]]` |
| `user_ccd` | string | Custom CCD definition string |
| `roi` | string | Region of interest in `"start,end"` format (1-based, inclusive). Slices the sequence before processing. Applies to `protein`, `rna`, `dna` only |

**Pulldown-only column:**

| Column | Type | Description |
|--------|------|-------------|
| `bait_or_target` | string | `bait` or `target`. Jobs from different groups are paired; same-group pairs are skipped |

**Deduplication:** `preprocessing.py` hashes each job's entity content and removes exact duplicates before writing fold inputs. A summary is written to `<output_dir>/rule_PREPROCESSING/metadata/duplicate_job_summary.json`.

**Example** (`example/custom.tsv`):
```
job_name  type     id  sequence        modifications                                                    msa_option  model_seeds  bonded_atom_pairs
job1      protein  A   PVLSCGEWQL      [{"ptmType":"HY3","ptmPosition":1},{"ptmType":"P1L","ptmPosition":5}]  auto        10,42        [[[...],[...]]]
job1      protein  B   RPACQLW                                                                          auto        10,42        [[[...],[...]]]
job1      dna      C   GACCTCT         [{"modificationType":"6OG","basePosition":1}]                   auto        10,42        ...
job1      rna      E   AGCU            [{"modificationType":"2MG","basePosition":1}]                   auto        10,42        ...
job1      ligand   F                                                                    ATP              auto        10,42        ...
job1      ligand   Z                                                CC(=O)OC1C[NH+]2CCC1CC2             auto        10,42        ...
```

### 3.2 `data_pipeline_ready` — Pre-computed MSA Sheet

Used to skip the `AF3_DATA_SPEEDY_PIPELINE` rule when MSAs have already been computed.

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | Job name (stem of the JSON file) |
| `file` | string | Path to the monomer fold-input JSON (output of a previous data pipeline run) |

### 3.3 `inference_ready` — Pre-merged Multimer Sheet

Used to skip both `AF3_DATA_SPEEDY_PIPELINE` and `MERGE_MONO_AND_MULTI_JSON` when merged multimer JSONs are already available.

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | Job name |
| `file` | string | Path to the merged multimer `_data.json` file |

### 3.4 `merge_ready` — Multimer-to-Monomer Mapping Sheet

Used to skip `PREPROCESSING` and `AF3_DATA_SPEEDY_PIPELINE` and go directly to `MERGE_MONO_AND_MULTI_JSON`.

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | Multimer job name |
| `multimer_file` | string | Path to the multimer template JSON |
| `monomer_chain_id` | string | Chain ID within the multimer |
| `monomer_file` | string | Path to the corresponding monomer `_data.json` (with MSA) |

This sheet is auto-generated by `preprocessing.py` at `<output_dir>/rule_PREPROCESSING/metadata/inference_to_data_pipeline_map.tsv` and can be reused directly.

### 3.5 `virtual-drug-screen` Format

A compact format used exclusively with `mode: virtual-drug-screen`. Transformed internally by `transform_vds_to_af3()` before processing.

| Column | Type | Description |
|--------|------|-------------|
| `job_name` | string | Job identifier |
| `type` | string | `protein`, `rna`, `dna`, or `ligand` |
| `data` | string | Sequence (for polymers) or CCD code / SMILES (for ligands) |
| `count` | integer | Number of copies of this entity in the complex |
| `user_ccd` | string | Optional custom CCD definition |
| `model_seeds` | string | Optional comma-separated seeds |
| `msa_option` | string | Optional MSA strategy |
| `modifications` | JSON string | Optional modifications |
| `bonded_atom_pairs` | JSON string | Optional bonded atom pairs |

Ligand type detection heuristic: if `data` contains any of `=`, `#`, `(`, `)`, `1`, `2`, `3` it is treated as SMILES; otherwise as a CCD code.

**Example** (`example/virtual_drug_screen.tsv`):
```
job_name    type     data                                   count
vls_run_1   protein  MSTVTTINLEDIKEIMH...SLVRSSLVLNG        2
vls_run_1   ligand   9LC                                    1
```

### 3.6 `stoichio-screen` Format

Same columns as `raw_data`, but the `count` column accepts either a fixed integer (`"2"`) or a range (`"1,5"`). All Cartesian combinations of stoichiometries across entities are generated. A summary CSV is written to `<output_dir>/rule_PREPROCESSING/metadata/stoichio_screen.csv`.

---

## 4. MSA Options (per entity)

Controlled by the `msa_option` column (or global `msa_option` config key).

| Value | Behaviour |
|-------|-----------|
| `auto` | `unpairedMsa` and `pairedMsa` set to `null` — AlphaFold 3 builds MSAs automatically |
| `none` | `unpairedMsa` and `pairedMsa` set to `""` — completely MSA-free prediction |
| `upload` | Custom MSA provided via `unpaired_msa` and `paired_msa` columns (paths to A3M files) |

For RNA: only `unpairedMsa` is used. For DNA: no MSA fields are set.

---

## 5. Template Options (proteins only)

Controlled by the `templates` column.

| Value | Behaviour |
|-------|-----------|
| omitted / `null` | Auto template search by AlphaFold 3 |
| `[]` (empty JSON array) | Template-free prediction |
| JSON list of dicts | Custom template dicts, e.g. `[{"mmcif": "", "queryIndices": [...], "templateIndices": [...]}]` |
| `"path/to/structure.cif,CHAIN"` | Path + chain string; processed by `prepare_af3_templates` to generate aligned template dicts |

---

## 6. Fold Input JSON Structure

`preprocessing.py` writes AlphaFold 3 fold-input JSON files with the following top-level structure:

```json
{
  "name": "<job_name>",
  "modelSeeds": [1, 2, 3],
  "sequences": [
    {"protein": {"sequence": "...", "id": "A", "unpairedMsa": null, "pairedMsa": null}},
    {"rna":     {"sequence": "...", "id": "B", "unpairedMsa": null}},
    {"dna":     {"sequence": "...", "id": "C"}},
    {"ligand":  {"ccdCodes": ["ATP"], "id": "D"}}
  ],
  "dialect": "alphafold3",
  "version": 1,
  "bondedAtomPairs": [...],   // optional
  "userCCD": "..."            // optional
}
```

Multimer jobs are written with one file per seed: `<job_name>_seed-<N>.json`.  
Monomer jobs (for the data pipeline) are written as a single file: `<job_name>.json`.

---

## 7. Required External Resources

The following paths must be accessible inside the AlphaFold 3 Singularity container (mounted at the paths expected by `run_alphafold.py`):

| Resource | Container path |
|----------|---------------|
| AF3 model weights | `/root/models` |
| Genetic databases | `/root/public_databases` |
| AF3 inference output | `/root/af_output/rule_AF3_INFERENCE` |
| AF3 data pipeline output | `/root/af_output/rule_AF3_DATA_PIPELINE` |

The Singularity image path is set via `af3_flags: --af3_container:` in `config.yaml`.
