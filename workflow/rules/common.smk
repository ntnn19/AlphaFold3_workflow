"""
common.smk — shared helpers, config validation, and wildcard resolution.

All rules import this module via `include: "rules/common.smk"` in the
main Snakefile.  Nothing here emits jobs; it only defines Python-level
constants and input functions used by the rule files.
"""


import os
import re
from pathlib import Path
from typing import Any, Iterable
import pandas as pd
import resource
from snakemake.utils import validate, min_version
from snakemake.exceptions import WorkflowError
from itertools import product
import json
from typing import (
    Any,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Union,
    Literal,
    Tuple
)
# ── Snakemake version guard ──────────────────────────────────────────────────
min_version("8.0")

# ── Config validation ────────────────────────────────────────────────────────
validate(config, schema="../schemas/config.schema.yaml")

# ── Top-level constants ──────────────────────────────────────────────────────
OUTPUT_DIR   = config.get("output_dir", "results")
TMP_DIR      = config.get("tmp_dir", "tmp/af3")
MODE         = config["mode"]
N_SEEDS      = config.get("n_seeds")
N_SAMPLES    = config.get("n_samples", 5)
MSA_OPTION   = config.get("msa_option", "auto")
EXCLUSIVE_LOCK    = config.get("exclusive_lock", False)
N_SPLITS     = config.get("n_node_splits", 1)
RUN_OST      = config.get("run_ost_scoring", False)
GT_DIR       = config.get("ground_truth_dir", "")
N_OST_SPLITS = config.get("n_scoring_splits", 4)
AF3_CONTAINER   = config["af3_flags"]["af3_container"]
EXTRA_AF3_FLAGS = config["af3_flags"].get("--extra_af3_flags", config["af3_flags"].get("extra_af3_flags", ""))
MODELS_DIR   = config["af3_flags"]["models_dir"]
DB_DIR       = config["af3_flags"]["databases_dir"]
PREDICT_INDIVIDUAL_COMPONENTS = '--predict-individual-components' if config.get('predict_individual_components', False) else ''
AF3_VERSION = config.get("af3_version","v3.0.2")
print("AF3_VERSION=", AF3_VERSION)
# ── Utility ──────────────────────────────────────────────────────────────────
_ALLOWED = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-.")
NORMALIZED_INPUTS_DIR = os.path.join(OUTPUT_DIR, "normalized_inputs")
os.makedirs(NORMALIZED_INPUTS_DIR, exist_ok=True)
# A4: flatten a list-of-lists into a flat list
def flatten(lst):
    """Flatten one level of nesting from a list of lists."""
    return [item for sublist in lst for item in sublist]


SAMPLE_SHEET_SCHEMAS = {
    "raw_data": [
        "job_name", "type", "id", "sequence", "modifications", "ccd_codes",
        "smiles", "msa_option", "unpaired_msa", "paired_msa", "templates",
        "model_seeds", "bonded_atom_pairs", "user_ccd"
    ],
    "data_pipeline_ready": ["sample_id", "file"],
    "inference_ready": ["sample_id", "file"],
    "merge_ready_samples": ["sample_id", "multimer_file", "monomer_chain_id", "monomer_file"],
    "mutations": ["sample_id", "type", "id", "mutation"],
    "scoring_ready": ["sample_id", "type", "model", "confidences"],
}

def sanitise(name: str) -> str:
    """Lowercase, replace spaces, strip non-alphanumeric chars."""
    return "".join(c for c in name.lower().replace(" ", "_") if c in _ALLOWED)


def _seed_tag(seed: int) -> str:
    return f"seed-{seed}"


def _job_seed(job: str, seed: int) -> str:
    return f"{job}_{_seed_tag(seed)}"


# ── Sample-sheet loading ─────────────────────────────────────────────────────

def load_sample_sheet(sheet_key):
    """Safely load a sample sheet, returns (path, dataframe)."""
    path = config.get("sample_sheets", {}).get(sheet_key)
    columns = SAMPLE_SHEET_SCHEMAS.get(sheet_key, [])

    if not path:
        return None, pd.DataFrame(columns=columns)

    if not Path(path).exists():
        return path, pd.DataFrame(columns=columns)

    try:
        df = pd.read_csv(path, sep="\t")
        return path, df
    except Exception as e:
        raise WorkflowError(
            f"Failed to load sample sheet '{path}': {e}"
        ) from e

# Reverse / round-trip utility
#
# main() consumes a "sample sheet" (read at pd.read_csv(sample_sheet, sep="\t"))
# whose rows describe AF3 *entities* (job_name, type, id, sequence, ...) and
# expands it into AF3 fold-input JSON files plus three bookkeeping TSVs:
#   * data_pipeline_samples.tsv / inference_samples.tsv   -> columns: sample_id, file, ...
#   * inference_to_data_pipeline_map.tsv                  -> columns: multimer_file,
#                                                            monomer_chain_id, monomer_file, sample_id
#
# The helpers below invert that process: given one of those TSVs (which list AF3
# JSON files), they read the referenced JSON(s) and rebuild a sample sheet in the
# exact schema main() expects, so the result can be fed straight back into line
# `df = pd.read_csv(sample_sheet, sep="\t")`.
# ---------------------------------------------------------------------------

# Canonical column order of the sample sheet consumed by main().
SAMPLE_SHEET_COLUMNS = [
    "job_name",
    "type",
    "id",
    "sequence",
    "modifications",
    "msa_option",
    "unpaired_msa",
    "paired_msa",
    "templates",
    "ccd_codes",
    "smiles",
    "model_seeds",
    "bonded_atom_pairs",
    "user_ccd",
]


def _json_dump_field(value):
    """Serialise a list/dict field (modifications, templates, bonded_atom_pairs)
    back to the compact JSON string form that parse_json_field() expects.

    None is returned unchanged (-> becomes an empty cell in the TSV); a bare
    string is passed through (parse_json_field also accepts raw strings, e.g. a
    template path)."""
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def _msa_option_from_entry(entry: Mapping[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
    """Recover (msa_option, unpaired_msa, paired_msa) from a polymer entry.

    Mirrors create_protein_sequence_data / create_rna_sequence_data:
      * field absent or null  -> "auto"   (AF3 builds the MSA)
      * field == ""           -> "none"   (MSA-free)
      * field is a non-empty  -> "upload" (custom MSA carried through)
    Protein entries may also carry unpairedMsaPath / pairedMsaPath (the 'upload'
    branch); those are preferred as the round-tripped MSA payload when present.
    """
    if "unpairedMsa" not in entry and "unpairedMsaPath" not in entry:
        # e.g. DNA entries, or a ligand-like polymer with no MSA keys at all.
        return "auto", None, None

    unpaired = entry.get("unpairedMsaPath", entry.get("unpairedMsa"))
    paired = entry.get("pairedMsaPath", entry.get("pairedMsa"))

    if unpaired is None:
        return "auto", None, None
    if unpaired == "":
        return "none", None, None
    return "upload", unpaired, (None if paired in (None, "") else paired)


def af3_json_to_sample_rows(task: Mapping[str, Any]) -> list:
    """Invert create_batch_task(): turn one parsed AF3 fold-input JSON into a
    list of sample-sheet row dicts (one per entity/chain), preserving the order
    of the ``sequences`` array."""
    job_name = task.get("name")

    model_seeds = task.get("modelSeeds")
    if isinstance(model_seeds, (list, tuple)):
        model_seeds = ",".join(str(s) for s in model_seeds)
    elif model_seeds is not None:
        model_seeds = str(model_seeds)

    bonded_atom_pairs = _json_dump_field(task.get("bondedAtomPairs"))
    user_ccd = task.get("userCCD")

    rows = []
    for seq_wrapper in task.get("sequences", []):
        # Each element is a single-key dict: {"protein"|"rna"|"dna"|"ligand": {...}}
        if not isinstance(seq_wrapper, Mapping) or len(seq_wrapper) != 1:
            raise ValueError(f"Unexpected sequence entry in job {job_name!r}: {seq_wrapper!r}")
        (entity_type, entry), = seq_wrapper.items()

        row = {c: None for c in SAMPLE_SHEET_COLUMNS}
        row["job_name"] = job_name
        row["type"] = entity_type
        row["id"] = entry.get("id")
        row["modifications"] = _json_dump_field(entry.get("modifications"))
        row["model_seeds"] = model_seeds
        row["bonded_atom_pairs"] = bonded_atom_pairs
        row["user_ccd"] = user_ccd

        if entity_type in ("protein", "rna", "dna"):
            row["sequence"] = entry.get("sequence")
            msa_option, unpaired, paired = _msa_option_from_entry(entry)
            row["msa_option"] = msa_option
            row["unpaired_msa"] = unpaired
            row["paired_msa"] = paired
            # templates: None -> auto search; [] -> template-free; list -> custom.
            if "templates" in entry:
                row["templates"] = _json_dump_field(entry.get("templates"))
        elif entity_type == "ligand":
            ccd_codes = entry.get("ccdCodes")
            if isinstance(ccd_codes, (list, tuple)):
                ccd_codes = ",".join(str(c) for c in ccd_codes)
            row["ccd_codes"] = ccd_codes
            row["smiles"] = entry.get("smiles")
        else:
            raise ValueError(
                f"Unknown entity type {entity_type!r} in job {job_name!r}. "
                "Must be 'protein', 'rna', 'dna', or 'ligand'."
            )

        rows.append(row)
    return rows


def build_sample_sheet_from_json_tsv(
        tsv_path: Union[str, Path],
        output_tsv: Optional[Union[str, Path]] = None,
        json_column: Optional[str] = None,
) -> pd.DataFrame:
    """Reconstruct the line-`pd.read_csv(sample_sheet)` sample sheet from a TSV
    that lists AF3 JSON files.

    This is the inverse of main(): it reads the AF3 fold-input JSON files
    referenced by one of the pipeline's bookkeeping TSVs and rebuilds a sample
    sheet in the schema main() consumes (SAMPLE_SHEET_COLUMNS), one row per
    entity/chain.

    Two TSV layouts are auto-detected (override with ``json_column`` if needed):

    * inference / data-pipeline sheet -> has a ``file`` column
      (as written to data_pipeline_samples.tsv / inference_samples.tsv), e.g.::

          sample_id   file
          s1          s1.json
          s2          s2.json

    * merge sheet -> has a ``multimer_file`` column
      (as written to inference_to_data_pipeline_map.tsv), e.g.::

          multimer_file  monomer_chain_id  monomer_file  sample_id
          s1.json        A                 s2.json       s1
          s1.json        B                 s2.json       s1

      Because this sheet repeats one row per chain, the referenced multimer JSON
      files are de-duplicated (preserving first-seen order) and each multimer
      JSON is reconstructed once into its constituent chains.

    :param tsv_path: path to the JSON-listing TSV.
    :param output_tsv: if given, the reconstructed sample sheet is written there
        (tab-separated, no index) so it can be fed straight back into main().
    :param json_column: force which column holds the JSON paths. If None, the
        column is auto-detected (``multimer_file`` -> else ``file``).
    :return: the reconstructed sample-sheet DataFrame.
    """
    if tsv_path is None:
        return pd.DataFrame()

    listing = pd.read_csv(tsv_path, sep="\t")

    if json_column is None:
        if "multimer_file" in listing.columns:
            json_column = "multimer_file"      # merge-ready sheet
        elif "file" in listing.columns:
            json_column = "file"               # inference / data-pipeline sheet
        else:
            raise ValueError(
                f"Could not find a JSON-path column in {tsv_path}. Expected "
                f"'multimer_file' or 'file'; got columns {list(listing.columns)}. "
                "Pass json_column= explicitly."
            )
    elif json_column not in listing.columns:
        raise ValueError(f"json_column={json_column!r} not present in {tsv_path} "
                         f"(columns: {list(listing.columns)}).")

    # De-duplicate JSON paths while preserving first-seen order (the merge sheet
    # lists the same multimer_file once per chain).
    json_paths = list(dict.fromkeys(listing[json_column].dropna().tolist()))

    rows = []
    for jp in json_paths:
        with open(jp) as fh:
            task = json.load(fh)
        rows.extend(af3_json_to_sample_rows(task))

    sample_sheet = pd.DataFrame(rows, columns=SAMPLE_SHEET_COLUMNS)

    if output_tsv is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_tsv)), exist_ok=True)
        sample_sheet.to_csv(output_tsv, sep="\t", index=False)
        logger.info(f"Reconstructed sample sheet ({len(sample_sheet)} rows from "
                    f"{len(json_paths)} JSON files) written to {output_tsv}")

    return sample_sheet


RAW_DATA_PATH, RAW_DATA_DF = load_sample_sheet("raw_data")
DATA_PIPELINE_READY_PATH, DATA_PIPELINE_READY_DF = load_sample_sheet("data_pipeline_ready")
INFERENCE_READY_PATH, INFERENCE_READY_DF = load_sample_sheet("inference_ready")
MERGE_READY_PATH, MERGE_READY_DF = load_sample_sheet("merge_ready")
MUTATION_DF_PATH, MUTATION_DF = load_sample_sheet("mutations")
SCORING_READY_PATH, SCORING_READY_DF = load_sample_sheet("scoring_ready")
# ── Sample-sheet validation ──────────────────────────────────────────────────
if not RAW_DATA_DF.empty:
    validate(RAW_DATA_DF, schema="../schemas/sample_sheet.raw_data.schema.yaml")
if not DATA_PIPELINE_READY_DF.empty:
    validate(DATA_PIPELINE_READY_DF, schema="../schemas/sample_sheet.data_pipeline_ready.schema.yaml")
if not INFERENCE_READY_DF.empty:
    validate(INFERENCE_READY_DF, schema="../schemas/sample_sheet.inference_ready.schema.yaml")
if not MERGE_READY_DF.empty:
    validate(MERGE_READY_DF, schema="../schemas/sample_sheet.merge_ready.schema.yaml")

DATA_PIPELINE_READY_DF_AS_RAW_DATA_DF = build_sample_sheet_from_json_tsv(DATA_PIPELINE_READY_PATH)
INFERENCE_READY_DF_AS_RAW_DATA_DF = build_sample_sheet_from_json_tsv(INFERENCE_READY_PATH)
MERGE_READY_DF_AS_RAW_DATA_DF = build_sample_sheet_from_json_tsv(MERGE_READY_PATH)
#SCORING_READY_DF_AS_RAW_DATA_DF = build_sample_sheet_from_json_tsv(SCORING_READY_PATH)#

RAW_DATA_DF = pd.concat([RAW_DATA_DF, DATA_PIPELINE_READY_DF_AS_RAW_DATA_DF, INFERENCE_READY_DF_AS_RAW_DATA_DF, MERGE_READY_DF_AS_RAW_DATA_DF])
RAW_DATA_PATH = os.path.join(NORMALIZED_INPUTS_DIR, "raw_data.tsv")
RAW_DATA_DF.to_csv(RAW_DATA_PATH,sep="\t",index=False)
DATA_PIPELINE_READY_DF, INFERENCE_READY_DF, MERGE_READY_DF = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
# ── Synthetic merge rows for data_pipeline_ready entries ────────────────────
# data_pipeline_ready jobs are routed through MERGE_MONO_AND_MULTI_JSON, which
# looks up each wildcard in MERGE_READY_DF.  When data_pipeline_ready is
# provided we synthesise a trivial merge row for each monomer (chain A, monomer
# file = AF3_DATA_PIPELINE output) and append it to MERGE_READY_DF.  This
# works whether or not the user also supplied a merge_ready sheet.
DATA_PIPELINE_OUTPUTS = []
if not DATA_PIPELINE_READY_DF.empty:
    _dp_synthetic = DATA_PIPELINE_READY_DF.copy()
    _dp_synthetic = _dp_synthetic.rename(columns={"file": "multimer_file"})
    _dp_synthetic["monomer_file"] = _dp_synthetic["multimer_file"].apply(
        lambda x: os.path.join(OUTPUT_DIR, "rule_AF3_DATA_PIPELINE", Path(x).stem, f"{Path(x).stem}_data.json")
    )
    _dp_synthetic["monomer_chain_id"] = "A"
    DATA_PIPELINE_OUTPUTS = _dp_synthetic["monomer_file"].tolist()
    # Append to any user-supplied merge_ready rows (or replace the empty DF)
    MERGE_READY_DF = pd.concat([MERGE_READY_DF, _dp_synthetic], ignore_index=True)


# ── Wildcard resolution helpers ──────────────────────────────────────────────
def get_preprocessing_outputs(wildcards):
    PREPROCESSING_DIR = checkpoints.PREPROCESSING.get(**wildcards).output[0]
    JOB_NAMES, = glob_wildcards(os.path.join(PREPROCESSING_DIR, "{i}.json"))
    return list(expand(os.path.join(PREPROCESSING_DIR, "{i}.json"), i=JOB_NAMES))

def get_individual_jobs(wildcards):
    # C2 fix: use rule_ prefix to match the actual rule output directory
    PREPROCESSING_DIR = checkpoints.PREPROCESSING.get(**wildcards).output[0]
    JOB_NAMES, = glob_wildcards(os.path.join(PREPROCESSING_DIR, "{i}.json"))
    return list(expand(os.path.join(OUTPUT_DIR, "rule_CREATE_AF3_INFERENCE_JOBS", "{i}_af3_inference_job.txt"), i=JOB_NAMES))

def get_data_pipeline_outputs(wildcards):
    PREPROCESSING_DIR = checkpoints.PREPROCESSING.get(**wildcards).output[0]
    JOB_NAMES, = glob_wildcards(os.path.join(PREPROCESSING_DIR, "{i}.json"))
    return list(expand(os.path.join(OUTPUT_DIR, "rule_AF3_DATA_PIPELINE", "{i}/{i}_data.json"), i=JOB_NAMES))

def get_multimeric_json_outputs(wildcards):
    PREPROCESSING_DIR = checkpoints.PREPROCESSING.get(**wildcards).output[0]
    JOB_NAMES_MULTIMERS, = glob_wildcards(os.path.join(PREPROCESSING_DIR, "multimers", "{multi}.json"))
    return [
        *expand(os.path.join(PREPROCESSING_DIR, "multimers", "{multi}.json"), multi=JOB_NAMES_MULTIMERS),
        *expand(os.path.join(OUTPUT_DIR, "rule_MERGE_MONOMERS_TO_MULTIMERS", "{multi}_data.json"), multi=JOB_NAMES_MULTIMERS),
    ]

def get_monomeric_json_outputs(wildcards):
    PREPROCESSING_DIR = checkpoints.PREPROCESSING.get(**wildcards).output[0]
    JOB_NAMES_MONOMERS, = glob_wildcards(os.path.join(PREPROCESSING_DIR, "monomers", "{mono}.json"))
    return list(expand(os.path.join(OUTPUT_DIR, "rule_AF3_DATA_PIPELINE", "{mono}/{mono}_data.json"), mono=JOB_NAMES_MONOMERS))

def get_multi_to_monomeric_dict(wildcards):
    PREPROCESSING_DIR = checkpoints.PREPROCESSING.get(**wildcards).output[0]
    map_df = pd.read_csv(os.path.join(PREPROCESSING_DIR, "metadata", "inference_to_data_pipeline_map.tsv"), sep="\t")
    return map_df

def get_multi_to_monomeric_dict_(wildcards):
    PREPROCESSING_DIR = checkpoints.PREPROCESSING.get(**wildcards).output[0]
    map_df = pd.read_csv(os.path.join(PREPROCESSING_DIR, "metadata", "inference_samples.tsv"), sep="\t")
    print(map_df)
    print(map_df.sample_id.to_list())
    return map_df.sample_id.to_list()

def get_merge_inputs(wildcards):
    """
    Get inputs for merging. Check user-provided MERGE_READY_DF first,
    otherwise use checkpoint-generated mapping.

    Returns dict with:
    - multimer_template: the multimer JSON file
    - monomer_files: list of processed monomer files (one per chain)
    """
    if not MERGE_READY_DF.empty and wildcards.multi in MERGE_READY_DF["sample_id"].values:
        multimer_rows = MERGE_READY_DF[MERGE_READY_DF["sample_id"] == wildcards.multi]
        multimer_template = multimer_rows.iloc[0]["multimer_file"]
        monomer_files = multimer_rows["monomer_file"].tolist()
        return {
            "multimer_template": multimer_template,
            "monomer_files": monomer_files,
        }

    checkpoint_output = os.path.join(
        checkpoints.PREPROCESSING.get(**wildcards).output[0],
        "metadata",
        "inference_to_data_pipeline_map.tsv",
    )
    mapping = pd.read_csv(checkpoint_output, sep="\t")
    multimer_rows = mapping[mapping["multimer_file"].str.contains(wildcards.multi)]
    monomers = multimer_rows["monomer_file"].tolist()
    multimer_template = os.path.join(
        OUTPUT_DIR,
        "rule_PREPROCESSING",
        "multimers",
        f"{wildcards.multi}.json",
    )
    return {
        "multimer_template": multimer_template,
        "monomer_files": monomers,
    }


# ── Inference target collection ──────────────────────────────────────────────
def get_seeds(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data["modelSeeds"]  # adjust key if your AF3 json uses a different field

def _collect_inference_targets(wildcards, *, use_lock: bool) -> list:
    """
    Shared implementation for inference_outputs() and get_multimeric_json_with_msas().

    Collects all final inference targets from both external entry-point DataFrames
    (data_pipeline_ready, merge_ready, inference_ready) and checkpoint-generated
    multimer lists (raw_data path).

    All four entry points are independent streams and can be combined freely.
    Each stream contributes its own set of inference targets to the DAG.

    :param wildcards: Snakemake wildcards object.
    :param use_lock: When True, return lock-file targets instead of .cif files
                     for the raw_data path (used when EXCLUSIVE_LOCK is set).
    :returns: Flat list of target file paths.
    """
    internal = []
    # external = []

    # if not DATA_PIPELINE_READY_DF.empty:
    #     def expand_paths(x):
    #         stem = Path(x).stem
    #         seeds = get_seeds(x)
    #         return (
    #             [
    #                 f"{OUTPUT_DIR}/rule_AF3_INFERENCE/{stem}/seed-{seed}_sample-{sample}/"
    #                 + (f"{stem}_seed-{seed}_sample-{sample}_model.cif"
    #                    if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "model.cif")
    #                 for seed, sample in product(seeds, range(N_SAMPLES))
    #             ]
    #             + [
    #                 f"{OUTPUT_DIR}/rule_AF3_INFERENCE/{stem}/seed-{seed}_sample-{sample}/"
    #                 + (f"{stem}_seed-{seed}_sample-{sample}_model_15_15.txt"
    #                    if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "model_15_15.txt")
    #                 for seed, sample in product(seeds, range(N_SAMPLES))
    #             ]
    #             + [
    #                 f"{OUTPUT_DIR}/rule_AF3_INFERENCE/{stem}/seed-{seed}_sample-{sample}/"
    #                 + (f"{stem}_seed-{seed}_sample-{sample}_model_10_15.txt"
    #                    if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "model_10_15.txt")
    #                 for seed, sample in product(seeds, range(N_SAMPLES))
    #             ]
    #         )

    #     external.extend(
    #         path
    #         for paths in DATA_PIPELINE_READY_DF["file"].apply(expand_paths)
    #         for path in paths
    #     )
    # if not MERGE_READY_DF.empty:
    #     # MERGE_READY_DF may contain both user-supplied merge_ready rows AND
    #     # synthetic rows appended from data_pipeline_ready.  Collect unique
    #     # inference targets from the user-supplied merge_ready rows only
    #     # (identified by the original MERGE_READY_PATH) to avoid double-counting
    #     # dp_ready targets that are already collected above.
    #     _user_mr = MERGE_READY_DF[
    #         ~MERGE_READY_DF["multimer_file"].isin(
    #             DATA_PIPELINE_READY_DF["file"].tolist() if not DATA_PIPELINE_READY_DF.empty else []
    #         )
    #     ]
    #     if not _user_mr.empty:
    #         def expand_paths_mr(x):
    #             stem = Path(x).stem
    #             seeds = get_seeds(x)
    #             combos = list(product(seeds, range(N_SAMPLES)))
    #             versioned = AF3_VERSION not in ["v3.0.0", "v3.0.1"]
    #             base = lambda seed, sample: (
    #                 f"{OUTPUT_DIR}/rule_AF3_INFERENCE/{stem}/seed-{seed}_sample-{sample}/{stem}_seed-{seed}_sample-{sample}"
    #                 if versioned
    #                 else f"{OUTPUT_DIR}/rule_AF3_INFERENCE/{stem}/seed-{seed}_sample-{sample}/"
    #             )
    #             return (
    #                 [f"{base(seed, sample)}_model.cif" if versioned else f"{base(seed, sample)}model.cif" for seed, sample in combos]
    #                 + [f"{base(seed, sample)}_model_15_15.txt" if versioned else f"{base(seed, sample)}model_15_15.txt" for seed, sample in combos]
    #                 + [f"{base(seed, sample)}_model_10_15.txt" if versioned else f"{base(seed, sample)}model_10_15.txt" for seed, sample in combos]
    #             )

    #         external.extend(
    #             path
    #             for paths in _user_mr["multimer_file"].apply(expand_paths_mr)
    #             for path in paths
    #         )

    # if not INFERENCE_READY_DF.empty:
    #     def expand_paths_inf(x):
    #         stem = Path(x).stem
    #         seeds = get_seeds(x)
    #         combos = list(product(seeds, range(N_SAMPLES)))
    #         versioned = AF3_VERSION not in ["v3.0.0", "v3.0.1"]
    #         base_dir = lambda seed, sample: f"{OUTPUT_DIR}/rule_AF3_INFERENCE/{stem}/seed-{seed}_sample-{sample}"
    #         fname = lambda seed, sample, suffix: (
    #             f"{stem}_seed-{seed}_sample-{sample}_{suffix}" if versioned else suffix
    #         )
    #         return (
    #             [f"{base_dir(seed, sample)}/{fname(seed, sample, 'model.cif')}" for seed, sample in combos]
    #             + [f"{base_dir(seed, sample)}/{fname(seed, sample, 'model_15_15.txt')}" for seed, sample in combos]
    #             + [f"{base_dir(seed, sample)}/{fname(seed, sample, 'model_10_15.txt')}" for seed, sample in combos]
    #         )
    #     external.extend(
    #         path
    #         for paths in INFERENCE_READY_DF["file"].apply(expand_paths_inf)
    #         for path in paths
    #     )

    # external = list(dict.fromkeys(external))  # dedupe while preserving order


    PREPROCESSING_DIR = checkpoints.PREPROCESSING.get(**wildcards).output[0]
    JOB_NAMES_MULTIMERS, = glob_wildcards(os.path.join(PREPROCESSING_DIR, "multimers", "{multi}.json"))
    base_names_with_mutations = set(MUTATION_DF["sample_id"].unique()) if not MUTATION_DF.empty else set()
    SEEDS = list(map(lambda x: re.search(r'seed-(\d+)', x).group(1), JOB_NAMES_MULTIMERS))


    internal.append([
        path
        for multi, seed in zip(JOB_NAMES_MULTIMERS, SEEDS)
        for path in expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{multi}",
                            "seed-{seed}_sample-{sample}",
                            "{multi}_seed-{seed}_sample-{sample}_model.cif" if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "model.cif"),
            multi=multi, seed=seed, sample=range(N_SAMPLES)
        )
    ])
    internal.append([
        path
        for multi, seed in zip(JOB_NAMES_MULTIMERS, SEEDS)
        for path in expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{multi}",
                            "seed-{seed}_sample-{sample}",
                            "{multi}_seed-{seed}_sample-{sample}_confidences.json" if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "confidences.json"),
            multi=multi, seed=seed, sample=range(N_SAMPLES)
        )
    ])
    internal.append([
        path
        for multi, seed in zip(JOB_NAMES_MULTIMERS, SEEDS)
        for path in expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{multi}",
                            "seed-{seed}_sample-{sample}",
                            "{multi}_seed-{seed}_sample-{sample}_model_15_15.txt" if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "model_15_15.txt"),
            multi=multi, seed=seed, sample=range(N_SAMPLES)
        )
    ])
    internal.append([
        path
        for multi, seed in zip(JOB_NAMES_MULTIMERS, SEEDS)
        for path in expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{multi}",
                            "seed-{seed}_sample-{sample}",
                            "{multi}_seed-{seed}_sample-{sample}_model_10_15.txt" if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "model_10_15.txt"),
            multi=multi, seed=seed, sample=range(N_SAMPLES)
        )
    ])


    if not MUTATION_DF.empty:
        all_mutations = []
        all_seeds = []

        PREPROCESSING_DIR = checkpoints.PREPROCESSING.get(**wildcards).output[0]
        JOB_NAMES_MULTIMERS, = glob_wildcards(
            os.path.join(PREPROCESSING_DIR, "multimers", "{multi}.json")
        )
        base_names_with_mutations = set(MUTATION_DF["sample_id"].unique())
        for multi in JOB_NAMES_MULTIMERS:
            if re.sub(r"_seed-\d+$", "", multi) not in base_names_with_mutations:
                continue
        MUTATE_DIR = checkpoints.MUTATE.get(**wildcards).output[0]
        muts, = glob_wildcards(os.path.join(MUTATE_DIR, "{mut}.json"))
        seeds = list(map(lambda x: re.search(r'seed-(\d+)', x).group(1), muts))
        all_mutations.extend(muts)
        all_seeds.extend(seeds)
        internal.append([
            path
            for mut, seed in zip(all_mutations, all_seeds)
            for path in expand(
                os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{mut}",
                                "seed-{seed}_sample-{sample}",
                                "{mut}_seed-{seed}_sample-{sample}_model.cif" if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "model.cif"),
                mut=mut, seed=seed, sample=range(N_SAMPLES)
            )
        ])
        internal.append([
            path
            for mut, seed in zip(all_mutations, all_seeds)
            for path in expand(
                os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{mut}",
                                "seed-{seed}_sample-{sample}",
                                "{mut}_seed-{seed}_sample-{sample}_model_10_15.txt" if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "model_10_15.txt"),
                mut=mut, seed=seed, sample=range(N_SAMPLES)
            )
        ])
        internal.append([
            path
            for mut, seed in zip(all_mutations, all_seeds)
            for path in expand(
                os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{mut}",
                                "seed-{seed}_sample-{sample}",
                                "{mut}_seed-{seed}_sample-{sample}_model_15_15.txt" if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "model_15_15.txt"),
                mut=mut, seed=seed, sample=range(N_SAMPLES)
            )
        ])

    return flatten(internal)



def inference_outputs(wildcards):
    """Return all final inference output paths for the `rule all` target."""
    return _collect_inference_targets(wildcards, use_lock=EXCLUSIVE_LOCK)



def aggregate_outputs(wildcards):
    """Return global TSV paths for all inference jobs (one per job)."""
    cif_paths = _collect_inference_targets(wildcards, use_lock=False)
    model_suffix = "_model.cif" if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "model.cif"

    global_ = [
        os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES", f"{Path(p).parent.parent.name}", f"{Path(p).parent.parent.name}_{Path(p).parent.name}_af_global.tsv")
        for p in cif_paths
        if p.endswith(model_suffix)
    ]
    per_chain_ = [
        os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES", f"{Path(p).parent.parent.name}", f"{Path(p).parent.parent.name}_{Path(p).parent.name}_af_per_chain.tsv")
        for p in cif_paths
        if p.endswith(model_suffix)
    ]
    per_chain_pair_ = [
        os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES", f"{Path(p).parent.parent.name}", f"{Path(p).parent.parent.name}_{Path(p).parent.name}_af_per_chain_pair.tsv")
        for p in cif_paths
        if p.endswith(model_suffix)
    ]
    ipsae = [
        os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES", f"{Path(p).parent.parent.name}", f"{Path(p).parent.parent.name}_{Path(p).parent.name}_ipsae.tsv")
        for p in cif_paths
        if p.endswith(model_suffix)
    ]
    return [*global_, *per_chain_, *per_chain_pair_, *ipsae]

def meta_aggregate_outputs(wildcards):
    """Return the three project-level summary TSV paths."""
    return [
        os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all_af_global.tsv"),
        os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all_af_per_chain.tsv"),
        os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all_af_per_chain_pair.tsv"),
        os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all_ipsae.tsv"),
    ]

def datavzrd_output(wildcards):
    """Return the datavzrd HTML report directory path."""
    return [os.path.join(OUTPUT_DIR, "rule_REPORT")]

# ── Singularity / Apptainer utils ────────────────────────────────────────────

def _first_level_root(p: Path) -> Path | None:
    try:
        p = p.expanduser()
        if not p.is_absolute():
            p = p.resolve()
        parts = p.parts
        if len(parts) >= 2:
            root = Path("/").joinpath(parts[1])
            if root.exists():
                return root
    except (OSError, RuntimeError):
        pass
    return None

def _collect_roots(paths: Iterable[str | Path]) -> set[str]:
    # Credit: https://github.com/KosinskiLab/AlphaPulldownSnakemake
    roots: set[str] = set()
    for raw in paths:
        try:
            p = Path(raw)
            r1 = _first_level_root(p)
            if r1:
                roots.add(str(r1))
            try:
                rp = p.expanduser().resolve()
                r2 = _first_level_root(rp)
                if r2:
                    roots.add(str(r2))
            except (OSError, RuntimeError):
                pass
        except (TypeError, OSError, RuntimeError):
            pass
    return roots

def prepare_container_binds(
    *,
    output_directory: str,
    config: dict[str, Any]
) -> None:
    # Credit: https://github.com/KosinskiLab/AlphaPulldownSnakemake
    """Populate Singularity/Apptainer bind paths based on config."""
    interest: set[Path] = {
        Path(__file__).parent,
        Path.cwd(),
        Path(output_directory),
    }
    for key in ("databases_dir", "models_dir"):
        value = config["af3_flags"].get(key)
        if value:
            interest.add(Path(value))
    roots = sorted(_collect_roots(interest))
    bind_spec = ",".join(f"{r}:{r}" for r in roots)
    for var in ("APPTAINER_BINDPATH", "SINGULARITY_BINDPATH"):
        os.environ.setdefault(var, bind_spec)
    for var in ("APPTAINER_NV", "SINGULARITY_NV"):
        os.environ.setdefault(var, "1")


def _raise_nofile_limit(target: int = 65535) -> None:
    """Raise soft RLIMIT_NOFILE up to target or hard limit."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(target, hard)
        if new_soft > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except (OSError, ValueError):
        pass
