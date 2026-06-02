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

# ── Utility ──────────────────────────────────────────────────────────────────
_ALLOWED = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-.")

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

RAW_DATA_PATH, RAW_DATA_DF = load_sample_sheet("raw_data")
DATA_PIPELINE_READY_PATH, DATA_PIPELINE_READY_DF = load_sample_sheet("data_pipeline_ready")
INFERENCE_READY_PATH, INFERENCE_READY_DF = load_sample_sheet("inference_ready")
MERGE_READY_PATH, MERGE_READY_DF = load_sample_sheet("merge_ready")

# ── Sample-sheet validation ──────────────────────────────────────────────────
if not RAW_DATA_DF.empty:
    validate(RAW_DATA_DF, schema="../schemas/sample_sheet.raw_data.schema.yaml")
if not DATA_PIPELINE_READY_DF.empty:
    validate(DATA_PIPELINE_READY_DF, schema="../schemas/sample_sheet.data_pipeline_ready.schema.yaml")
if not INFERENCE_READY_DF.empty:
    validate(INFERENCE_READY_DF, schema="../schemas/sample_sheet.inference_ready.schema.yaml")
if not MERGE_READY_DF.empty:
    validate(MERGE_READY_DF, schema="../schemas/sample_sheet.merge_ready.schema.yaml")

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
    external = []

    if not DATA_PIPELINE_READY_DF.empty:
        external.append(
            DATA_PIPELINE_READY_DF["file"]
            .apply(lambda x: f"{OUTPUT_DIR}/rule_AF3_INFERENCE/{Path(x).stem}/{Path(x).stem}_model.cif")
            .unique().tolist()
        )

    if not MERGE_READY_DF.empty:
        # MERGE_READY_DF may contain both user-supplied merge_ready rows AND
        # synthetic rows appended from data_pipeline_ready.  Collect unique
        # inference targets from the user-supplied merge_ready rows only
        # (identified by the original MERGE_READY_PATH) to avoid double-counting
        # dp_ready targets that are already collected above.
        _user_mr = MERGE_READY_DF[
            ~MERGE_READY_DF["multimer_file"].isin(
                DATA_PIPELINE_READY_DF["file"].tolist() if not DATA_PIPELINE_READY_DF.empty else []
            )
        ]
        if not _user_mr.empty:
            external.append(
                _user_mr["multimer_file"]
                .apply(lambda x: f"{OUTPUT_DIR}/rule_AF3_INFERENCE/{Path(x).stem}/{Path(x).stem}_model.cif")
                .unique().tolist()
            )

    if not INFERENCE_READY_DF.empty:
        external.append(
            INFERENCE_READY_DF["file"]
            .apply(lambda x: f"{OUTPUT_DIR}/rule_AF3_INFERENCE/{Path(x).stem}/{Path(x).stem}_model.cif")
            .unique().tolist()
        )

    if not RAW_DATA_DF.empty:
        PREPROCESSING_DIR = checkpoints.PREPROCESSING.get(**wildcards).output[0]
        JOB_NAMES_MULTIMERS, = glob_wildcards(os.path.join(PREPROCESSING_DIR, "multimers", "{multi}.json"))
        internal.append(list(expand(os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{multi}", "{multi}_model.cif"),multi=JOB_NAMES_MULTIMERS)))
        #if use_lock:
        #    internal.append(list(expand(
        #        os.path.join(OUTPUT_DIR, "rule_CREATE_AF3_INFERENCE_JOBS", "{multi}_af3_inference_job.txt"),
        #        multi=JOB_NAMES_MULTIMERS
        #    )))
        #else:
        #    internal.append(list(expand(
        #        os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{multi}", "{multi}_model.cif"),
        #        multi=JOB_NAMES_MULTIMERS
        #    )))

    if internal and external:
        return [*flatten(internal), *flatten(external)]
    if internal:
        return flatten(internal)
    if external:
        return flatten(external)
    return []


def inference_outputs(wildcards):
    """Return all final inference output paths for the `rule all` target."""
    return _collect_inference_targets(wildcards, use_lock=EXCLUSIVE_LOCK)


def get_multimeric_json_with_msas(wildcards):
    """Return inference targets used by the report rule."""
    return _collect_inference_targets(wildcards, use_lock=EXCLUSIVE_LOCK)

def aggregate_outputs(wildcards):
    """Return global TSV paths for all inference jobs (one per job)."""
    cif_paths = _collect_inference_targets(wildcards, use_lock=False)
    global_ = [
        os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", f"{Path(p).parent.name}_global.tsv")
        for p in cif_paths
        if p.endswith("_model.cif")
    ] 
    per_chain_ = [
        os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", f"{Path(p).parent.name}_per_chain.tsv")
        for p in cif_paths
        if p.endswith("_model.cif")
    ] 
    per_chain_pair_ = [
        os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", f"{Path(p).parent.name}_per_chain_pair.tsv")
        for p in cif_paths
        if p.endswith("_model.cif")
    ]

    return [*global_,*per_chain_,*per_chain_pair_]


def meta_aggregate_outputs(wildcards):
    """Return the three project-level summary TSV paths."""
    return [
        os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all_global.tsv"),
        os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all_per_chain.tsv"),
        os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all_per_chain_pair.tsv"),
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
    config: dict[str, Any],
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
