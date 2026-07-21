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
    # NOTE: loaded via load_sample_sheet("merge_ready"); keep BOTH keys mapped to
    # the same columns so an absent merge_ready sheet yields a DataFrame WITH the
    # expected columns (previously "merge_ready" was unmapped -> empty-columns DF
    # -> KeyError/silent drops when combined with data_pipeline_ready).
    "merge_ready": ["sample_id", "multimer_file", "monomer_chain_id", "monomer_file"],
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


def strip_seed(name: str) -> str:
    """Remove a trailing _seed-<number> suffix from a sample name.

    Mirrors scripts/mutate.py::strip_seed so canonical base-name matching is
    identical on both sides (target enumeration here and mutation naming there).
    """
    return re.sub(r"_seed-\d+$", "", name)


def _write_json_if_changed(path: str, obj) -> bool:
    """Write ``obj`` as JSON to ``path`` only when the content would change.

    This keeps the load-time normalization IDEMPOTENT: common.smk executes on
    every snakemake invocation (including --dry-run), so unconditional writes
    would churn mtimes and force AF3_INFERENCE to rerun each time.  We compare a
    canonical serialization (sorted keys) against any existing file and skip the
    write when identical, preserving the existing file's mtime.

    :returns: True if the file was (re)written, False if left untouched.
    """
    new_text = json.dumps(obj, sort_keys=True, indent=2)
    p = Path(path)
    if p.exists():
        try:
            if p.read_text() == new_text:
                return False
        except OSError:
            pass  # unreadable -> fall through and rewrite
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(new_text)
    return True


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
MUTATION_DF_PATH, MUTATION_DF = load_sample_sheet("mutations")
SCORING_READY_PATH, SCORING_READY_DF = load_sample_sheet("scoring_ready")

# ── Sample-sheet validation ──────────────────────────────────────────────────
# The raw_data sheet has mode-dependent required columns:
#   - custom / all-vs-all / pulldown: chain 'id' + 'sequence' are supplied
#     directly, so validate against the strict raw_data schema.
#   - virtual-drug-screen / stoichio-screen: the raw sheet carries a 'data'
#     column (+ optional 'count') and NO 'id' — the chain 'id' (A, B, C ...) is
#     generated during PREPROCESSING. Validate against the screen-mode schema so
#     these legitimate sheets are not rejected for a missing 'id'.
_SCREEN_MODES = {"virtual-drug-screen", "stoichio-screen"}
if not RAW_DATA_DF.empty:
    if MODE in _SCREEN_MODES:
        validate(RAW_DATA_DF, schema="../schemas/sample_sheet.raw_data.screen.schema.yaml")
    else:
        validate(RAW_DATA_DF, schema="../schemas/sample_sheet.raw_data.schema.yaml")
if not DATA_PIPELINE_READY_DF.empty:
    validate(DATA_PIPELINE_READY_DF, schema="../schemas/sample_sheet.data_pipeline_ready.schema.yaml")
if not INFERENCE_READY_DF.empty:
    validate(INFERENCE_READY_DF, schema="../schemas/sample_sheet.inference_ready.schema.yaml")
if not MERGE_READY_DF.empty:
    validate(MERGE_READY_DF, schema="../schemas/sample_sheet.merge_ready.schema.yaml")

# ── Canonical normalization of the inference_ready entry point ────────────────
# inference_ready samples SKIP preprocessing/data-pipeline/merge and go straight
# to AF3 inference.  They are the ONLY stream that is not already exploded into
# one-JSON-per-seed by the time it reaches inference (raw_data is exploded by
# preprocessing; data_pipeline_ready / merge_ready arrive single-seed at the
# merged-multimer step by construction/contract).  A user-supplied inference_ready
# JSON may legitimately carry modelSeeds=[1,2,3] in a single seedless-named file.
#
# To give the whole downstream ONE canonical representation (single seed per
# JSON, with `_seed-N` in both the filename stem and the JSON `name`, and
# modelSeeds=[N]), we normalize at load time here — no new rules/checkpoints.
# Each exploded JSON is written to OUTPUT_DIR/normalized_inputs/inference_ready/
# (a directory owned by NO rule) and listed in a derived, persisted sample sheet
# that REPLACES INFERENCE_READY_DF downstream.  The original is kept as
# INFERENCE_READY_DF_RAW for provenance.
#
# Design decisions (confirmed with user):
#   * seeds: the JSON's own modelSeeds win; config n_seeds does NOT override an
#     externally supplied inference_ready file.  Default only when absent/empty.
#   * writes are idempotent (write-if-content-differs) to avoid rerun churn.
#   * canonical stem = sample_id = filename stem = JSON name = f"{base}_seed-{s}".
NORMALIZED_INPUTS_DIR = os.path.join(OUTPUT_DIR, "normalized_inputs", "inference_ready")
_DEFAULT_SEED_WHEN_MISSING = [1]  # only used if a JSON has no modelSeeds

INFERENCE_READY_DF_RAW = INFERENCE_READY_DF.copy()


def _load_af3_json(path: str) -> dict:
    """Load an AF3 inference JSON. Accepts a single dict (this workflow's format)
    or a 1-element list; raises on a multi-entry list (ambiguous seed/name)."""
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, list):
        if len(data) != 1:
            raise WorkflowError(
                f"inference_ready JSON '{path}' contains a list of "
                f"{len(data)} entries; expected a single AF3 job object."
            )
        data = data[0]
    if not isinstance(data, dict):
        raise WorkflowError(f"inference_ready JSON '{path}' is not a JSON object.")
    return data


def normalize_inference_ready_df(df: pd.DataFrame) -> pd.DataFrame:
    """Explode every inference_ready row into one canonical single-seed row.

    For each input row (sample_id, file):
      * read the JSON; take seeds = modelSeeds (or the default when absent);
      * base = strip_seed(json["name"]) (fall back to the file stem);
      * for each seed s: write {base}_seed-{s}.json with name={base}_seed-{s}
        and modelSeeds=[s] into NORMALIZED_INPUTS_DIR (idempotently);
      * emit a row sample_id={base}_seed-{s}, file=<new path>.

    Returns the expanded DataFrame (same columns as the input plus provenance).
    """
    if df.empty:
        return df.copy()

    rows = []
    for _, r in df.iterrows():
        src = str(r["file"])
        if not os.path.isabs(src):
            # Paths in the sheet are relative to snakemake --directory (cwd).
            src = os.path.join(os.getcwd(), src)
        if not os.path.exists(src):
            # Keep the row as-is; downstream will surface a clear missing-input
            # error. We cannot explode what we cannot read.
            rows.append({"sample_id": r["sample_id"], "file": r["file"],
                         "source_sample_id": r["sample_id"], "source_file": r["file"]})
            continue

        data = _load_af3_json(src)
        seeds = data.get("modelSeeds") or _DEFAULT_SEED_WHEN_MISSING
        if not isinstance(seeds, (list, tuple)):
            seeds = [seeds]
        # De-duplicate seeds while preserving order (a user JSON could repeat one).
        seeds = list(dict.fromkeys(seeds))

        orig_stem = Path(src).stem  # the current AF3_INFERENCE {name} wildcard
        name = data.get("name") or orig_stem

        # IMPORTANT — the AF3_INFERENCE output dir is
        #   rule_AF3_INFERENCE/{stem}/seed-{seed}_sample-N/
        # so the seed is ALREADY namespaced by the seed-{seed}/ subdirectory.
        # The stem therefore does NOT need to carry _seed-N for correctness.
        # We only need per-seed-distinct FILENAMES when a single input JSON holds
        # MULTIPLE seeds (otherwise the N single-seed files would collide on disk).
        #
        #   * single-seed input  -> keep the original stem verbatim (this makes
        #     the normalization a no-op on already-canonical seeded files and
        #     preserves exact backward compatibility / target paths);
        #   * multi-seed input   -> split into N files, disambiguating the stem
        #     with a trailing _seed-{s} (stripping any pre-existing trailing
        #     _seed-N first so we never double it).
        multi = len(seeds) > 1
        base_for_split = strip_seed(orig_stem)

        for s in seeds:
            if multi:
                new_stem = f"{base_for_split}_seed-{s}"
            else:
                new_stem = orig_stem
            new_path = os.path.join(NORMALIZED_INPUTS_DIR, f"{new_stem}.json")
            new_obj = dict(data)
            new_obj["name"] = new_stem if multi else name
            new_obj["modelSeeds"] = [s]
            _write_json_if_changed(new_path, new_obj)
            rows.append({
                "sample_id": new_stem,
                "file": new_path,
                "source_sample_id": r["sample_id"],
                "source_file": r["file"],
            })

    return pd.DataFrame(rows, columns=["sample_id", "file", "source_sample_id", "source_file"])


if not INFERENCE_READY_DF_RAW.empty:
    INFERENCE_READY_DF = normalize_inference_ready_df(INFERENCE_READY_DF_RAW)
    # Persist the expanded sheet for provenance / debugging.
    Path(NORMALIZED_INPUTS_DIR).mkdir(parents=True, exist_ok=True)
    _expanded_sheet = os.path.join(OUTPUT_DIR, "normalized_inputs", "inference_ready.expanded.tsv")
    INFERENCE_READY_DF.to_csv(_expanded_sheet, sep="\t", index=False)


# ── merge_ready single-seed contract enforcement ─────────────────────────────
# By contract (see scripts/merge_mono_and_multi_jsons.py), a merge_ready multimer
# JSON encodes exactly ONE seed — the {multi} wildcard is expected to already be a
# single-seed job.  We do NOT explode merge_ready (per the "explode at inference
# stage only" decision); instead we FAIL FAST if a user-supplied merge_ready
# multimer carries multiple seeds, so the violation is loud rather than silently
# producing mismatched seed wildcards downstream.
def _enforce_merge_ready_single_seed(df: pd.DataFrame, source_path) -> None:
    if df is None or df.empty or "multimer_file" not in df.columns:
        return
    offenders = []
    for _, r in df.iterrows():
        mf = r.get("multimer_file")
        if not isinstance(mf, str) or not mf:
            continue
        p = mf if os.path.isabs(mf) else os.path.join(os.getcwd(), mf)
        if not os.path.exists(p):
            continue  # generated later / not a user file we can inspect now
        try:
            data = _load_af3_json(p)
        except WorkflowError:
            continue
        seeds = data.get("modelSeeds")
        if isinstance(seeds, (list, tuple)) and len(seeds) > 1:
            offenders.append((r.get("sample_id", mf), mf, list(seeds)))
    if offenders:
        lines = "\n".join(f"  - {sid}: {mf} has modelSeeds={seeds}" for sid, mf, seeds in offenders)
        raise WorkflowError(
            "merge_ready multimer JSON(s) violate the single-seed contract "
            "(a merge_ready file must contain exactly one seed). Split these into "
            "one file per seed before running, or supply them via inference_ready "
            "(which is exploded automatically):\n" + lines
        )


# Only the USER-supplied merge_ready rows are subject to the contract; the
# synthetic dp_ready rows are appended later and are single-seed by construction.
if MERGE_READY_PATH is not None and not MERGE_READY_DF.empty:
    _enforce_merge_ready_single_seed(MERGE_READY_DF, MERGE_READY_PATH)


# ── Cross-stream canonical-stem collision check (load-time, static part) ──────
# When all streams are unified into one inference namespace, two DISTINCT sources
# that resolve to the SAME AF3_INFERENCE {name} stem would write into the same
# rule_AF3_INFERENCE/{name}/... directory -> silent overwrite / wrong provenance.
# We fail fast on such collisions.
#
# Raw_data multimer stems are only fully known AFTER the PREPROCESSING checkpoint
# (seed explosion is mode/n_seeds dependent), so the DEFINITIVE check runs during
# target enumeration (see _assert_no_stem_collisions used in
# _collect_inference_targets).  Here we do the cheap static check across the
# statically-known stems so the common cases fail early with a clear message.
def _static_canonical_stems() -> dict:
    """Return {stem: {sources}} for statically-known inference stems.

    Sources are collected as a SET per stem: a single stream may legitimately
    repeat a stem (e.g. merge_ready lists one row per chain of the same
    multimer), which is NOT a collision.  A collision is the same canonical stem
    arising from TWO DISTINCT entry-point sources.
    """
    stems: dict[str, set] = {}

    def _add(stem: str, source: str) -> None:
        stems.setdefault(str(stem), set()).add(source)

    # inference_ready: normalized stems (already exploded)
    if not INFERENCE_READY_DF.empty and "sample_id" in INFERENCE_READY_DF.columns:
        for sid in INFERENCE_READY_DF["sample_id"]:
            _add(sid, "inference_ready")
    # merge_ready (user rows only): stem of multimer_file
    if MERGE_READY_PATH is not None and not MERGE_READY_DF.empty and "multimer_file" in MERGE_READY_DF.columns:
        for mf in MERGE_READY_DF["multimer_file"]:
            if isinstance(mf, str) and mf:
                _add(Path(mf).stem, "merge_ready")
    # data_pipeline_ready: stem of file (becomes {multi} via synthetic merge)
    if not DATA_PIPELINE_READY_DF.empty and "file" in DATA_PIPELINE_READY_DF.columns:
        for f in DATA_PIPELINE_READY_DF["file"]:
            if isinstance(f, str) and f:
                _add(Path(f).stem, "data_pipeline_ready")
    return stems


def _assert_no_stem_collisions(stems: dict) -> None:
    # Collision == a stem produced by >1 DISTINCT source.
    dupes = {s: srcs for s, srcs in stems.items() if len(srcs) > 1}
    if dupes:
        lines = "\n".join(
            f"  - '{s}' produced by: {', '.join(sorted(srcs))}"
            for s, srcs in sorted(dupes.items())
        )
        raise WorkflowError(
            "Canonical inference-name collision across entry points detected. "
            "Two distinct sources map to the same AF3_INFERENCE output name, which "
            "would overwrite each other. Rename the offending sample(s) so each "
            "canonical name is unique:\n" + lines
        )


_assert_no_stem_collisions(_static_canonical_stems())

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


def get_mutatable_merged_multimers(wildcards):
    """Return every merged-multimer JSON path that MUTATE should depend on.

    Merged multimers (rule_MERGE_MONOMERS_TO_MULTIMERS/{multi}_data.json) are the
    canonical single-seed multimer inputs from the raw_data, data_pipeline_ready
    and merge_ready streams. They are enumerated from two independent sources:

      * raw_data — the multimer stems are known only after the PREPROCESSING
        checkpoint (via inference_samples.tsv). PREPROCESSING declares NO output
        when RAW_DATA_DF is empty, so we must guard the checkpoint access on
        RAW_DATA_DF — otherwise `.output[0]` raises IndexError for streams that
        never run preprocessing (e.g. an inference_ready-only run).
      * data_pipeline_ready / merge_ready — the stems are the MERGE_READY_DF
        sample_ids (this frame already includes the synthetic data_pipeline rows),
        known at load time with no checkpoint dependency.

    Returning [] cleanly when no merged-multimer stream exists is what lets MUTATE
    be used with an inference_ready-only sheet (mutations applied purely to the
    exploded inference_ready JSONs).
    """
    multis: list = []

    # raw_data stream (post-PREPROCESSING); guarded so we never touch an empty
    # checkpoint output.
    if not RAW_DATA_DF.empty:
        multis.extend(get_multi_to_monomeric_dict_(wildcards))

    # data_pipeline_ready + merge_ready stream (load-time stems; dedupe by stem
    # because a merge_ready sample may span multiple chain rows).
    if not MERGE_READY_DF.empty and "sample_id" in MERGE_READY_DF.columns:
        for sid in MERGE_READY_DF["sample_id"]:
            if isinstance(sid, str) and sid:
                multis.append(sid)

    # De-duplicate while preserving order.
    seen: set = set()
    ordered: list = []
    for m in multis:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return list(
        expand(
            os.path.join(OUTPUT_DIR, "rule_MERGE_MONOMERS_TO_MULTIMERS", "{multi}_data.json"),
            multi=ordered,
        )
    )

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
    """Return the modelSeeds list from an AF3 JSON.

    Robust to a JSON that is a single object OR a 1-element list, and to a
    missing modelSeeds key (returns the workflow default rather than KeyError).
    Reading seeds from the JSON — never by regex-parsing a filename — is what
    makes seedless inference_ready files safe here.
    """
    try:
        data = _load_af3_json(json_path)
    except WorkflowError:
        with open(json_path) as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            data = data[0]
    seeds = data.get("modelSeeds") if isinstance(data, dict) else None
    if not seeds:
        return list(_DEFAULT_SEED_WHEN_MISSING)
    if not isinstance(seeds, (list, tuple)):
        seeds = [seeds]
    return list(seeds)


# The four per-(stem, seed, sample) artifacts every inference target produces.
# Emitting all four consistently (including confidences.json, which the previous
# dp_ready / inference_ready / mutation blocks omitted) makes downstream score
# extraction and aggregation uniform across every entry point.
_INFERENCE_SUFFIXES = ("model.cif", "confidences.json", "model_15_15.txt", "model_10_15.txt")


def _inference_target_paths(stem: str, seeds, *, n_samples: int = None) -> list:
    """Return all AF3_INFERENCE target paths for one canonical (stem, seeds).

    The output directory is rule_AF3_INFERENCE/{stem}/seed-{seed}_sample-{n}/ so
    the seed is namespaced by the subdirectory; filenames are versioned
    (prefixed with the stem/seed/sample) unless AF3_VERSION is a legacy version.
    """
    if n_samples is None:
        n_samples = N_SAMPLES
    versioned = AF3_VERSION not in ["v3.0.0", "v3.0.1"]
    out = []
    for seed, sample in product(seeds, range(n_samples)):
        d = os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", f"{stem}",
                         f"seed-{seed}_sample-{sample}")
        for suf in _INFERENCE_SUFFIXES:
            fname = f"{stem}_seed-{seed}_sample-{sample}_{suf}" if versioned else suf
            out.append(os.path.join(d, fname))
    return out


def _collect_inference_targets(wildcards, *, use_lock: bool) -> list:
    """Collect every final AF3 inference target as ONE unified union.

    All entry points are normalized to a common canonical representation before
    this point, so a single code path can enumerate targets for every stream:

      * data_pipeline_ready / merge_ready — merged multimers produced by
        rule_MERGE_MONOMERS_TO_MULTIMERS; the merge step already yields one seed
        per {multi} (single-seed by construction/contract).
      * inference_ready — exploded at load time into single-seed JSONs
        (INFERENCE_READY_DF holds the expanded rows); seeds come from each JSON.
      * raw_data — multimers produced by the PREPROCESSING checkpoint, already
        one-JSON-per-seed (seed encoded in the {multi} stem).

    Mutations (all-streams): when MUTATION_DF is non-empty, ANY canonical sample
    whose seed-stripped name matches a MUTATION_DF.sample_id contributes mutated
    targets. Mutated JSONs are produced by the MUTATE checkpoint into rule_MUTATE/;
    we glob that directory and read each mutated JSON's own modelSeeds via
    get_seeds() — we NEVER regex-parse the seed out of the filename (the previous
    implementation did, and crashed / dropped targets for seedless names).

    A definitive cross-stream canonical-name collision check runs here, where the
    raw multimer stems (known only after the PREPROCESSING checkpoint) become
    available, complementing the static load-time check.

    :param wildcards: Snakemake wildcards object.
    :param use_lock: Present for backward-compatible signature; retained for
                     callers that pass EXCLUSIVE_LOCK. (Lock targets are handled
                     by the inference rule itself, not by target enumeration.)
    :returns: Deduplicated flat list of target file paths.
    """
    # Map of canonical stem -> set of contributing sources, for the definitive
    # (post-checkpoint) collision check.
    stem_sources: dict = {}

    def _register(stem: str, source: str) -> None:
        stem_sources.setdefault(str(stem), set()).add(source)

    # (stem, seeds) pairs for NON-mutated targets. Mutated targets are enumerated
    # separately (below) by globbing the MUTATE checkpoint output, which is the
    # authoritative source of which samples were actually mutated — so we do NOT
    # need to (and must not) parse MUTATION_DF columns here. That keeps this path
    # agnostic to the mutations-table format (headerless legacy vs. header-based),
    # both of which scripts/mutate.py accepts.
    canonical: list = []              # list[tuple[str, list]]

    # ---- external streams (data_pipeline_ready / merge_ready / inference_ready) ----
    # data_pipeline_ready: monomer 'file' -> merged multimer named by the file stem.
    if not DATA_PIPELINE_READY_DF.empty:
        for f in DATA_PIPELINE_READY_DF["file"]:
            stem = Path(f).stem
            canonical.append((stem, get_seeds(f)))
            _register(stem, "data_pipeline_ready")

    # merge_ready: USER rows only (exclude synthetic dp rows to avoid double count).
    if not MERGE_READY_DF.empty and "multimer_file" in MERGE_READY_DF.columns:
        _dp_files = (
            DATA_PIPELINE_READY_DF["file"].tolist()
            if not DATA_PIPELINE_READY_DF.empty else []
        )
        _user_mr = MERGE_READY_DF[~MERGE_READY_DF["multimer_file"].isin(_dp_files)]
        # A merge_ready sample may span multiple chains (repeated rows); dedupe by stem.
        _seen_mr = set()
        for mf in _user_mr["multimer_file"]:
            if not isinstance(mf, str) or not mf:
                continue
            stem = Path(mf).stem
            if stem in _seen_mr:
                continue
            _seen_mr.add(stem)
            canonical.append((stem, get_seeds(mf)))
            _register(stem, "merge_ready")

    # inference_ready: expanded single-seed rows (file column points at the
    # normalized JSON). Stem == sample_id == filename stem.
    if not INFERENCE_READY_DF.empty and "file" in INFERENCE_READY_DF.columns:
        for _, r in INFERENCE_READY_DF.iterrows():
            f = r["file"]
            stem = Path(f).stem
            canonical.append((stem, get_seeds(f)))
            _register(stem, "inference_ready")

    # ---- raw_data stream (checkpoint-generated multimers) ----
    if not RAW_DATA_DF.empty:
        PREPROCESSING_DIR = checkpoints.PREPROCESSING.get(**wildcards).output[0]
        JOB_NAMES_MULTIMERS, = glob_wildcards(
            os.path.join(PREPROCESSING_DIR, "multimers", "{multi}.json")
        )
        for multi in JOB_NAMES_MULTIMERS:
            # Seed is encoded in the multimer stem (…_seed-N); read it robustly
            # from the multimer JSON so we never depend on filename parsing.
            multimer_json = os.path.join(PREPROCESSING_DIR, "multimers", f"{multi}.json")
            try:
                seeds = get_seeds(multimer_json)
            except (FileNotFoundError, OSError):
                # Fall back to the seed embedded in the stem if the file is not
                # yet readable during DAG construction.
                m = re.search(r"seed-(\d+)", multi)
                seeds = [int(m.group(1))] if m else list(_DEFAULT_SEED_WHEN_MISSING)
            canonical.append((multi, seeds))
            _register(multi, "raw_data")

    # ---- definitive collision check (raw stems now known) ----
    _assert_no_stem_collisions(stem_sources)

    # ---- assemble non-mutated targets ----
    targets: list = []
    for stem, seeds in canonical:
        targets.extend(_inference_target_paths(stem, seeds))

    # ---- mutation targets (all streams) ----
    # Mutated JSONs are produced by the MUTATE checkpoint; enumerate them by
    # globbing rule_MUTATE and reading each mutated JSON's own modelSeeds. This
    # covers mutated raw multimers AND mutated exploded inference_ready files
    # (the MUTATE rule takes the union of both — see rules/mutate.smk).
    #
    # Gate on MUTATION_DF_PATH (not MUTATION_DF.empty): the MUTATE checkpoint
    # declares its rule_MUTATE output under exactly this condition, so this is
    # the precise condition under which the checkpoint output exists to glob.
    # It is also format-agnostic (a headerless mutations TSV yields a non-empty
    # frame with non-canonical column names, which .empty handles but column
    # access would not).
    if MUTATION_DF_PATH is not None:
        MUTATE_DIR = checkpoints.MUTATE.get(**wildcards).output[0]
        muts, = glob_wildcards(os.path.join(MUTATE_DIR, "{mut}.json"))
        for mut in muts:
            mut_json = os.path.join(MUTATE_DIR, f"{mut}.json")
            try:
                seeds = get_seeds(mut_json)   # crash-proof, seedless-safe
            except (FileNotFoundError, OSError):
                m = re.search(r"seed-(\d+)", mut)
                seeds = [int(m.group(1))] if m else list(_DEFAULT_SEED_WHEN_MISSING)
            targets.extend(_inference_target_paths(mut, seeds))

    # Deduplicate while preserving order (streams can legitimately overlap on the
    # WT copy vs mutated variants; identical paths must collapse).
    return list(dict.fromkeys(targets))

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
    workflow_directory: str,
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
    bind_spec +=  f",{Path(workflow_directory)}:{Path(workflow_directory)}"
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
