"""Aggregate AF3 per-sample confidence outputs into three TSV tables.

Usage:
    python aggregate_results.py <inference_dir> <global_tsv> <per_chain_tsv> <per_chain_pair_tsv>

For a given job directory (OUTPUT_DIR/rule_AF3_INFERENCE/{job}/), this script:
  - Discovers all seed-*/sample-* subdirectories
  - Reads per-sample summary_confidences.json and confidences.json
  - Writes three TSV files:
      global          — one row per sample (ranking_score, iptm, ptm, mean_plddt, ...)
      per_chain       — one row per (sample, chain)
      per_chain_pair  — one row per (sample, chain_i, chain_j)

Chain IDs are always derived from token_chain_ids in confidences.json (insertion-
order deduplicated), so the script works regardless of whether unique_chain_ids
is present in summary_confidences.json.

AF3 naming conventions
----------------------
New (≥ some AF3 version): files inside seed-{seed}_sample-{sample}/ are prefixed
with the job name, e.g. {job_name}_seed-{seed}_sample-{sample}_model.cif.

Old (< that version): files are simply named model.cif, confidences.json,
summary_confidences.json with no job-name prefix.

This script probes for the new convention first and falls back to the old one.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _unique_ordered(seq):
    """Return deduplicated list preserving first-occurrence order."""
    seen = {}
    for x in seq:
        seen.setdefault(x, None)
    return list(seen)


def _parse_seed_sample(subdir_name: str) -> tuple[int, int]:
    """Parse seed and sample integers from 'seed-{seed}_sample-{sample}'."""
    m = re.fullmatch(r"seed-(\d+)_sample-(\d+)", subdir_name)
    if not m:
        raise ValueError(f"Unexpected subdir name: {subdir_name!r}")
    return int(m.group(1)), int(m.group(2))


def _chain_ids_from_confidences(conf: dict) -> list[str]:
    """Derive ordered unique chain IDs from token_chain_ids in confidences.json."""
    return _unique_ordered(conf["token_chain_ids"])


def _mean_plddt_global(conf: dict) -> float:
    return float(np.mean(conf["atom_plddts"]))


def _mean_plddt_per_chain(conf: dict, chain_ids: list[str]) -> dict[str, float]:
    """Return {chain_id: mean_plddt} using atom_plddts grouped by atom_chain_ids."""
    atom_plddts = np.array(conf["atom_plddts"], dtype=np.float64)
    atom_chain_ids = conf["atom_chain_ids"]
    result = {}
    for chain_id in chain_ids:
        mask = np.array([c == chain_id for c in atom_chain_ids], dtype=bool)
        result[chain_id] = float(np.mean(atom_plddts[mask])) if mask.any() else float("nan")
    return result


def _resolve_sample_paths(
    subdir: Path,
    job_name: str,
    seed: int,
    sample: int,
) -> tuple[Path, Path, Path]:
    """Resolve (cif_path, summary_path, conf_path) for one sample.

    Probes the new AF3 naming convention (prefixed with job_name) first,
    then falls back to the old convention (bare filenames).

    Raises FileNotFoundError if neither convention produces existing files.
    """
    prefix = f"{job_name}_seed-{seed}_sample-{sample}"

    # ── New convention (AF3 ≥ some version) ──────────────────────────────────
    new_cif     = subdir / f"{prefix}_model.cif"
    new_summary = subdir / f"{prefix}_summary_confidences.json"
    new_conf    = subdir / f"{prefix}_confidences.json"

    if new_cif.exists() and new_summary.exists() and new_conf.exists():
        return new_cif, new_summary, new_conf

    # ── Old convention (bare filenames) ──────────────────────────────────────
    old_cif     = subdir / "model.cif"
    old_summary = subdir / "summary_confidences.json"
    old_conf    = subdir / "confidences.json"

    if old_cif.exists() and old_summary.exists() and old_conf.exists():
        return old_cif, old_summary, old_conf

    # ── Neither found ─────────────────────────────────────────────────────────
    raise FileNotFoundError(
        f"Could not find AF3 output files in {subdir} under either naming convention.\n"
        f"  New convention checked: {new_cif.name}, {new_summary.name}, {new_conf.name}\n"
        f"  Old convention checked: {old_cif.name}, {old_summary.name}, {old_conf.name}"
    )


# ── Per-sample processing ─────────────────────────────────────────────────────

def process_sample(
    job_name: str,
    base_name: str,
    seed: int,
    sample: int,
    subdir: Path,
) -> tuple[dict, list[dict], list[dict]]:
    """Return (global_row, per_chain_rows, per_pair_rows) for one sample.

    Parameters
    ----------
    job_name:
        Full directory name, e.g. ``job3_test_seed-10``. Used only for
        resolving file paths under the new AF3 naming convention.
    base_name:
        Seed-stripped job name, e.g. ``job3_test``. Used as the ``name``
        column in all output rows so that samples from the same input but
        different seeds share a common grouping key.
    """
    cif_path, summary_path, conf_path = _resolve_sample_paths(
        subdir, job_name, seed, sample
    )

    with open(summary_path) as f:
        summary = json.load(f)
    with open(conf_path) as f:
        conf = json.load(f)

    chain_ids = _chain_ids_from_confidences(conf)
    mean_plddt_by_chain = _mean_plddt_per_chain(conf, chain_ids)

    id_cols = {
        "name":   base_name,
        "seed":   seed,
        "sample": sample,
        "file":   str(cif_path),
    }

    # ── Global row ────────────────────────────────────────────────────────────
    global_row = {
        **id_cols,
        "ranking_score":       summary["ranking_score"],
        "iptm":                summary["iptm"],
        "ptm":                 summary["ptm"],
        "mean_plddt":          _mean_plddt_global(conf),
        "fraction_disordered": summary["fraction_disordered"],
        "has_clash":           summary["has_clash"],
    }

    # ── Per-chain rows ────────────────────────────────────────────────────────
    chain_ptm  = summary["chain_ptm"]   # list[float], indexed by chain position
    chain_iptm = summary["chain_iptm"]  # list[float], indexed by chain position

    per_chain_rows = [
        {
            **id_cols,
            "chain_id":   chain_id,
            "mean_plddt": mean_plddt_by_chain[chain_id],
            "chain_ptm":  chain_ptm[i],
            "chain_iptm": chain_iptm[i],
        }
        for i, chain_id in enumerate(chain_ids)
    ]

    # ── Per-chain-pair rows ───────────────────────────────────────────────────
    chain_pair_pae_min = summary["chain_pair_pae_min"]  # list[list[float]]
    chain_pair_iptm    = summary["chain_pair_iptm"]     # list[list[float]]

    per_pair_rows = [
        {
            **id_cols,
            "chain_i":            chain_ids[i],
            "chain_j":            chain_ids[j],
            "chain_pair_pae_min": chain_pair_pae_min[i][j],
            "chain_pair_iptm":    chain_pair_iptm[i][j],
        }
        for i in range(len(chain_ids))
        for j in range(len(chain_ids))
    ]

    return global_row, per_chain_rows, per_pair_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main(inference_dir: str, global_tsv: str, per_chain_tsv: str, per_pair_tsv: str):
    inference_dir = Path(inference_dir)
    job_name  = inference_dir.name
    # Strip _seed-{N} suffix so that all seeds of the same input share a name.
    # For entry points where job_name has no seed suffix this is a no-op.
    base_name = re.sub(r"_seed-\d+$", "", job_name)

    # Discover all seed-*_sample-* subdirectories, sorted for determinism
    subdirs = sorted(
        d for d in inference_dir.iterdir()
        if d.is_dir() and re.fullmatch(r"seed-\d+_sample-\d+", d.name)
    )
    if not subdirs:
        raise FileNotFoundError(
            f"No seed-*_sample-* subdirectories found in {inference_dir}"
        )

    global_rows, per_chain_rows, per_pair_rows = [], [], []

    for subdir in subdirs:
        seed, sample = _parse_seed_sample(subdir.name)
        g, c, p = process_sample(job_name, base_name, seed, sample, subdir)
        global_rows.append(g)
        per_chain_rows.extend(c)
        per_pair_rows.extend(p)

    pd.DataFrame(global_rows).to_csv(global_tsv, sep="\t", index=False)
    pd.DataFrame(per_chain_rows).to_csv(per_chain_tsv, sep="\t", index=False)
    pd.DataFrame(per_pair_rows).to_csv(per_pair_tsv, sep="\t", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
