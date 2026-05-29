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


# ── Per-sample processing ─────────────────────────────────────────────────────

def process_sample(
    job_name: str,
    seed: int,
    sample: int,
    subdir: Path,
    cif_path: Path,
) -> tuple[dict, list[dict], list[dict]]:
    """
    Returns (global_row, per_chain_rows, per_pair_rows) for one sample.
    """
    prefix = f"{job_name}_seed-{seed}_sample-{sample}"

    summary_path = subdir / f"{prefix}_summary_confidences.json"
    conf_path    = subdir / f"{prefix}_confidences.json"

    with open(summary_path) as f:
        summary = json.load(f)
    with open(conf_path) as f:
        conf = json.load(f)

    chain_ids = _chain_ids_from_confidences(conf)
    mean_plddt_by_chain = _mean_plddt_per_chain(conf, chain_ids)

    id_cols = {
        "name":   job_name,
        "seed":   seed,
        "sample": sample,
        "file":   str(cif_path),
    }

    # ── Global row ────────────────────────────────────────────────────────────
    global_row = {
        **id_cols,
        "ranking_score":      summary["ranking_score"],
        "iptm":               summary["iptm"],
        "ptm":                summary["ptm"],
        "mean_plddt":         _mean_plddt_global(conf),
        "fraction_disordered": summary["fraction_disordered"],
        "has_clash":          summary["has_clash"],
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
            "chain_i":          chain_ids[i],
            "chain_j":          chain_ids[j],
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
    job_name = inference_dir.name

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
        cif_path = subdir / f"{job_name}_seed-{seed}_sample-{sample}_model.cif"

        g, c, p = process_sample(job_name, seed, sample, subdir, cif_path)
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
