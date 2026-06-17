"""Aggregate AF3 per-sample confidence outputs into three TSV tables.

Usage:
    python aggregate_results.py --inference_dir <dir> --out_dir <dir>

Writes per-sample TSVs into --out_dir named:
    {job}_seed-{seed}_sample-{sample}_global.tsv
    {job}_seed-{seed}_sample-{sample}_per_chain.tsv
    {job}_seed-{seed}_sample-{sample}_per_chain_pair.tsv

The per_chain_pair TSV merges AF3 confidences.json chain-pair metrics with
ipSAE/pDockQ/LIS scores from the _model_10_15.txt and _model_15_15.txt files.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ── Helpers ───────────────────────────────────────────────────────────────────


def _unique_ordered(seq):
    seen = {}
    for x in seq:
        seen.setdefault(x, None)
    return list(seen)


def _parse_seed_sample(subdir_name: str) -> tuple[int, int]:
    m = re.fullmatch(r"seed-(\d+)_sample-(\d+)", subdir_name)
    if not m:
        raise ValueError(f"Unexpected subdir name: {subdir_name!r}")
    return int(m.group(1)), int(m.group(2))


def _chain_ids_from_confidences(conf: dict) -> list[str]:
    return _unique_ordered(conf["token_chain_ids"])


def _mean_plddt_global(conf: dict) -> float:
    return float(np.mean(conf["atom_plddts"]))


def _mean_plddt_per_chain(conf: dict, chain_ids: list[str]) -> dict[str, float]:
    atom_plddts = np.array(conf["atom_plddts"], dtype=np.float64)
    atom_chain_ids = conf["atom_chain_ids"]
    result = {}
    for chain_id in chain_ids:
        mask = np.array([c == chain_id for c in atom_chain_ids], dtype=bool)
        result[chain_id] = (
            float(np.mean(atom_plddts[mask])) if mask.any() else float("nan")
        )
    return result


def _resolve_sample_paths(
    subdir: Path, job_name: str, seed: int, sample: int
) -> tuple[Path, Path, Path, Path, Path]:
    """Return (cif, summary_confidences, confidences, txt_10_15, txt_15_15)."""
    prefix = f"{job_name}_seed-{seed}_sample-{sample}"

    new_cif = subdir / f"{prefix}_model.cif"
    new_summary = subdir / f"{prefix}_summary_confidences.json"
    new_conf = subdir / f"{prefix}_confidences.json"
    new_10_15 = subdir / f"{prefix}_model_10_15.txt"
    new_15_15 = subdir / f"{prefix}_model_15_15.txt"

    if new_cif.exists() and new_summary.exists() and new_conf.exists():
        return new_cif, new_summary, new_conf, new_10_15, new_15_15

    old_cif = subdir / "model.cif"
    old_summary = subdir / "summary_confidences.json"
    old_conf = subdir / "confidences.json"
    old_10_15 = subdir / "model_10_15.txt"
    old_15_15 = subdir / "model_15_15.txt"

    if old_cif.exists() and old_summary.exists() and old_conf.exists():
        return old_cif, old_summary, old_conf, old_10_15, old_15_15

    raise FileNotFoundError(
        f"Could not find AF3 output files in {subdir} under either naming convention.\n"
        f"  New: {new_cif.name}, {new_summary.name}, {new_conf.name}\n"
        f"  Old: {old_cif.name}, {old_summary.name}, {old_conf.name}"
    )


def _read_ipsae_txt(path: Path) -> pd.DataFrame | None:
    """Read a _model_{pae}_{dist}.txt file, returning one row per (Chn1, Chn2, Type)."""
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\s+", skiprows=1)
    return df


# ── Per-sample processing ─────────────────────────────────────────────────────


def process_sample(
    job_name: str, base_name: str, seed: int, sample: int, subdir: Path
) -> tuple[dict, list[dict], pd.DataFrame, pd.DataFrame]:
    cif_path, summary_path, conf_path, txt_10_15, txt_15_15 = _resolve_sample_paths(
        subdir, job_name, seed, sample
    )

    with open(summary_path) as f:
        summary = json.load(f)
    with open(conf_path) as f:
        conf = json.load(f)

    chain_ids = _chain_ids_from_confidences(conf)
    mean_plddt_by_chain = _mean_plddt_per_chain(conf, chain_ids)

    id_cols = {"name": base_name, "seed": seed, "sample": sample, "file": str(cif_path)}

    # ── Global row ────────────────────────────────────────────────────────────
    global_row = {
        **id_cols,
        "ranking_score": summary["ranking_score"],
        "iptm": summary["iptm"],
        "ptm": summary["ptm"],
        "mean_plddt": _mean_plddt_global(conf),
        "fraction_disordered": summary["fraction_disordered"],
        "has_clash": summary["has_clash"],
    }

    # ── Per-chain rows ────────────────────────────────────────────────────────
    per_chain_rows = [
        {
            **id_cols,
            "chain_id": chain_id,
            "mean_plddt": mean_plddt_by_chain[chain_id],
            "chain_ptm": summary["chain_ptm"][i],
            "chain_iptm": summary["chain_iptm"][i],
        }
        for i, chain_id in enumerate(chain_ids)
    ]

    # ── Per-chain-pair: AF3 confidences ──────────────────────────────────────
    pair_rows = [
        {
            **id_cols,
            "chain_i": chain_ids[i],
            "chain_j": chain_ids[j],
            "chain_pair_pae_min": summary["chain_pair_pae_min"][i][j],
            "chain_pair_iptm": summary["chain_pair_iptm"][i][j],
        }
        for i in range(len(chain_ids))
        for j in range(len(chain_ids))
    ]
    per_pair_df = pd.DataFrame(pair_rows)

    # ── Merge ipSAE outputs ───────────────────────────────────────────────────
    ipsae_dfs = []
    for path in (txt_10_15, txt_15_15):
        df = _read_ipsae_txt(path)
        if df is not None:
            ipsae_dfs.append(df)

    if ipsae_dfs:
        ipsae_all = pd.concat(ipsae_dfs, ignore_index=True)
        ipsae_all["seed"] = ipsae_all.Model.apply(
            lambda x: f"{re.sub(r'seed-', '', Path(x).parent.name)}"
        ).apply(lambda x: f"{re.sub(r'_sample-\d+', '', x)}")
        ipsae_all["sample"] = ipsae_all.Model.apply(
            lambda x: f"{re.sub(r'seed-\d+_sample-', '', Path(x).parent.name)}"
        )
        ipsae_all["name"] = base_name
        # ipsae_all["group_key"] = ipsae_all.Model.apply(lambda x: f"{Path(x).parents[1].name}_{Path(x).parent.name.split('_')[-1]}")
    else:
        ipsae_columns = [
            "Chn1",
            "Chn2",
            "PAE",
            "Dist",
            "Type",
            "ipSAE",
            "ipSAE_d0chn",
            "ipSAE_d0dom",
            "ipTM_af",
            "ipTM_d0chn",
            "pDockQ",
            "pDockQ2",
            "LIS",
            "n0res",
            "n0chn",
            "n0dom",
            "d0res",
            "d0chn",
            "d0dom",
            "nres1",
            "nres2",
            "dist1",
            "dist2",
            "Model",
            "seed",
            "sample",
            "name",
        ]
        ipsae_all = pd.DataFrame.from_records(data=[], columns=ipsae_columns)

    return global_row, per_chain_rows, per_pair_df, ipsae_all


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    inference_dir = Path(args.inference_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    job_name = inference_dir.name
    base_name = re.sub(r"_seed-\d+$", "", job_name)

    subdirs = sorted(
        d
        for d in inference_dir.iterdir()
        if d.is_dir() and re.fullmatch(r"seed-\d+_sample-\d+", d.name)
    )
    if not subdirs:
        raise FileNotFoundError(
            f"No seed-*_sample-* subdirectories found in {inference_dir}"
        )

    #    global_rows, per_chain_rows, per_pair_dfs, ipsae_dfs = [], [], [], []

    for subdir in subdirs:
        seed, sample = _parse_seed_sample(subdir.name)
        global_, per_chain, per_chain_pair, ipase = process_sample(
            job_name, base_name, seed, sample, subdir
        )
        stem = f"{job_name}_seed-{seed}_sample-{sample}"
        pd.DataFrame([global_]).to_csv(
            out_dir / f"{stem}_af_global.tsv", sep="\t", index=False
        )
        pd.DataFrame(per_chain).to_csv(
            out_dir / f"{stem}_af_per_chain.tsv", sep="\t", index=False
        )
        pd.DataFrame(per_chain_pair).to_csv(
            out_dir / f"{stem}_af_per_chain_pair.tsv", sep="\t", index=False
        )
        pd.DataFrame(ipase).to_csv(out_dir / f"{stem}_ipsae.tsv", sep="\t", index=False)

    # stem = job_name
    # pd.DataFrame(global_rows).to_csv(
    #    out_dir / f"{stem}_af_global.tsv", sep="\t", index=False
    # )
    # pd.DataFrame(per_chain_rows).to_csv(
    #    out_dir / f"{stem}_af_per_chain.tsv", sep="\t", index=False
    # )
    # pd.concat(per_pair_dfs, ignore_index=True).to_csv(
    #    out_dir / f"{stem}_af_per_chain_pair.tsv", sep="\t", index=False
    # )
    # pd.concat(ipsae_dfs, ignore_index=True).to_csv(
    #    out_dir / f"{stem}_ipsae.tsv", sep="\t", index=False
    # )


if __name__ == "__main__":
    main()
