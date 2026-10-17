"""Extract AF3 confidence metrics and ipSAE scores for ONE sample.

Usage:
    python extract_scores.py \
        --inference_dir <.../rule_ALPHAFAST/batch_X/{job_name}> \
        --seed 1 --sample 0 \
        --global_tsv      out_af_global.tsv \
        --per_chain_tsv   out_af_per_chain.tsv \
        --per_pair_tsv    out_af_per_chain_pair.tsv \
        --ipsae_tsv       out_ipsae.tsv

The sample directory read is <inference_dir>/seed-{seed}_sample-{sample}.
The per-chain-pair table holds the AF3 chain-pair metrics; the ipSAE/pDockQ/LIS
scores from _model_10_15.txt and _model_15_15.txt go into the ipsae table.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

IPSAE_COLUMNS = [
    "Chn1", "Chn2", "PAE", "Dist", "Type", "ipSAE", "ipSAE_d0chn", "ipSAE_d0dom",
    "ipTM_af", "ipTM_d0chn", "pDockQ", "pDockQ2", "LIS", "n0res", "n0chn", "n0dom",
    "d0res", "d0chn", "d0dom", "nres1", "nres2", "dist1", "dist2", "Model",
    "seed", "sample", "name",
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def _unique_ordered(seq):
    seen = {}
    for x in seq:
        seen.setdefault(x, None)
    return list(seen)


def _chain_ids_from_confidences(conf: dict) -> list[str]:
    return _unique_ordered(conf["token_chain_ids"])


def _mean_plddt_global(conf: dict) -> float:
    return float(np.mean(conf["atom_plddts"]))


def _mean_plddt_per_chain(conf: dict, chain_ids: list[str]) -> dict[str, float]:
    atom_plddts = np.array(conf["atom_plddts"], dtype=np.float64)
    atom_chain_ids = np.array(conf["atom_chain_ids"])
    result = {}
    for chain_id in chain_ids:
        mask = atom_chain_ids == chain_id
        result[chain_id] = (
            float(np.mean(atom_plddts[mask])) if mask.any() else float("nan")
        )
    return result


def _resolve_sample_paths(
    subdir: Path, job_name: str, seed: int, sample: int
) -> tuple[Path, Path, Path, Path, Path]:
    """Return (cif, summary_confidences, confidences, txt_10_15, txt_15_15)."""
    prefix = f"{job_name}_seed-{seed}_sample-{sample}"

    new = (
        subdir / f"{prefix}_model.cif",
        subdir / f"{prefix}_summary_confidences.json",
        subdir / f"{prefix}_confidences.json",
        subdir / f"{prefix}_model_10_15.txt",
        subdir / f"{prefix}_model_15_15.txt",
    )
    if all(p.exists() for p in new[:3]):
        return new

    old = (
        subdir / "model.cif",
        subdir / "summary_confidences.json",
        subdir / "confidences.json",
        subdir / "model_10_15.txt",
        subdir / "model_15_15.txt",
    )
    if all(p.exists() for p in old[:3]):
        return old

    raise FileNotFoundError(
        f"Could not find AF3 output files in {subdir} under either naming convention.\n"
        f"  New: {', '.join(p.name for p in new[:3])}\n"
        f"  Old: {', '.join(p.name for p in old[:3])}"
    )


def _read_ipsae_txt(path: Path) -> pd.DataFrame | None:
    """Read a _model_{pae}_{dist}.txt file (one row per Chn1/Chn2/Type)."""
    if not path.exists():
        return None
    return pd.read_csv(path, sep=r"\s+", skiprows=1)


# ── Single-sample processing ──────────────────────────────────────────────────


def process_sample(
    job_name: str, base_name: str, seed: int, sample: int, subdir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    global_df = pd.DataFrame(
        [
            {
                **id_cols,
                "ranking_score": summary["ranking_score"],
                "iptm": summary["iptm"],
                "ptm": summary["ptm"],
                "mean_plddt": _mean_plddt_global(conf),
                "fraction_disordered": summary["fraction_disordered"],
                "has_clash": summary["has_clash"],
            }
        ]
    )

    per_chain_df = pd.DataFrame(
        [
            {
                **id_cols,
                "chain_id": chain_id,
                "mean_plddt": mean_plddt_by_chain[chain_id],
                "chain_ptm": summary["chain_ptm"][i],
                "chain_iptm": summary["chain_iptm"][i],
            }
            for i, chain_id in enumerate(chain_ids)
        ]
    )

    per_pair_df = pd.DataFrame(
        [
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
    )

    ipsae_dfs = [
        df for df in (_read_ipsae_txt(txt_10_15), _read_ipsae_txt(txt_15_15))
        if df is not None
    ]
    if ipsae_dfs:
        ipsae_df = pd.concat(ipsae_dfs, ignore_index=True)
        # seed/sample are known, no need to parse them out of the Model path
        ipsae_df["seed"] = seed
        ipsae_df["sample"] = sample
        ipsae_df["name"] = base_name
    else:
        ipsae_df = pd.DataFrame(columns=IPSAE_COLUMNS)

    return global_df, per_chain_df, per_pair_df, ipsae_df


# ── Main ──────────────────────────────────────────────────────────────────────


def _write(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference_dir", required=True,
                        help="Job directory, e.g. .../rule_ALPHAFAST/batch_1/<job_name>")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--global_tsv", required=True)
    parser.add_argument("--per_chain_tsv", required=True)
    parser.add_argument("--per_pair_tsv", required=True)
    parser.add_argument("--ipsae_tsv", required=True)
    args = parser.parse_args()

    inference_dir = Path(args.inference_dir)
    job_name = inference_dir.name
    base_name = re.sub(r"_seed-\d+$", "", job_name)

    subdir = inference_dir / f"seed-{args.seed}_sample-{args.sample}"
    if not subdir.is_dir():
        raise FileNotFoundError(f"Sample directory not found: {subdir}")

    global_df, per_chain_df, per_pair_df, ipsae_df = process_sample(
        job_name, base_name, args.seed, args.sample, subdir
    )

    _write(global_df, args.global_tsv)
    _write(per_chain_df, args.per_chain_tsv)
    _write(per_pair_df, args.per_pair_tsv)
    _write(ipsae_df, args.ipsae_tsv)


if __name__ == "__main__":
    main()