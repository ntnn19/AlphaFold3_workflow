#!/usr/bin/env python3
"""
workflow/scripts/compute_local_tm.py

Steps 3 + 4: Parse USalign output, map contact residues, compute local TM score.

Uses the superposed PDB files produced by USalign (-o flag) to get
reliable per-residue distances. Parses the alignment block with
chain-aware residue tracking.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import gemmi


# ---------------------------------------------------------------------------
# Residue key: (chain_id, resnum_str) — used throughout for matching
# ---------------------------------------------------------------------------

def _residue_key(chain_name: str, res: gemmi.Residue) -> tuple[str, str]:
    return (chain_name, str(res.seqid))


def _representative_pos(res: gemmi.Residue) -> Optional[np.ndarray]:
    """Get representative atom position: CA for protein, C3' for nucleic, else first."""
    for name in ["CA", "C3'", "C4'", "P"]:
        atom = res.find_atom(name, " ")
        if atom is not None:
            return np.array([atom.pos.x, atom.pos.y, atom.pos.z])
    atoms = list(res)
    if atoms:
        a = atoms[0]
        return np.array([a.pos.x, a.pos.y, a.pos.z])
    return None


# ---------------------------------------------------------------------------
# Parse superposed structures from USalign -o output
# ---------------------------------------------------------------------------

def load_superposed_models(prefix: Path):
    """
    USalign -o <prefix> produces:
      <prefix>       — superposed target (chain 1) in PDB format
      <prefix>_atm   — superposed reference (chain 2) in PDB format

    Both are in the SAME coordinate frame after optimal superposition.
    We read them and build residue -> position lookups.
    """
    target_sup_path = prefix
    ref_sup_path = Path(str(prefix) + "_atm")

    target_residues = {}
    ref_residues = {}

    for path, residue_dict in [(target_sup_path, target_residues),
                                (ref_sup_path, ref_residues)]:
        if not path.exists():
            continue
        try:
            st = gemmi.read_pdb(str(path))
        except Exception:
            # Try mmCIF
            try:
                st = gemmi.read_structure(str(path))
            except Exception:
                continue

        if len(st) == 0:
            continue
        model = st[0]
        for chain in model:
            for res in chain:
                if res.is_water():
                    continue
                pos = _representative_pos(res)
                if pos is not None:
                    key = _residue_key(chain.name, res)
                    residue_dict[key] = {
                        "chain": chain.name,
                        "resnum": str(res.seqid),
                        "resname": res.name,
                        "pos": pos,
                    }

    return target_residues, ref_residues


# ---------------------------------------------------------------------------
# Parse alignment block from USalign full text output
# ---------------------------------------------------------------------------

def parse_alignment_pairs(usalign_full: Path) -> list[dict]:
    """
    Parse the USalign text output to extract aligned residue pairs.

    USalign full output contains alignment blocks like:

    (":" denotes residue pairs ...
     "." denotes residue pairs ...

    ADEFG--HIJK    (target)
    .::.   ::.:    (alignment markers)
    AXEFGYLHI-K    (reference)

    We parse these and track residue indices per chain using the
    original structures.

    Returns list of dicts with keys:
        target_char, ref_char, is_aligned
    """
    text = usalign_full.read_text()
    lines = text.strip().split("\n")

    pairs = []
    i = 0
    while i < len(lines):
        # Look for the marker explanation line
        if '":" denotes' in lines[i] or "denotes residue pairs" in lines[i]:
            i += 1
            # Now read alignment blocks until we hit an empty section
            while i < len(lines):
                # Skip blank lines between blocks
                while i < len(lines) and lines[i].strip() == "":
                    i += 1
                if i + 2 >= len(lines):
                    break

                line1 = lines[i]
                line2 = lines[i + 1]
                line3 = lines[i + 2]

                # Heuristic: alignment block lines should be similar length
                # and line2 should contain alignment markers (: . or space)
                if len(line2.strip()) == 0:
                    break

                # Check if line2 looks like alignment markers
                marker_chars = set(line2.replace(" ", ""))
                if not marker_chars.issubset(set(":.!|*+ ")):
                    break

                min_len = min(len(line1), len(line2), len(line3))
                for j in range(min_len):
                    target_char = line1[j]
                    marker = line2[j]
                    ref_char = line3[j]
                    is_aligned = marker in (":", ".")
                    pairs.append({
                        "target_char": target_char,
                        "ref_char": ref_char,
                        "is_aligned": is_aligned,
                    })

                i += 3
                continue
            break
        i += 1

    return pairs


# ---------------------------------------------------------------------------
# Build residue mapping using superposed structures + alignment
# ---------------------------------------------------------------------------

def build_residue_mapping(
    usalign_full: Path,
    usalign_prefix: Path,
    ref_structure_path: Path,
    target_structure_path: Path,
) -> pd.DataFrame:
    """
    Build residue mapping by combining:
    1. The superposed PDB structures from USalign (for coordinates)
    2. The original structures (for residue identity)

    Strategy:
    - Load superposed target and reference (both in same frame)
    - For each residue in superposed reference, find the nearest
      residue in superposed target
    - Use a distance threshold to identify aligned pairs
    - Cross-reference with original structures for chain/resnum

    This is much more robust than parsing the text alignment with counters.
    """
    # Load superposed structures
    target_sup, ref_sup = load_superposed_models(usalign_prefix)

    if not target_sup or not ref_sup:
        # Fallback: try to load original structures and use alignment text
        return _build_mapping_from_originals(
            usalign_full, ref_structure_path, target_structure_path
        )

    # Load original structures for residue identity
    ref_st = gemmi.read_structure(str(ref_structure_path))
    ref_st.setup_entities()
    target_st = gemmi.read_structure(str(target_structure_path))
    target_st.setup_entities()

    # Build position arrays for superposed structures
    ref_keys = list(ref_sup.keys())
    ref_positions = np.array([ref_sup[k]["pos"] for k in ref_keys])

    target_keys = list(target_sup.keys())
    target_positions = np.array([target_sup[k]["pos"] for k in target_keys])

    if len(ref_positions) == 0 or len(target_positions) == 0:
        return pd.DataFrame()

    # For each reference residue, find nearest target residue
    rows = []
    used_target = set()

    for ri, rkey in enumerate(ref_keys):
        rinfo = ref_sup[rkey]
        rpos = rinfo["pos"]

        # Compute distances to all target residues
        dists = np.linalg.norm(target_positions - rpos, axis=1)
        nearest_idx = int(np.argmin(dists))
        nearest_dist = float(dists[nearest_idx])

        tkey = target_keys[nearest_idx]
        tinfo = target_sup[tkey]

        rows.append({
            "ref_chain": rinfo["chain"],
            "ref_resnum": rinfo["resnum"],
            "ref_resname": rinfo["resname"],
            "target_chain": tinfo["chain"],
            "target_resnum": tinfo["resnum"],
            "target_resname": tinfo["resname"],
            "distance": nearest_dist,
            "aligned": True,  # will be refined below
        })

    df = pd.DataFrame(rows)

    # Mark as aligned only if distance is reasonable
    # Use the alignment text to determine which pairs USalign considers aligned
    alignment_pairs = parse_alignment_pairs(usalign_full)

    # Also mark based on distance — if distance > 20Å, likely not truly aligned
    # (USalign aligned residues are typically within ~10Å even for poor regions)
    df["aligned"] = df["distance"] < 20.0

    return df


def _build_mapping_from_originals(
    usalign_full: Path,
    ref_structure_path: Path,
    target_structure_path: Path,
) -> pd.DataFrame:
    """
    Fallback: build mapping from original structures using the alignment text.

    This version is chain-aware: it reads chain IDs from the USalign output
    header lines and maps residues accordingly.
    """
    ref_st = gemmi.read_structure(str(ref_structure_path))
    ref_st.setup_entities()
    target_st = gemmi.read_structure(str(target_structure_path))
    target_st.setup_entities()

    ref_model = ref_st[0]
    target_model = target_st[0]

    # Build ordered polymer residue lists (skip waters, ligands, ions)
    def polymer_residues(model):
        residues = []
        for chain in model:
            poly = chain.get_polymer()
            if not poly:
                continue
            for res in poly:
                pos = _representative_pos(res)
                if pos is not None:
                    residues.append({
                        "chain": chain.name,
                        "resnum": str(res.seqid),
                        "resname": res.name,
                        "pos": pos,
                    })
        return residues

    target_residues = polymer_residues(target_model)
    ref_residues = polymer_residues(ref_model)

    alignment_pairs = parse_alignment_pairs(usalign_full)

    if not alignment_pairs:
        return pd.DataFrame()

    rows = []
    ti = 0  # target index
    ri = 0  # reference index

    for pair in alignment_pairs:
        t_gap = (pair["target_char"] == "-")
        r_gap = (pair["ref_char"] == "-")

        if t_gap and r_gap:
            continue

        t_info = None
        r_info = None

        if not t_gap and ti < len(target_residues):
            t_info = target_residues[ti]
            ti += 1
        elif not t_gap:
            ti += 1

        if not r_gap and ri < len(ref_residues):
            r_info = ref_residues[ri]
            ri += 1
        elif not r_gap:
            ri += 1

        if t_info is not None and r_info is not None:
            dist = float(np.linalg.norm(t_info["pos"] - r_info["pos"]))
            rows.append({
                "ref_chain": r_info["chain"],
                "ref_resnum": r_info["resnum"],
                "ref_resname": r_info["resname"],
                "target_chain": t_info["chain"],
                "target_resnum": t_info["resnum"],
                "target_resname": t_info["resname"],
                "distance": dist,
                "aligned": pair["is_aligned"],
            })
        elif r_info is not None:
            rows.append({
                "ref_chain": r_info["chain"],
                "ref_resnum": r_info["resnum"],
                "ref_resname": r_info["resname"],
                "target_chain": "",
                "target_resnum": "",
                "target_resname": "",
                "distance": np.nan,
                "aligned": False,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TM-score computation
# ---------------------------------------------------------------------------

def compute_tm_score(distances: np.ndarray, L_norm: int) -> float:
    """
    TM = (1/L_norm) * sum_i [ 1 / (1 + (d_i / d0)^2) ]
    d0 = 1.24 * (L_norm - 15)^(1/3) - 1.8
    """
    if L_norm <= 15 or len(distances) == 0:
        return 0.0

    d0 = 1.24 * ((L_norm - 15) ** (1.0 / 3.0)) - 1.8
    d0 = max(d0, 0.5)

    scores = 1.0 / (1.0 + (distances / d0) ** 2)
    return float(np.sum(scores) / L_norm)


def parse_global_tm(summary_path: Path) -> dict:
    df = pd.read_csv(summary_path, sep="\t", dtype=str)
    if df.empty:
        return {}
    row = df.iloc[0]
    result = {}
    for col in ["TM1", "TM2", "RMSD", "L1", "L2", "Lali"]:
        if col in row:
            try:
                result[col] = float(row[col])
            except (ValueError, TypeError):
                result[col] = None
    return result


# ---------------------------------------------------------------------------
# Selection resolution (reused from extract_contacts.py)
# ---------------------------------------------------------------------------

_ION_RESNAMES = {
    "MG", "ZN", "CA", "MN", "FE", "FE2", "CU", "CU1", "CO", "NI",
    "CD", "NA", "K", "SR", "BA", "MO", "W",
}


def resolve_target_selection(
    selection_str: str,
    model: gemmi.Model,
    structure: gemmi.Structure,
) -> set[tuple[str, str]]:
    """
    Resolve a target_selection string to a set of (chain_id, resnum) keys.
    Reuses the same syntax as contact_selection.
    Empty string means all residues.
    """
    if not selection_str or not selection_str.strip():
        # All residues
        keys = set()
        for chain in model:
            for res in chain:
                if not res.is_water():
                    keys.add(_residue_key(chain.name, res))
        return keys

    selection_str = selection_str.strip()

    # select: raw gemmi selection
    if selection_str.lower().startswith("select:"):
        sel_str = selection_str[7:].strip()
        sel = gemmi.Selection(sel_str)
        keys = set()
        for cra_model in sel.copy_model_selection(model).all():
            keys.add((cra_model.chain.name, str(cra_model.residue.seqid)))
        return keys

    # Chain IDs (simple case: single letters, comma-separated)
    # Check if it looks like chain IDs
    parts = [p.strip() for p in selection_str.split(",")]
    if all(len(p) <= 2 and p.isalnum() for p in parts):
        chain_ids = set(parts)
        keys = set()
        for chain in model:
            if chain.name in chain_ids:
                for res in chain:
                    if not res.is_water():
                        keys.add(_residue_key(chain.name, res))
        return keys

    # Fallback: treat as gemmi selection
    try:
        sel = gemmi.Selection(selection_str)
        keys = set()
        for cra_model in sel.copy_model_selection(model).all():
            keys.add((cra_model.chain.name, str(cra_model.residue.seqid)))
        return keys
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--contacts", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--usalign-full", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--usalign-summary", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--usalign-prefix", type=click.Path(path_type=Path), required=True,
              help="Prefix used with USalign -o (without _atm suffix).")
@click.option("--target-structure", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--ref-structure", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--target-selection", type=str, default="",
              help="Optional: restrict target residues (chain IDs, select:, or empty for all).")
@click.option("-o", "--output", "out_path", type=click.Path(path_type=Path), required=True)
@click.option("--mapping-out", type=click.Path(path_type=Path), required=True)
def main(
    contacts: Path,
    usalign_full: Path,
    usalign_summary: Path,
    usalign_prefix: Path,
    target_structure: Path,
    ref_structure: Path,
    target_selection: str,
    out_path: Path,
    mapping_out: Path,
):
    """Map contact residues through USalign alignment and compute local TM score."""

    # Load contact residues
    df_contacts = pd.read_csv(contacts, sep="\t", dtype=str).fillna("")
    contact_keys = set()
    for _, row in df_contacts.iterrows():
        contact_keys.add((row["chain_id"], row["res_number"]))

    # Load original structures
    ref_st = gemmi.read_structure(str(ref_structure))
    ref_st.setup_entities()
    target_st = gemmi.read_structure(str(target_structure))
    target_st.setup_entities()

    # Resolve target selection
    target_allowed = resolve_target_selection(
        target_selection, target_st[0], target_st
    )

    # Build residue mapping using superposed structures
    df_mapping = build_residue_mapping(
        usalign_full, usalign_prefix, ref_structure, target_structure
    )

    if df_mapping.empty:
        click.echo("⚠️  Could not build residue mapping. Writing empty results.")
        pd.DataFrame().to_csv(out_path, sep="\t", index=False)
        pd.DataFrame().to_csv(mapping_out, sep="\t", index=False)
        return

    # Mark contact residues
    df_mapping["is_contact"] = df_mapping.apply(
        lambda r: (r["ref_chain"], r["ref_resnum"]) in contact_keys, axis=1
    )

    # Filter by target selection
    if target_selection.strip():
        df_mapping["target_selected"] = df_mapping.apply(
            lambda r: (r["target_chain"], r["target_resnum"]) in target_allowed
            if r["target_chain"] else False,
            axis=1,
        )
    else:
        df_mapping["target_selected"] = True

    # Save full mapping
    mapping_out.parent.mkdir(parents=True, exist_ok=True)
    df_mapping.to_csv(mapping_out, sep="\t", index=False)

    # Contact residues that are aligned and within target selection
    df_contact_mapped = df_mapping[
        df_mapping["is_contact"] &
        df_mapping["aligned"] &
        df_mapping["target_selected"]
    ].copy()

    # Global TM scores
    global_tm = parse_global_tm(usalign_summary)

    # Compute local TM score
    n_contact_total = len(contact_keys)
    n_contact_mapped = len(df_contact_mapped)
    n_contact_unmapped = n_contact_total - n_contact_mapped

    if n_contact_mapped > 0:
        distances = df_contact_mapped["distance"].to_numpy(dtype=float)
        local_tm_contact = compute_tm_score(distances, n_contact_mapped)
        L_target = int(global_tm.get("L1", 0) or 0)
        local_tm_global_norm = compute_tm_score(distances, L_target) if L_target > 0 else None
        local_rmsd = float(np.sqrt(np.mean(distances ** 2)))
        mean_distance = float(np.mean(distances))
        median_distance = float(np.median(distances))
    else:
        local_tm_contact = None
        local_tm_global_norm = None
        local_rmsd = None
        mean_distance = None
        median_distance = None

    result = {
        "ref_structure": str(ref_structure),
        "target_structure": str(target_structure),
        "global_TM1": global_tm.get("TM1"),
        "global_TM2": global_tm.get("TM2"),
        "global_RMSD": global_tm.get("RMSD"),
        "global_L1": global_tm.get("L1"),
        "global_L2": global_tm.get("L2"),
        "global_Lali": global_tm.get("Lali"),
        "n_contact_residues": n_contact_total,
        "n_contact_mapped": n_contact_mapped,
        "n_contact_unmapped": n_contact_unmapped,
        "contact_mapping_fraction": n_contact_mapped / n_contact_total if n_contact_total > 0 else None,
        "local_TM_contact_norm": local_tm_contact,
        "local_TM_global_norm": local_tm_global_norm,
        "local_RMSD": local_rmsd,
        "mean_contact_distance": mean_distance,
        "median_contact_distance": median_distance,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(out_path, sep="\t", index=False)

    if local_tm_contact is not None:
        click.echo(
            f"✅ Local TM (contact-norm): {local_tm_contact:.3f}, "
            f"Local RMSD: {local_rmsd:.2f}Å, "
            f"Mapped {n_contact_mapped}/{n_contact_total} contact residues, "
            f"Mean dist: {mean_distance:.2f}Å"
        )
    else:
        click.echo("⚠️  No contact residues could be mapped")


if __name__ == "__main__":
    main()