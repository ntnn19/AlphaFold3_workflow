#!/usr/bin/env python3
"""
workflow/scripts/compute_local_tm.py

Steps 3 + 4: Parse USalign output, map contact residues, compute local TM score.

The local TM score uses per-residue distances from the GLOBAL alignment
(no re-fitting) and normalizes by the number of contact residues.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import gemmi


def parse_usalign_alignment(usalign_full: Path) -> list[tuple[str, str, str]]:
    """
    Parse the structural alignment block from USalign full output.

    Returns list of (ref_token, alignment_char, target_token).
    Each token is a single residue character or '-' for gap.
    USalign outputs three lines:
      line 1: target sequence
      line 2: alignment markers
      line 3: reference sequence
    """
    text = usalign_full.read_text()
    lines = text.strip().split("\n")

    # Find alignment blocks — they follow "denotes residue pairs"
    alignment_triples = []
    i = 0
    while i < len(lines):
        if "denotes residue pairs" in lines[i]:
            # Next non-empty lines are the alignment block(s)
            i += 1
            while i < len(lines):
                # Skip empty lines
                while i < len(lines) and lines[i].strip() == "":
                    i += 1
                if i + 2 >= len(lines):
                    break
                target_seq = lines[i]
                align_marks = lines[i + 1]
                ref_seq = lines[i + 2]

                # Verify lengths match
                min_len = min(len(target_seq), len(align_marks), len(ref_seq))
                if min_len == 0:
                    i += 3
                    continue

                for j in range(min_len):
                    alignment_triples.append((
                        ref_seq[j],
                        align_marks[j],
                        target_seq[j],
                    ))
                i += 3

                # Check if next block continues
                if i < len(lines) and lines[i].strip() == "":
                    continue
                else:
                    break
        i += 1

    return alignment_triples


def parse_usalign_residue_mapping(
    usalign_full: Path,
    ref_structure: gemmi.Structure,
    target_structure: gemmi.Structure,
) -> pd.DataFrame:
    """
    Build a residue-level mapping from the USalign structural alignment.

    Returns DataFrame with columns:
        ref_chain, ref_resnum, ref_resname,
        target_chain, target_resnum, target_resname,
        aligned (bool), distance (float or NaN)
    """
    triples = parse_usalign_alignment(usalign_full)
    if not triples:
        return pd.DataFrame()

    # Build ordered residue lists for ref and target
    ref_model = ref_structure[0]
    target_model = target_structure[0]

    def ordered_residues(model):
        residues = []
        for chain in model:
            for res in chain:
                if res.is_water():
                    continue
                # Get CA or first atom position
                ca = res.find_atom("CA", " ")
                if ca is None:
                    # Try C3' for nucleic acids
                    ca = res.find_atom("C3'", " ")
                if ca is None:
                    # Fallback to first atom
                    atoms = list(res)
                    ca = atoms[0] if atoms else None
                if ca is not None:
                    residues.append({
                        "chain": chain.name,
                        "resnum": str(res.seqid),
                        "resname": res.name,
                        "pos": np.array([ca.pos.x, ca.pos.y, ca.pos.z]),
                    })
        return residues

    ref_residues = ordered_residues(ref_model)
    target_residues = ordered_residues(target_model)

    # Walk the alignment to pair residues
    ref_idx = 0
    target_idx = 0
    rows = []

    for ref_char, mark, target_char in triples:
        ref_gap = (ref_char == "-")
        target_gap = (target_char == "-")

        if ref_gap and target_gap:
            continue

        ref_info = None
        target_info = None

        if not ref_gap and ref_idx < len(ref_residues):
            ref_info = ref_residues[ref_idx]
            ref_idx += 1
        elif not ref_gap:
            ref_idx += 1

        if not target_gap and target_idx < len(target_residues):
            target_info = target_residues[target_idx]
            target_idx += 1
        elif not target_gap:
            target_idx += 1

        if ref_info is not None and target_info is not None:
            dist = float(np.linalg.norm(ref_info["pos"] - target_info["pos"]))
            rows.append({
                "ref_chain": ref_info["chain"],
                "ref_resnum": ref_info["resnum"],
                "ref_resname": ref_info["resname"],
                "target_chain": target_info["chain"],
                "target_resnum": target_info["resnum"],
                "target_resname": target_info["resname"],
                "aligned": True,
                "distance": dist,
            })
        elif ref_info is not None:
            rows.append({
                "ref_chain": ref_info["chain"],
                "ref_resnum": ref_info["resnum"],
                "ref_resname": ref_info["resname"],
                "target_chain": "",
                "target_resnum": "",
                "target_resname": "",
                "aligned": False,
                "distance": np.nan,
            })
        elif target_info is not None:
            rows.append({
                "ref_chain": "",
                "ref_resnum": "",
                "ref_resname": "",
                "target_chain": target_info["chain"],
                "target_resnum": target_info["resnum"],
                "target_resname": target_info["resname"],
                "aligned": False,
                "distance": np.nan,
            })

    return pd.DataFrame(rows)


def compute_tm_score(distances: np.ndarray, L_norm: int) -> float:
    """
    Compute TM-score from per-residue distances and normalization length.

    TM = (1/L_norm) * sum_i [ 1 / (1 + (d_i / d0)^2) ]
    d0 = 1.24 * (L_norm - 15)^(1/3) - 1.8
    """
    if L_norm <= 15 or len(distances) == 0:
        return 0.0

    d0 = 1.24 * ((L_norm - 15) ** (1.0 / 3.0)) - 1.8
    d0 = max(d0, 0.5)  # floor to avoid division issues

    scores = 1.0 / (1.0 + (distances / d0) ** 2)
    return float(np.sum(scores) / L_norm)


def parse_global_tm(summary_path: Path) -> dict:
    """Parse global TM scores from USalign -outfmt 2 output."""
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


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--contacts", type=click.Path(exists=True, path_type=Path), required=True,
              help="Contact residues TSV from extract_contacts.")
@click.option("--usalign-full", type=click.Path(exists=True, path_type=Path), required=True,
              help="USalign full output (default format).")
@click.option("--usalign-summary", type=click.Path(exists=True, path_type=Path), required=True,
              help="USalign summary TSV (-outfmt 2).")
@click.option("--target-structure", type=click.Path(exists=True, path_type=Path), required=True,
              help="Target structure file.")
@click.option("--ref-structure", type=click.Path(exists=True, path_type=Path), required=True,
              help="Reference structure file.")
@click.option("--target-chains", type=str, default="",
              help="Optional: restrict target to these chain IDs (comma-separated).")
@click.option("-o", "--output", "out_path", type=click.Path(path_type=Path), required=True,
              help="Output TSV with local TM results.")
@click.option("--mapping-out", type=click.Path(path_type=Path), required=True,
              help="Output TSV with full residue mapping.")
def main(
    contacts: Path,
    usalign_full: Path,
    usalign_summary: Path,
    target_structure: Path,
    ref_structure: Path,
    target_chains: str,
    out_path: Path,
    mapping_out: Path,
):
    """Map contact residues through USalign alignment and compute local TM score."""

    # Load contact residues
    df_contacts = pd.read_csv(contacts, sep="\t", dtype=str).fillna("")
    contact_keys = set()
    for _, row in df_contacts.iterrows():
        contact_keys.add((row["chain_id"], row["res_number"]))

    # Load structures
    ref_st = gemmi.read_structure(str(ref_structure))
    ref_st.setup_entities()
    target_st = gemmi.read_structure(str(target_structure))
    target_st.setup_entities()

    # Parse alignment and build residue mapping
    df_mapping = parse_usalign_residue_mapping(usalign_full, ref_st, target_st)

    if df_mapping.empty:
        click.echo("⚠️  Could not parse alignment. Writing empty results.")
        pd.DataFrame().to_csv(out_path, sep="\t", index=False)
        pd.DataFrame().to_csv(mapping_out, sep="\t", index=False)
        return

    # Mark which mapping rows correspond to contact residues
    df_mapping["is_contact"] = df_mapping.apply(
        lambda r: (r["ref_chain"], r["ref_resnum"]) in contact_keys, axis=1
    )

    # Filter by target_chains if specified
    if target_chains:
        allowed = {c.strip() for c in target_chains.split(",")}
        df_mapping["target_chain_ok"] = (
            df_mapping["target_chain"].isin(allowed) | (df_mapping["target_chain"] == "")
        )
    else:
        df_mapping["target_chain_ok"] = True

    # Save full mapping
    mapping_out.parent.mkdir(parents=True, exist_ok=True)
    df_mapping.to_csv(mapping_out, sep="\t", index=False)

    # Extract contact residue mapping
    df_contact_mapped = df_mapping[
        df_mapping["is_contact"] &
        df_mapping["aligned"] &
        df_mapping["target_chain_ok"]
    ].copy()

    df_contact_unmapped = df_mapping[
        df_mapping["is_contact"] &
        (~df_mapping["aligned"])
    ].copy()

    # Global TM scores
    global_tm = parse_global_tm(usalign_summary)

    # Compute local TM score
    n_contact_total = len(contact_keys)
    n_contact_mapped = len(df_contact_mapped)
    n_contact_unmapped = n_contact_total - n_contact_mapped

    if n_contact_mapped > 0:
        distances = df_contact_mapped["distance"].to_numpy(dtype=float)
        # Local TM normalized by number of contact residues
        local_tm_contact = compute_tm_score(distances, n_contact_mapped)
        # Local TM normalized by target length (for comparability with global)
        L_target = int(global_tm.get("L1", 0) or 0)
        local_tm_global_norm = compute_tm_score(distances, L_target) if L_target > 0 else None
        # Local RMSD
        local_rmsd = float(np.sqrt(np.mean(distances ** 2)))
    else:
        local_tm_contact = None
        local_tm_global_norm = None
        local_rmsd = None

    # Build result row
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
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(out_path, sep="\t", index=False)

    click.echo(
        f"✅ Local TM (contact-norm): {local_tm_contact:.3f}, "
        f"mapped {n_contact_mapped}/{n_contact_total} contact residues"
        if local_tm_contact is not None else
        f"⚠️  No contact residues could be mapped"
    )


if __name__ == "__main__":
    main()