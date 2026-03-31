#!/usr/bin/env python3
"""
workflow/scripts/extract_contacts.py

Step 1: Extract residues within distance_cutoff of the contact_selection
from a reference structure.

Output: TSV with columns chain_id, res_number, res_name
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import click
import numpy as np
import gemmi


# ---- Selection resolution ------------------------------------------------

_ENTITY_KEYWORDS = {"rna", "dna", "protein", "ligand", "ion"}

_ION_RESNAMES = {
    "MG", "ZN", "CA", "MN", "FE", "FE2", "CU", "CU1", "CO", "NI",
    "CD", "NA", "K", "SR", "BA", "MO", "W",
}


def _is_chain_type(chain: gemmi.Chain, kind: str, structure: gemmi.Structure) -> bool:
    """Check whether a chain matches a high-level entity type."""
    polymer = chain.get_polymer()
    if kind == "protein":
        return polymer.check_polymer_type() in (
            gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD,
        ) if polymer else False
    if kind == "rna":
        return polymer.check_polymer_type() == gemmi.PolymerType.Rna if polymer else False
    if kind == "dna":
        return polymer.check_polymer_type() in (
            gemmi.PolymerType.Dna, gemmi.PolymerType.DnaRnaHybrid,
        ) if polymer else False
    # ligand / ion fall through to residue-level checks
    return False


def _atoms_for_token(token: str, model: gemmi.Model, structure: gemmi.Structure) -> list[gemmi.Atom]:
    """Resolve a single selection token to a list of atoms."""
    token = token.strip()

    # select: raw — pass to gemmi Selection
    if token.lower().startswith("select:"):
        sel_str = token[7:].strip()
        sel = gemmi.Selection(sel_str)
        atoms = []
        for cra in sel.copy_model_selection(model).all():
            atoms.append(cra.atom)
        return atoms

    # resname:XXX,YYY
    if token.lower().startswith("resname:"):
        names = {n.strip().upper() for n in token[8:].split(",")}
        atoms = []
        for chain in model:
            for res in chain:
                if res.name.upper() in names:
                    for atom in res:
                        atoms.append(atom)
        return atoms

    # entity type keyword
    if token.lower() in _ENTITY_KEYWORDS:
        kind = token.lower()
        atoms = []
        for chain in model:
            if kind == "ligand":
                for res in chain:
                    if not res.is_water() and res.name.upper() not in _ION_RESNAMES:
                        if res.het_flag == "H" or (not chain.get_polymer()):
                            for atom in res:
                                atoms.append(atom)
            elif kind == "ion":
                for res in chain:
                    if res.name.upper() in _ION_RESNAMES:
                        for atom in res:
                            atoms.append(atom)
            elif _is_chain_type(chain, kind, structure):
                for res in chain:
                    for atom in res:
                        atoms.append(atom)
        return atoms

    # Chain ID(s): single uppercase letters, comma-separated
    chain_ids = {c.strip() for c in token.split(",")}
    atoms = []
    for chain in model:
        if chain.name in chain_ids:
            for res in chain:
                for atom in res:
                    atoms.append(atom)
    return atoms


def resolve_selection(selection_str: str, model: gemmi.Model, structure: gemmi.Structure) -> list[gemmi.Atom]:
    """
    Parse a contact_selection string and return matching atoms.

    Supports:
      - Chain IDs: "C" or "C,D"
      - Entity types: RNA, DNA, protein, ligand, ion
      - Residue names: resname:ATP or resname:ATP,MG
      - Compound: RNA+ion, C,D+resname:MG
      - Raw: select:<gemmi selection string>
    """
    tokens = selection_str.split("+")
    all_atoms = []
    for tok in tokens:
        all_atoms.extend(_atoms_for_token(tok, model, structure))
    return all_atoms


def extract_contact_residues(
    structure: gemmi.Structure,
    model: gemmi.Model,
    contact_atoms: list[gemmi.Atom],
    distance_cutoff: float,
) -> list[dict]:
    """
    Find residues NOT in the contact selection that have at least one atom
    within distance_cutoff of any contact atom.
    """
    if not contact_atoms:
        return []

    # Build array of contact atom positions
    contact_pos = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in contact_atoms])

    # Collect contact atom chain+residue IDs to exclude them
    contact_res_keys = set()
    # We need to walk the model to find which chain/residue each atom belongs to
    atom_id_to_chain_res = {}
    for chain in model:
        for res in chain:
            for atom in res:
                atom_id_to_chain_res[id(atom)] = (chain.name, str(res.seqid), res.name)

    for a in contact_atoms:
        key = atom_id_to_chain_res.get(id(a))
        if key:
            contact_res_keys.add((key[0], key[1]))

    # Find nearby residues
    found = {}  # (chain, seqid) -> res_name
    for chain in model:
        for res in chain:
            res_key = (chain.name, str(res.seqid))
            if res_key in contact_res_keys:
                continue
            if res.is_water():
                continue

            for atom in res:
                pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                dists = np.linalg.norm(contact_pos - pos, axis=1)
                if np.min(dists) <= distance_cutoff:
                    found[res_key] = res.name
                    break  # one atom enough

    rows = []
    for (chain_id, seqid), res_name in sorted(found.items()):
        rows.append({
            "chain_id": chain_id,
            "res_number": seqid,
            "res_name": res_name,
        })
    return rows


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--structure", type=click.Path(exists=True, path_type=Path), required=True,
              help="Reference structure (mmCIF or PDB).")
@click.option("--contact-selection", type=str, required=True,
              help="Contact selection string (chain IDs, entity type, resname:, select:, or compound with +).")
@click.option("--distance-cutoff", type=float, required=True,
              help="Distance cutoff in Angstroms.")
@click.option("-o", "--output", "out_path", type=click.Path(path_type=Path), required=True,
              help="Output TSV with contact residues.")
def main(structure: Path, contact_selection: str, distance_cutoff: float, out_path: Path):
    """Extract residues within distance_cutoff of the contact selection."""
    import pandas as pd

    st = gemmi.read_structure(str(structure))
    st.setup_entities()
    model = st[0]

    contact_atoms = resolve_selection(contact_selection, model, st)
    if not contact_atoms:
        click.echo(f"⚠️  No atoms matched contact_selection='{contact_selection}'")

    rows = extract_contact_residues(st, model, contact_atoms, distance_cutoff)

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["chain_id", "res_number", "res_name"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)

    click.echo(f"✅ {len(rows)} contact residues written to {out_path}")


if __name__ == "__main__":
    main()