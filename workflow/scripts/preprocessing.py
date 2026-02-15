# Adapted from https://github.com/Hanziwww/AlphaFold3-GUI/blob/main/afusion/api.py

from collections import defaultdict
import numpy as np
import os
import json
import pandas as pd
import string
from loguru import logger
import click
import itertools
from pathlib import Path
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



def has_multimers(df: pd.DataFrame) -> bool:
    has_multimers_ = (
            df.job_name.duplicated().any()
            and not df.empty
    )
    return has_multimers_


def expand_df(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()

    # Repeat each row n_samples times
    repeats = df["n_samples"]

    # Repeat input column
    fold_input_col = np.repeat(df[column].to_numpy(), repeats)

    # Build sample column (1, 2, 3, ... for each original row)
    sample_col = np.concatenate([
        np.arange(1, n + 1)
        for n in df["n_samples"]
    ])

    # Keep seed column as is (repeat it)
    seed_col = np.repeat(df["n_seeds"].to_numpy(), repeats)

    return pd.DataFrame({
        column: fold_input_col,
        "seed": seed_col,
        "sample": sample_col,
    })

def sanitised_name(name: str) -> str:
    """Returns sanitised version of the name that can be used as a filename."""
    name = str(name)
    lower_spaceless_name = name.lower().replace(' ', '_')
    allowed_chars = set(string.ascii_lowercase + string.digits + '_-.')
    return ''.join(l for l in lower_spaceless_name if l in allowed_chars)





def transform_vds_to_af3(df,n_seeds=None) -> pd.DataFrame:
    rows = []

    # Process by parent_job_id to reset molecule IDs (A, B, C...) per job
    for job_id, group in df.groupby('job_name'):
        # Generate ID sequence: A, B, C, D...
        letters = list(string.ascii_uppercase)
        letters = letters + [''.join(pair) for pair in itertools.product(letters, repeat=2)]
        id_generator = iter(letters)

        for _, row in group.iterrows():
            # Expand based on the 'count' column
            for _ in range(int(row.get('count',1))):
                entity_id = next(id_generator)

                # Logic to determine where 'data' goes
                data_val = str(row['data'])
                sequence = ""
                smiles = ""
                ccd_codes = ""
                bonded_atom_pairs = None
                user_ccd = None
                modifications = parse_json_field(row.get('modifications'))
                msa_option = row.get('msa_option', 'auto')
                unpaired_msa = row.get('unpaired_msa')
                paired_msa = row.get('paired_msa')
                templates = parse_json_field(row.get('templates'))
                # Job-level parameters (assumed consistent within group)
                if n_seeds is None and pd.notna(row.get('model_seeds')):
                    model_seeds = parse_list_field(row.get('model_seeds'), data_type=int)
                if bonded_atom_pairs is None and pd.notna(row.get('bonded_atom_pairs')):
                    bonded_atom_pairs = parse_json_field(row.get('bonded_atom_pairs'))
                if user_ccd is None and pd.notna(row.get('user_ccd')):
                    user_ccd = row.get('user_ccd')

                if row['type'] in ['protein', 'dna', 'rna']:
                    sequence = data_val
                elif row['type'] == 'ligand':
                    # Heuristic: if it looks like a SMILES (contains = or #), put in smiles
                    # Otherwise, assume it's a CCD code
                    if any(char in data_val for char in "=#()123"):
                        smiles = data_val
                    else:
                        ccd_codes = data_val

                if n_seeds is not None:
                    model_seeds = ",".join(list(map(lambda x: str(x),list(range(1, n_seeds + 1)))))
                elif not pd.notna(row.get('model_seeds')):
                    model_seeds = "1"  # default seed


                rows.append({
                    'job_name': job_id,
                    'type': row['type'],
                    'id': entity_id,
                    'sequence': sequence,
                    'modifications': modifications,
                    'ccd_codes': ccd_codes,
                    'smiles': smiles,
                    'msa_option': msa_option,
                    'unpaired_msa': unpaired_msa,
                    'paired_msa': paired_msa,
                    'templates': templates,
                    'model_seeds': model_seeds,
                    'bonded_atom_pairs': bonded_atom_pairs,
                    'user_ccd': user_ccd
                    # Default values for the columns seen in your screenshot

                })

    return pd.DataFrame(rows)


def transform_stoichio_screen_to_af3(df: pd.DataFrame, n_seeds: Optional[int] = None) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    rows = []
    summary_data = []

    # 1. Group by parent job name
    for parent_job, group in df.groupby('job_name'):

        sequences_metadata = []
        ranges = []
        monomer_labels = [f"monomer_{i + 1}" for i in range(len(group))]

        for _, row in group.iterrows():
            sequences_metadata.append(row.to_dict())

            # Parse count string (e.g., "1,5" or "1")
            count_val = str(row.get('count', '1'))
            if ',' in count_val:
                start, end = map(int, count_val.split(','))
                ranges.append(list(range(start, end + 1)))
            else:
                ranges.append([int(count_val)])

        # 2. Generate Cartesian Product of all stoichiometry possibilities
        for combo_idx, combo in enumerate(itertools.product(*ranges)):
            specific_job_id = f"{parent_job}_c{combo_idx}"

            # Populate Metadata Summary
            current_summary = {'job_name': specific_job_id, 'parent_job': parent_job}
            for i, count in enumerate(combo):
                label = monomer_labels[i]
                current_summary[label] = count
                # Fingerprint: First 10 chars
                f_seq = str(sequences_metadata[i]['data'])
                current_summary[f"{label}_prefix"] = (f_seq[:10] + '...') if len(f_seq) > 10 else f_seq
            summary_data.append(current_summary)

            # Reset Chain IDs (A, B, C...) for this combination
            letters = list(string.ascii_uppercase)
            letters = letters + [''.join(pair) for pair in itertools.product(letters, repeat=2)]
            id_generator = iter(letters)

            # 3. Expand chains based on counts using your specific logic
            for i, count in enumerate(combo):
                row = sequences_metadata[i]

                for _ in range(count):
                    entity_id = next(id_generator)

                    # --- YOUR REQUESTED LOGIC ---
                    data_val = str(row['data'])
                    sequence = ""
                    smiles = ""
                    ccd_codes = ""
                    bonded_atom_pairs = None
                    user_ccd = None
                    model_seeds = None
                    modifications = parse_json_field(row.get('modifications'))
                    msa_option = row.get('msa_option', 'auto')
                    unpaired_msa = row.get('unpaired_msa')
                    paired_msa = row.get('paired_msa')
                    templates = parse_json_field(row.get('templates'))

                    # Handle model_seeds: n_seeds arg takes priority over df column
                    model_seeds = n_seeds
                    if model_seeds is None and pd.notna(row.get('model_seeds')):
                        model_seeds = parse_list_field(row.get('model_seeds'), data_type=int)

                    if bonded_atom_pairs is None and pd.notna(row.get('bonded_atom_pairs')):
                        bonded_atom_pairs = parse_json_field(row.get('bonded_atom_pairs'))
                    if user_ccd is None and pd.notna(row.get('user_ccd')):
                        user_ccd = row.get('user_ccd')

                    if row['type'] in ['protein', 'dna', 'rna']:
                        sequence = data_val
                    elif row['type'] == 'ligand':
                        if any(char in data_val for char in "=#()123"):
                            smiles = data_val
                        else:
                            ccd_codes = data_val

                    if n_seeds is not None:
                        model_seeds = ",".join([str(i) for i in list(range(1, n_seeds + 1))])
                    #if not pd.notna(row.get('model_seeds')):
                    #    model_seeds = "1"  # default seed

                    rows.append({
                        'job_name': specific_job_id,
                        'type': row['type'],
                        'id': entity_id,
                        'sequence': sequence,
                        'modifications': modifications,
                        'ccd_codes': ccd_codes,
                        'smiles': smiles,
                        'msa_option': msa_option,
                        'unpaired_msa': unpaired_msa,
                        'paired_msa': paired_msa,
                        'templates': templates,
                        'model_seeds': model_seeds,
                        'bonded_atom_pairs': bonded_atom_pairs,
                        'user_ccd': user_ccd
                    })

    df_af3 = pd.DataFrame(rows)
    df_summary = pd.DataFrame(summary_data)

    # Sort columns for the summary CSV
    count_cols = [c for c in df_summary.columns if c.startswith('monomer_') and not c.endswith('_prefix')]
    prefix_cols = [c for c in df_summary.columns if c.endswith('_prefix')]
    df_summary = df_summary[['job_name', 'parent_job'] + count_cols + prefix_cols].fillna(0)

    return df_af3, df_summary

def create_all_vs_all_df(jobs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create combined jobs with letter IDs unique within each combined job.

    Parameters:
    - jobs_df: original jobs DataFrame

    Returns:
    - combined_jobs_df: DataFrame with new combined jobs and IDs added to the original jobs DataFrame
    """

    unique_jobs = jobs_df['job_name'].unique()
    pairs = list(itertools.combinations_with_replacement(unique_jobs, 2))
    pairing_df = pd.DataFrame(pairs, columns=['job_name_1', 'job_name_2'])
    letters = list(string.ascii_uppercase)
    letters = letters + [''.join(pair) for pair in itertools.product(letters, repeat=2)]

    pairs_df = pairing_df.copy()
    pairs_df['combined_job_name'] = pairs_df['job_name_1'] + '_' + pairs_df['job_name_2']

    combined_list = []

    for _, row in pairs_df.iterrows():
        job1 = row['job_name_1']
        job2 = row['job_name_2']
        combined_name = row['combined_job_name']

        df1 = jobs_df[jobs_df['job_name'] == job1].copy()
        df2 = jobs_df[jobs_df['job_name'] == job2].copy()

        # Tag sides for tracking
        df1['side'] = 'left'
        df2['side'] = 'right'

        # Keep original info
        df1['original_job_name'] = df1['job_name']
        df1['original_id'] = df1['id']
        df2['original_job_name'] = df2['job_name']
        df2['original_id'] = df2['id']

        # Assign combined job name
        df1['job_name'] = combined_name
        df2['job_name'] = combined_name

        # Concatenate both sides
        combined_list.append(pd.concat([df1, df2], ignore_index=True))

    combined_jobs_df = pd.concat(combined_list, ignore_index=True)

    # Assign new letter IDs unique within each combined job
    combined_jobs_df['id'] = combined_jobs_df.groupby('job_name').cumcount().map(lambda x: letters[x])
    # Create mapping
    return combined_jobs_df


def create_pulldown_df(jobs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create combined jobs only for pairs of jobs belonging to different groups.
    Letter IDs are unique within each combined job.
    """

    # Group jobs by group
    group_to_jobs = jobs_df.groupby("bait_or_target")["job_name"].unique()

    # Generate cross-group job pairs efficiently
    pairs = [
        (job1, job2)
        for (_, jobs1), (_, jobs2)
        in itertools.combinations(group_to_jobs.items(), 2)
        for job1, job2 in itertools.product(jobs1, jobs2)
    ]

    pairing_df = pd.DataFrame(pairs, columns=["job_name_1", "job_name_2"])

    # Letter pool
    letters = list(string.ascii_uppercase)
    letters += [''.join(p) for p in itertools.product(letters, repeat=2)]

    pairing_df["combined_job_name"] = (
            pairing_df["job_name_1"] + "_" + pairing_df["job_name_2"]
    )

    combined_list = []

    for _, row in pairing_df.iterrows():
        job1 = row["job_name_1"]
        job2 = row["job_name_2"]
        combined_name = row["combined_job_name"]

        df1 = jobs_df[jobs_df["job_name"] == job1].copy()
        df2 = jobs_df[jobs_df["job_name"] == job2].copy()

        df1["side"] = "left"
        df2["side"] = "right"

        df1["original_job_name"] = df1["job_name"]
        df1["original_id"] = df1["id"]
        df2["original_job_name"] = df2["job_name"]
        df2["original_id"] = df2["id"]

        df1["job_name"] = combined_name
        df2["job_name"] = combined_name

        combined_list.append(pd.concat([df1, df2], ignore_index=True))

    combined_jobs_df = pd.concat(combined_list, ignore_index=True)

    # Assign new letter IDs unique within each combined job
    combined_jobs_df["id"] = (
        combined_jobs_df.groupby("job_name")
        .cumcount()
        .map(lambda x: letters[x])
    )

    return combined_jobs_df


def write_fold_inputs(
        df: pd.DataFrame,
        output_dir: Union[str, Path],
        mode: str = "custom",
        n_seeds: Optional[int] = None,
) -> None:
    grouped = df.groupby('job_name')
    for job_name, group in grouped:
        entities = []
        model_seeds = None
        bonded_atom_pairs = None
        user_ccd = None

        for _, row in group.iterrows():
            entity_type = row['type']
            entity_id = row['id']
            sequence = row.get('sequence', '')

            # Parse optional fields
            modifications = parse_json_field(row.get('modifications'))
            msa_option = row.get('msa_option', 'auto')
            unpaired_msa = row.get('unpaired_msa')
            paired_msa = row.get('paired_msa')
            templates = parse_json_field(row.get('templates'))

            # Create sequence data based on entity type
            if entity_type == 'protein':
                sequence_data = create_protein_sequence_data(
                    sequence=sequence,
                    modifications=modifications,
                    msa_option=msa_option,
                    unpaired_msa=unpaired_msa,
                    paired_msa=paired_msa,
                    templates=templates
                )
            elif entity_type == 'rna':
                sequence_data = create_rna_sequence_data(
                    sequence=sequence,
                    modifications=modifications,
                    msa_option=msa_option,
                    unpaired_msa=unpaired_msa
                )
            elif entity_type == 'dna':
                sequence_data = create_dna_sequence_data(
                    sequence=sequence,
                    modifications=modifications
                )
            elif entity_type == 'ligand':
                ccd_codes = parse_list_field(row.get('ccd_codes'))
                smiles = row.get('smiles')
                sequence_data = create_ligand_sequence_data(
                    ccd_codes=ccd_codes,
                    smiles=smiles
                )
            else:
                logger.error(f"Unknown entity type: {entity_type}")
                continue

            entities.append({
                'type': entity_type,
                'id': entity_id,
                'sequence_data': sequence_data
            })

            # Job-level parameters (assumed consistent within group)
            if model_seeds is None and pd.notna(row.get('model_seeds')):
                model_seeds = parse_list_field(row.get('model_seeds'), data_type=int)
            if bonded_atom_pairs is None and pd.notna(row.get('bonded_atom_pairs')):
                bonded_atom_pairs = parse_json_field(row.get('bonded_atom_pairs'))
            if user_ccd is None and pd.notna(row.get('user_ccd')):
                user_ccd = row.get('user_ccd')

        if model_seeds is None:
            model_seeds = [1]  # default seed
        if n_seeds is not None:
            model_seeds = list(range(1, n_seeds + 1))

        task = create_batch_task(
            job_name=job_name,
            entities=entities,
            model_seeds=model_seeds,
            bonded_atom_pairs=bonded_atom_pairs,
            user_ccd=user_ccd
        )

        # Determine monomer vs multimer
        group_types = group.groupby("type").size()
        n_polymers = group_types.drop(labels=["ligand", "dna"], errors="ignore").sum()
        order = "multimers" if n_polymers > 1 else "monomers"
        if order == "monomers" and mode == "all-vs-all":
            continue

        output_dir_ = os.path.join(output_dir + "/rule_PREPROCESSING", order)
        os.makedirs(output_dir_, exist_ok=True)

        if order == "multimers":
            original_name = task["name"]  # Save original once
            for s in model_seeds:
                bname = job_name + f"_seed-{s}.json"
                fold_input = os.path.join(output_dir_, bname)
                task["modelSeeds"] = [s]
                task["name"] = original_name + f"_seed-{s}"  # Set from original, not append
                with open(fold_input, "w") as f:
                    json.dump(task, f, indent=4)

        else:
            fold_input = os.path.join(output_dir_, f"{job_name}.json")
            with open(fold_input, "w") as f:
                json.dump(task, f, indent=4)


def extract_multimer_jobs(
        df: pd.DataFrame,
        output_dir: Union[str, Path],
) -> pd.DataFrame:
    """
    Return rows belonging to multimeric jobs only.
    A multimer is defined as having >1 polymer (protein/RNA/DNA).
    """

    def is_multimer(group: pd.DataFrame) -> bool:
        group_types = group.groupby("type").size()
        n_polymers = group_types.drop(labels=["ligand", "dna"], errors="ignore").sum()
        return n_polymers > 1

    df["fold_input"] = output_dir + "/rule_PREPROCESSING/multimers/" + df["job_name"] + ".json"
    # Assuming model_seeds is a comma-separated string like "1,2,3,4,5"

    # Split model_seeds and expand into separate rows TODO check if this respects the individual seeds specs
    df["model_seeds"] = df["model_seeds"].str.split(",")  # or str.split() for space-separated
    df = df.explode("model_seeds").reset_index(drop=True)

    # Optional: convert seeds to integers and clean whitespace
    df["model_seeds"] = df["model_seeds"].str.strip()  # Remove whitespace
    df["seed"] = pd.to_numeric(df["model_seeds"], errors="coerce")  # Convert to int
    df["job_name"] = df["job_name"] + "_seed-" + df["seed"].astype(str)
    df["fold_input"] = df["fold_input"].apply(lambda x: x.replace(".json",""))
    df["fold_input"] = df["fold_input"]+ "_seed-" + df["seed"].astype(str) + ".json"
    return (
        df.groupby("job_name", group_keys=False)
        .filter(is_multimer)
        .copy()
    )


def extract_monomer_jobs(
        multimer_df: pd.DataFrame,
        output_dir: Union[str, Path],
        has_multimers: bool = False,
) -> pd.DataFrame:
    """
    From multimer jobs, create monomer jobs:
    - One polymer per job
    - job_name becomes <multimer>.<chain_id>
    """
    if multimer_df.empty:
        return pd.DataFrame()

    monomers = (
        multimer_df
        .loc[~multimer_df["type"].isin(["ligand", "dna"])]
        .copy()
    )

    monomers["original_job_name"] = monomers["job_name"]
    monomers["original_id"] = monomers["id"]

    monomers["job_name"] = (
        monomers["job_name"] + "_chain-" + monomers["id"]
        if has_multimers
        else monomers["job_name"]
    )
    monomers["job_name"] = monomers["job_name"].apply(lambda x: sanitised_name(x))
    monomers["fold_input"] = os.path.dirname(output_dir + "/rule_PREPROCESSING") + "/rule_AF3_DATA_PIPELINE/" + \
                             monomers["job_name"] + "_data.json"

    return monomers.reset_index(drop=True)


def create_batch_task(
        job_name: str,
        entities: Sequence[Mapping[str, Any]],
        model_seeds: Sequence[int],
        bonded_atom_pairs: Optional[Sequence[Sequence[int]]] = None,
        user_ccd: Optional[str] = None,
) -> dict[str, Any]:
    """,put_dir)True
    Creates a batch task dictionary for a single prediction.

    :param job_name: Name of the job.
    :type job_name: str
    :param entities: List of dictionaries, each representing an Entity.
        Each entity dict should have keys:
        - 'type': 'protein', 'rna', 'dna', or 'ligand'
        - 'id': str or list
        - 'sequence_data': dict with sequence information
    :type entities: list
    :param model_seeds: List of integers.
    :type model_seeds: list of int
    :param bonded_atom_pairs: Optional list of bonded atom pairs.
    :type bonded_atom_pairs: list, optional
    :param user_ccd: Optional user CCD.
    :type user_ccd: str, optional
    :return: Dictionary representing the AlphaFold input JSON structure.
    :rtype: dict
    """
    sequences = []
    for entity in entities:
        entity_type = entity['type']
        sequence_data = entity['sequence_data']
        entity_id = entity['id']

        sequence_entry = sequence_data.copy()
        sequence_entry['id'] = entity_id

        if entity_type == 'protein':
            sequences.append({'protein': sequence_entry})
        elif entity_type == 'rna':
            sequences.append({'rna': sequence_entry})
        elif entity_type == 'dna':
            sequences.append({'dna': sequence_entry})
        elif entity_type == 'ligand':
            sequences.append({'ligand': sequence_entry})
        else:
            logger.error(f"Unknown entity type: {entity_type}")
            continue

    alphafold_input = {
        "name": job_name,
        "modelSeeds": model_seeds,
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 1
    }

    if bonded_atom_pairs:
        alphafold_input["bondedAtomPairs"] = bonded_atom_pairs

    if user_ccd:
        alphafold_input["userCCD"] = user_ccd

    logger.info(f"Created task for job: {job_name}")

    return alphafold_input


def create_rna_sequence_data(
        sequence: str,
        modifications: Optional[Sequence[Mapping[str, Any]]] = None,
        msa_option: Literal["auto", "none", "custom"] = "auto",
        unpaired_msa: Optional[str] = None,
) -> dict[str, Any]:
    """
    Creates sequence data for an RNA entity.

    :param sequence: The RNA sequence.
    :type sequence: str
    :param modifications: Optional list of modifications.
    :type modifications: list of dict, optional
    :param msa_option: MSA option, 'auto', 'upload', or 'none'.
    :type msa_option: str
    :param unpaired_msa: Unpaired MSA (if msa_option is 'upload').
    :type unpaired_msa: str, optional
    :return: Sequence data dictionary.
    :rtype: dict
    """
    rna_entry = {
        "sequence": sequence
    }
    if modifications:
        rna_entry["modifications"] = modifications
    if msa_option == 'auto':
        rna_entry["unpairedMsa"] = None
    elif msa_option == 'none':
        rna_entry["unpairedMsa"] = ""
    elif msa_option == 'upload':
        rna_entry["unpairedMsa"] = unpaired_msa or ""
    else:
        logger.error(f"Invalid msa_option: {msa_option}")
    return rna_entry


# modify the following function to be compatible with the following documentation:
def create_protein_sequence_data(
        sequence: str,
        modifications: Optional[Sequence[Mapping[str, Any]]] = None,
        msa_option: Literal["auto", "none", "custom"] = "auto",
        unpaired_msa: Optional[str] = None,
        paired_msa: Optional[str] = None,
        templates: Optional[str] = None,
) -> dict[str, Any]:
    """
    Creates sequence data for a protein entity.

    :param sequence: The protein sequence.
    :type sequence: str
    :param modifications: Optional list of modifications, each with keys 'ptmType' and 'ptmPosition'.
    :type modifications: list of dict, optional
    :param msa_option: MSA option, 'auto', 'upload', or 'none'.
    :type msa_option: str
    :param unpaired_msa: Unpaired MSA (if msa_option is 'upload').
    :type unpaired_msa: str, optional
    :param paired_msa: Paired MSA (if msa_option is 'upload').
    :type paired_msa: str, optional
    :param templates: Optional list of template dicts. Use [] for template-free, None/null for auto template search.
    :type templates: list of dict, optional
    :return: Sequence data dictionary.
    :rtype: dict
    """
    protein_entry = {
        "sequence": sequence
    }

    if modifications:
        protein_entry["modifications"] = modifications

    if msa_option == 'auto':
        # Both MSAs unset (null) - AlphaFold 3 will build both MSAs automatically
        # Don't set unpairedMsa and pairedMsa fields at all (implicitly null)
        # Templates can be:
        # - Unset (null/omitted) for auto template search
        # - [] for template-free with auto MSA
        if templates is not None:
            protein_entry["templates"] = templates

    elif msa_option == 'none':
        # Both MSAs set to empty string - completely MSA-free
        protein_entry["unpairedMsa"] = ""
        protein_entry["pairedMsa"] = ""
        # Templates defaults to [] if not provided (template-free)
        protein_entry["templates"] = templates if templates is not None else []

    elif msa_option == 'upload':
        # Custom MSA provided
        # Both unpairedMsa and pairedMsa must be set (non-null)
        # Typically: unpairedMsa = custom A3M, pairedMsa = ""
        protein_entry["unpairedMsaPath"] = unpaired_msa if unpaired_msa is not None else ""
        protein_entry["pairedMsaPath"] = paired_msa if paired_msa is not None else ""
        # Templates can be:
        # - Unset (null) to let AF3 search for templates using the provided MSA
        # - [] for template-free with custom MSA
        # - List of template dicts for custom templates
        if templates is not None:
            protein_entry["templates"] = templates
    else:
        logger.error(f"Invalid msa_option: {msa_option}")

    return protein_entry


def create_dna_sequence_data(sequence, modifications=None):
    """
    Creates sequence data for a DNA entity.

    :param sequence: The DNA sequence.
    :type sequence: str
    :param modifications: Optional list of modifications.
    :type modifications: list of dict, optional
    :return: Sequence data dictionary.
    :rtype: dict
    """
    dna_entry = {
        "sequence": sequence
    }
    if modifications:
        dna_entry["modifications"] = modifications
    return dna_entry


def create_ligand_sequence_data(ccd_codes=None, smiles=None):
    """
    Creates sequence data for a ligand entity.

    :param ccd_codes: List of CCD codes.
    :type ccd_codes: list of str, optional
    :param smiles: SMILES string.
    :type smiles: str, optional
    :return: Sequence data dictionary.
    :rtype: dict
    """
    # if ccd_codes and smiles:
    #    logger.error("Please provide only one of CCD Codes or SMILES String.")
    #    return {}
    if ccd_codes:
        ligand_entry = {
            "ccdCodes": ccd_codes
        }
        return ligand_entry
    elif smiles:
        ligand_entry = {
            "smiles": smiles
        }
        return ligand_entry
    else:
        logger.error("Ligand requires either CCD Codes or SMILES String.")
        return {}


def parse_json_field(value):
    """
    Parses a JSON string field into a Python object.

    Returns None if the value is NaN or empty.

    :param value: JSON string to parse.
    :type value: str
    :return: Parsed Python object or None.
    :rtype: object or None
    """
    if pd.isna(value) or value == '':
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return None


def parse_list_field(value, data_type=str):
    """
    Parses a comma-separated string field into a list.

    Returns None if the value is NaN or empty.

    :param value: Comma-separated string to parse.
    :type value: str
    :param data_type: Data type to convert list items to.
    :type data_type: type
    :return: List of items converted to data_type, or None.
    :rtype: list or None
    """
    if pd.isna(value) or value == '':
        return None
    return [data_type(item.strip()) for item in value.split(',') if item.strip()]


def remove_duplicate_jobs_scalable(df, cols_to_compare, log_file=f'duplicate_jobs_summary.json'):
    """
    Scalable approach for thousands of duplicates.
    Logs summary statistics instead of all duplicate pairs.

    :param df: Input dataframe
    :param cols_to_compare: Columns to use for comparison
    :param log_file: Path to write summary (JSON format)
    :return: Deduplicated dataframe
    """

    # Create signatures for each job
    job_sigs = df.groupby('job_name').apply(
        lambda x: hash(x[cols_to_compare]
                       .sort_values(by=['type', 'id', 'sequence'])
                       .reset_index(drop=True)
                       .to_json(orient='records'))
    )

    # Group jobs by signature (O(n) memory)
    sig_to_jobs = defaultdict(list)
    for job_name, signature in job_sigs.items():
        sig_to_jobs[signature].append(job_name)

    # Calculate statistics
    duplicate_groups = {sig: jobs for sig, jobs in sig_to_jobs.items() if len(jobs) > 1}
    total_jobs = len(job_sigs)
    unique_jobs = len(sig_to_jobs)
    total_duplicates = total_jobs - unique_jobs

    # Build summary report (compact)
    summary = {
        'total_jobs': total_jobs,
        'unique_jobs': unique_jobs,
        'duplicate_jobs': total_duplicates,
        'duplicate_groups': len(duplicate_groups),
        'group_size_distribution': {},
        'largest_groups': [],
        'sample_duplicates': {}
    }

    if duplicate_groups:
        # Group size distribution
        group_sizes = [len(jobs) for jobs in duplicate_groups.values()]
        size_counts = pd.Series(group_sizes).value_counts().sort_index()
        summary['group_size_distribution'] = size_counts.to_dict()

        # Top 10 largest groups (most duplicates)
        largest = sorted(duplicate_groups.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        summary['largest_groups'] = [
            {
                'kept': jobs[0],
                'num_duplicates': len(jobs) - 1,
                'sample_duplicates': jobs[1:6]  # Show max 5 examples
            }
            for sig, jobs in largest
        ]

        # Sample of duplicate pairs (first 10 groups)
        for i, (sig, jobs) in enumerate(list(duplicate_groups.items())[:10], 1):
            summary['sample_duplicates'][f'group_{i}'] = {
                'kept': jobs[0],
                'removed': jobs[1:]
            }

        # Print summary
        logger.debug(f"⚠️  WARNING: Found duplicates")
        logger.debug(f"   Total jobs: {total_jobs}")
        logger.debug(f"   Unique jobs: {unique_jobs}")
        logger.debug(f"   Duplicates removed: {total_duplicates}")
        logger.debug(f"   Duplicate groups: {len(duplicate_groups)}")
        logger.debug(f"   Group size distribution:")
        for size, count in size_counts.items():
            logger.debug(f"      {count} groups with {size} duplicates each")

        logger.debug(f"   Top 5 largest duplicate groups:")
        for i, item in enumerate(summary['largest_groups'][:5], 1):
            logger.debug(f"      {i}. '{item['kept']}' has {item['num_duplicates']} duplicates")

        # Save summary to JSON
        with open(log_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.debug(f"✓ Summary saved to {log_file}")

        # Optional: Save full duplicate mapping to compressed file if needed
        if total_duplicates > 100:
            mapping_file = log_file.replace('.json', '_full_mapping.txt.gz')
            logger.debug(f"   Saving full duplicate mapping to {mapping_file} (compressed)")

            import gzip
            with gzip.open(mapping_file, 'wt') as f:
                f.write("kept_job\tremoved_jobs\n")
                for jobs in duplicate_groups.values():
                    f.write(f"{jobs[0]}\t{','.join(jobs[1:])}\n")
    else:
        logger.debug("✓ No duplicate jobs found")
        with open(log_file, 'w') as f:
            json.dump(summary, f, indent=2)

    # Return deduplicated dataframe
    unique_job_names = [jobs[0] for jobs in sig_to_jobs.values()]
    return df[df['job_name'].isin(unique_job_names)]


@click.command()
@click.argument('sample_sheet', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--mode', type=str, default='custom',
              help="Choose run mode: 'custom', 'all-vs-all', 'pulldown','virtual-drug-screen', or 'stoichio-screen'")
@click.option('--predict-individual-components', is_flag=True,
              help="The individual components of multimeric samples will also be predicted.")
@click.option('--n-seeds', type=int, default=None,
              help="Number of random seeds. Useful for massive sampling. If specified, the model_seeds column in the sample sheet is ignored")
@click.option('--n-samples', type=int, default=5,
              help="Number of models per seed. Useful for massive sampling")
def main(sample_sheet, output_dir, mode, predict_individual_components, n_seeds, n_samples):
    """
    Creates batch tasks from a DataFrame.

    :param sample_sheet: path to a tab separated table with columns representing parameters:
        - 'job_name': str
        - 'type': 'protein', 'rna', 'dna', or 'ligand'
        - 'id': str or list
        - 'sequence': str
        - Other optional parameters:
            - 'modifications': list of dicts (as JSON string)
            - 'msa_option': 'auto', 'upload', or 'none'
            - 'unpaired_msa': str
            - 'paired_msa': str
            - 'templates': list of dicts (as JSON string)
            - 'templates': list of dicts (as JSON string)
            - 'smiles':	str
            - 'ccd_codes':str or list of strings separated by comma
            - 'model_seeds': list of integers (as string)
            - 'bonded_atom_pairs': list (as JSON string)
            - 'user_ccd': str
    :type sample_sheet: path
    :return: None
    """
    logger.info(f"SAMPLES = {sample_sheet}")
    logger.info(f"OUTPUT_DIR = {output_dir}")
    logger.info(f"MODE = {mode}")
    logger.info(f"PREDICT_INDIVIDUAL_COMPONENTS = {predict_individual_components}")
    logger.info(f"#SEEDS = {predict_individual_components}")
    metadata_dir = os.path.join(f"{output_dir}","rule_PREPROCESSING","metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    df = pd.read_csv(sample_sheet, sep="\t")
    df["job_name"] = df["job_name"].apply(lambda x: sanitised_name(x))

    cols_to_compare = df.columns.difference(['job_name'])

    if mode != 'virtual-drug-screen' and mode != 'stoichio-screen':

        df_dedup = remove_duplicate_jobs_scalable(df, cols_to_compare,log_file=os.path.join(metadata_dir,"duplicate_job_summary.json"))
        has_multimers_ = has_multimers(df_dedup)
    if mode == "custom":
        # write originals
        write_fold_inputs(df_dedup, output_dir, n_seeds=n_seeds)

        # derive + write monomers from multimers
        multimer_df = extract_multimer_jobs(df_dedup, output_dir)

        if has_multimers_:
            monomer_df = extract_monomer_jobs(multimer_df, output_dir, has_multimers=True)
        else:
            monomer_df = extract_monomer_jobs(df_dedup, output_dir, has_multimers=False)
        # write_fold_inputs(monomer_df, output_dir,n_seeds=n_seeds)

    elif mode == "all-vs-all":
        write_fold_inputs(df_dedup, output_dir, n_seeds=n_seeds)

        combined_df = create_all_vs_all_df(df_dedup)
        write_fold_inputs(combined_df, output_dir, n_seeds=n_seeds)

        multimer_df = extract_multimer_jobs(combined_df, output_dir)
        monomer_df = extract_monomer_jobs(multimer_df, output_dir, has_multimers=True)

    elif mode == "pulldown":
        write_fold_inputs(df_dedup, output_dir, n_seeds=n_seeds)

        combined_df = create_pulldown_df(df_dedup)
        write_fold_inputs(combined_df, output_dir, n_seeds=n_seeds)

        multimer_df = extract_multimer_jobs(combined_df, output_dir)
        monomer_df = extract_monomer_jobs(multimer_df, output_dir, has_multimers=True)

    elif mode == "virtual-drug-screen":
        df = transform_vds_to_af3(df)
        cols_to_compare = df.columns.difference(['job_name'])

        df_dedup = remove_duplicate_jobs_scalable(df, cols_to_compare,log_file=os.path.join(metadata_dir,"duplicate_job_summary.json"))
        has_multimers_ = has_multimers(df_dedup)

        write_fold_inputs(df_dedup, output_dir, n_seeds=n_seeds)

        combined_df = df_dedup
        write_fold_inputs(combined_df, output_dir, n_seeds=n_seeds)

        multimer_df = extract_multimer_jobs(combined_df, output_dir)
        monomer_df = extract_monomer_jobs(multimer_df, output_dir, has_multimers=True)

    elif mode == "stoichio-screen":
        df, summary = transform_stoichio_screen_to_af3(df,n_seeds=n_seeds)
        summary.to_csv(os.path.join(output_dir,"rule_PREPROCESSING","metadata","stoichio_screen.csv"), index=False)
        cols_to_compare = df.columns.difference(['job_name'])

        df_dedup = remove_duplicate_jobs_scalable(df, cols_to_compare,log_file=os.path.join(metadata_dir,"duplicate_job_summary.json"))
        has_multimers_ = has_multimers(df_dedup)

        write_fold_inputs(df_dedup, output_dir, n_seeds=n_seeds)

        multimer_df = extract_multimer_jobs(df_dedup, output_dir)
        monomer_df = extract_monomer_jobs(multimer_df, output_dir, has_multimers=True)


    if has_multimers_:
        write_fold_inputs(monomer_df, output_dir, n_seeds=n_seeds)


        multimer_to_monomer_df = pd.merge(
            multimer_df[["job_name", "id", "fold_input", "model_seeds"]],
            monomer_df[
                ["job_name", "id", "sequence", "fold_input",
                 "original_job_name", "original_id"]
            ],
            left_on="job_name",
            right_on="original_job_name",
            how="right",
        )

        multimer_to_monomer_df["fold_input_mono"] = (
                output_dir
                + "/rule_AF3_DATA_PIPELINE/"
                + multimer_to_monomer_df.job_name_y
                + "_data.json"
        )

        replicate_monomer_groups = (
            multimer_to_monomer_df
            .groupby("sequence")["fold_input_mono"]
            .apply(set)
            .tolist()
        )

        canonical_monomer_map = {
            m: canonical
            for group in replicate_monomer_groups
            for canonical in [sorted(group)[0]]
            for m in group
        }


        #multimer_to_monomer_df["fold_input_x"] = multimer_to_monomer_df["fold_input_x"].apply(lambda x: x.replace(".json",""))
        #multimer_to_monomer_df["fold_input_x"] = multimer_to_monomer_df["fold_input_x"]+"_seed-"+multimer_to_monomer_df["model_seeds"]+".json"

        inference_to_data_pipeline_map = (
            multimer_to_monomer_df
            .groupby("fold_input_x", sort=False)[["id_y", "fold_input_mono"]]
            .apply(lambda g: {
                chain_id: canonical_monomer_map[file]
                for chain_id, file in zip(g["id_y"], g["fold_input_mono"])
            })
            .to_dict()
        )
    else:
        inference_to_data_pipeline_map = {
            row.fold_input: {row.id: row.fold_input}
            for _, row in monomer_df.iterrows()
        }

    referenced_monomer_files = {
        monomer_file
        for mapping in inference_to_data_pipeline_map.values()
        for monomer_file in mapping.values()
    }

    for monomer_path in os.listdir(f"{output_dir}/rule_PREPROCESSING/monomers"):
        if (monomer_path.endswith(".json") and
                os.path.join(f"{output_dir}/rule_AF3_DATA_PIPELINE",
                             os.path.basename(monomer_path).replace(".json",
                                                                    "_data.json")) not in referenced_monomer_files):
            logger.info(f"Deleting redundant fold input: {monomer_path}")
            Path(os.path.join(f"{output_dir}/rule_PREPROCESSING/monomers", monomer_path)).unlink()

    inference_to_data_pipeline_df = pd.DataFrame.from_dict(inference_to_data_pipeline_map, orient="index")
    inference_to_data_pipeline_df = inference_to_data_pipeline_df.reset_index()
    inference_to_data_pipeline_df = inference_to_data_pipeline_df.rename(
        columns={inference_to_data_pipeline_df.columns[0]: "multimer_file"}).reset_index(drop=True)

    long_inference_to_data_pipeline_df = inference_to_data_pipeline_df.melt(
        id_vars="multimer_file",
        var_name="monomer_chain_id",
        value_name="monomer_file"
    )

    #   Three sample sheets are generated:
    #   1. samples for data pipeline (i.e. everything in {output_dir}/rule_PREPROCESSING/monomers)
    #   2. samples for merging (Done: {output_dir}/rule_PREPROCESSING/inference_to_data_pipeline_map.tsv)
    #   3. samples for inference (i.e. everything in {output_dir}/rule_MERGE_MONOMERS_TO_MULTIMER)

    data_pipeline_df = pd.DataFrame(
        set([v.replace("rule_AF3_DATA_PIPELINE", "rule_PREPROCESSING/monomers").replace("_data.json", ".json") for d in
             inference_to_data_pipeline_map.values() for v in d.values()]),
        columns=["file"])
    data_pipeline_df["sample_id"] = data_pipeline_df["file"].apply(
        lambda x: Path(x).stem)
    data_pipeline_df["expected_output"] = data_pipeline_df["file"].apply(
        lambda x: x.replace("rule_PREPROCESSING/monomers", f"rule_AF3_DATA_PIPELINE/{os.path.basename(x).split(".json")[0]}"))
    data_pipeline_df["expected_output"] = data_pipeline_df["expected_output"].apply(
        lambda x: x.replace(".json", "_data.json"))

    # tmp/rule_PREPROCESSING/monomers/job1_job1_chain-B.json

    inference_df = pd.DataFrame(set([k.replace(
        "rule_PREPROCESSING/multimers" if has_multimers_ else "rule_AF3_DATA_PIPELINE",
        "rule_MERGE_MONOMERS_TO_MULTIMERS").replace(".json", "_data.json") for k in
                                     inference_to_data_pipeline_map.keys()]), columns=["inference_samples"])
    long_inference_to_data_pipeline_df = long_inference_to_data_pipeline_df.dropna(subset=["monomer_file"]).sort_values(
        ["multimer_file", "monomer_chain_id"])

    if predict_individual_components:
        if has_multimers_:
            inference_df = pd.DataFrame(np.vstack([inference_df.values,
                                                   pd.DataFrame(data_pipeline_df.file.str.replace(
                                                       "rule_PREPROCESSING/monomers",
                                                       "rule_MERGE_MONOMERS_TO_MULTIMERS").values)])
                                        , columns=inference_df.columns)
        if mode == "all-vs-all":
            long_inference_to_data_pipeline_df = pd.DataFrame(np.concatenate([long_inference_to_data_pipeline_df.values,
                                                                              long_inference_to_data_pipeline_df[
                                                                                  ["monomer_file", "monomer_chain_id",
                                                                                   "monomer_file"]].values]
                                                                             , axis=0),
                                                              columns=["multimer_file", "monomer_chain_id",
                                                                       "monomer_file"])
            long_inference_to_data_pipeline_df.loc[
                long_inference_to_data_pipeline_df.multimer_file == long_inference_to_data_pipeline_df.monomer_file, "monomer_chain_id"] = \
                long_inference_to_data_pipeline_df.monomer_file.str.split(".").str[-2].str.split("_").str[0]
    if n_seeds is None:

        fold_input_to_model_seeds_map = dict(
            zip(multimer_to_monomer_df[["fold_input_x", "model_seeds"]].drop_duplicates().fold_input_x,
                multimer_to_monomer_df[
                    ["fold_input_x", "model_seeds"]].drop_duplicates().model_seeds))

        fold_input_to_model_seeds_map = {**fold_input_to_model_seeds_map, **dict(
            zip(multimer_to_monomer_df[["fold_input_mono", "model_seeds"]].drop_duplicates().fold_input_mono,
                multimer_to_monomer_df[["fold_input_mono", "model_seeds"]].drop_duplicates().model_seeds))}

        fold_input_to_model_seeds_map = {k.replace("_data.json" if "_data.json" in k else ".json", ".json" if "_data.json" in k else "_data.json").replace("rule_PREPROCESSING/multimers",
                                                                                  "rule_MERGE_MONOMERS_TO_MULTIMERS").replace(
            "rule_PREPROCESSING/multimers", "rule_MERGE_MONOMERS_TO_MULTIMERS").replace("rule_AF3_DATA_PIPELINE",
                                                                                        "rule_MERGE_MONOMERS_TO_MULTIMERS"):v

        for k, v in fold_input_to_model_seeds_map.items()}

    inference_df = pd.concat([inference_df,
                              inference_df["inference_samples"].map(
                                  fold_input_to_model_seeds_map) if n_seeds is None else pd.Series(n_seeds,
                                                                                                   index=inference_df.index),
                              pd.Series(n_samples, index=inference_df.index)]
                             , axis=1)

    inference_df.columns = ["inference_samples", "n_seeds", "n_samples"]
    inference_df = expand_df(inference_df, "inference_samples")
#    inference_df["inference_samples"] = inference_df.apply(
#        lambda row: row["inference_samples"].replace("_data.json", f"_seed-{row['seed']}_data.json"),
#        # TODO: Act differently
#        axis=1
#    )
    inference_df["expected_output"] = inference_df["inference_samples"].apply(
        lambda x: x.replace("rule_MERGE_MONOMERS_TO_MULTIMERS",
                            "rule_AF3_INFERENCE"))
    inference_df["expected_output"] = inference_df.apply(
        lambda row: str(row["expected_output"]).replace(
            "_data.json", f"/seed-{Path(row["inference_samples"]).stem.split("_data")[0].split("_seed-")[-1]}_sample-{row['sample']}/model.cif"
        ),
        axis=1
    )

    pattern = r".*_seed-[0-9]+_chain-[a-z]{1,2}\.json$"  # TODO: when predict-individual-components is specified the number of seeds should also be included
    inference_df.loc[inference_df["inference_samples"].str.match(pattern), "inference_samples"] = inference_df[
        inference_df["inference_samples"].str.match(pattern)].apply(
        lambda row: row["inference_samples"].replace(".json", f"_data.json"),
        # TODO: Act differently
        axis=1
    )
    inference_df.loc[inference_df["expected_output"].str.match(pattern), "expected_output"] = inference_df.loc[
        inference_df["expected_output"].str.match(pattern)].apply(
        lambda row: str(row["expected_output"]).replace(
            ".json", f"/seed-{row['seed']}_sample-{row['sample']}/model.cif"
        ),
        axis=1
    )

    inference_df["job_name"] = inference_df["inference_samples"].apply(
        lambda x: os.path.basename(x).split("_data.json")[0])
    inference_df["job_name"] = inference_df["job_name"].apply(lambda x: os.path.basename(x).split("_data", 1)[0])
    inference_df["job_name"] = inference_df["job_name"].apply(lambda x: Path(x).stem)


    long_inference_to_data_pipeline_df["monomer_file"] = long_inference_to_data_pipeline_df["monomer_file"].apply(lambda x: os.path.join(os.path.dirname(x), os.path.basename(x).split("_data.json")[0], os.path.basename(x)))
    long_inference_to_data_pipeline_df["sample_id"] = long_inference_to_data_pipeline_df["multimer_file"].apply(lambda x: Path(x).stem)

    long_inference_to_data_pipeline_df.to_csv(f"{metadata_dir}/inference_to_data_pipeline_map.tsv",
                                              sep="\t", index=False)

    data_pipeline_df.to_csv(f"{metadata_dir}/data_pipeline_samples.tsv", sep="\t", index=False)

    inference_df.sort_values(["job_name", "seed", "sample"])[
        ["job_name", "inference_samples", "expected_output"]].rename(columns={"job_name":"sample_id","inference_samples":"file"}).to_csv(
        f"{metadata_dir}/inference_samples.tsv", sep="\t", index=False)

    logger.info(f"Rule PREPROCESSING was completed successfully!")
    logger.info(
        f"Fold input files were saved to {output_dir}/rule_PREPROCESSING/monomers and {output_dir}/rule_PREPROCESSING/multimers")
    logger.info(
        f"Multimer to monomer map was saved to {metadata_dir}/inference_to_data_pipeline_map.tsv")
    logger.info(f"Data pipeline sample sheet was saved to {metadata_dir}/data_pipeline_samples.tsv")
    logger.info(f"Inference sample sheet was saved to {metadata_dir}/inference_samples.tsv")


if __name__ == "__main__":
    main()
