# afusion/api.py

import os
import json
import pandas as pd
import string
from loguru import logger
import click
import itertools
import numpy as np

def sanitised_name(name):
    """Returns sanitised version of the name that can be used as a filename."""
    lower_spaceless_name = name.lower().replace(' ', '_')
    allowed_chars = set(string.ascii_lowercase + string.digits + '_-.')
    return ''.join(l for l in lower_spaceless_name if l in allowed_chars)

def create_batch_task(job_name, entities, model_seeds, bonded_atom_pairs=None, user_ccd=None):
    """
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

    logger.debug(f"Created task for job: {job_name}")
    return alphafold_input

def create_protein_sequence_data(sequence, modifications=None, msa_option='auto_template_free', unpaired_msa=None, paired_msa=None, templates=None):
    """
    Creates sequence data for a protein entity.

    :param sequence: The protein sequence.
    :type sequence: str
    :param modifications: Optional list of modifications, each with keys 'ptmType' and 'ptmPosition'.
    :type modifications: list of dict, optional
    :param msa_option: MSA option, 'auto_template_free','auto_template_based', 'upload', or 'none'.
    :type msa_option: str
    :param unpaired_msa: Unpaired MSA (if msa_option is 'upload').
    :type unpaired_msa: str, optional
    :param paired_msa: Paired MSA (if msa_option is 'upload').
    :type paired_msa: str, optional
    :param templates: Optional list of template dicts.
    :type templates: list of dict, optional
    :return: Sequence data dictionary.
    :rtype: dict
    """
    protein_entry = {
        "sequence": sequence
    }
    if modifications:
        protein_entry["modifications"] = modifications
    if msa_option == 'auto_template_free':
        protein_entry["unpairedMsa"] = None
        protein_entry["pairedMsa"] = None
        protein_entry["templates"] = []
    elif msa_option == 'auto_template_based':
        pass
    elif msa_option == 'none':
        protein_entry["unpairedMsa"] = ""
        protein_entry["pairedMsa"] = ""
        protein_entry["templates"] = []
    elif msa_option == 'upload':
        protein_entry["unpairedMsa"] = unpaired_msa or ""
        protein_entry["pairedMsa"] = paired_msa or ""
        protein_entry["templates"] = templates or []
    else:
        logger.error(f"Invalid msa_option: {msa_option}")
    return protein_entry


def create_rna_sequence_data(sequence, modifications=None, msa_option='auto_template_free', unpaired_msa=None):
    """
    Creates sequence data for an RNA entity.

    :param sequence: The RNA sequence.
    :type sequence: str
    :param modifications: Optional list of modifications.
    :type modifications: list of dict, optional
    :param msa_option: MSA option, 'auto_template_free','auto_template_based', 'upload', or 'none'.
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
    if msa_option in ['auto_template_free','auto_template_based']:
        rna_entry["unpairedMsa"] = None
    elif msa_option == 'none':
        rna_entry["unpairedMsa"] = ""
    elif msa_option == 'upload':
        rna_entry["unpairedMsa"] = unpaired_msa or ""
    else:
        logger.error(f"Invalid msa_option: {msa_option}")
    return rna_entry


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
    if ccd_codes and smiles:
        logger.error("Please provide only one of CCD Codes or SMILES String.")
        return {}
    elif ccd_codes:
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

def check_for_empty_values(df):
    """
    Checks for empty values
    """
    if df.isna().any().any():
        raise ValueError(f"Missing values found in:\n{df.isna().stack()[lambda x: x].index.tolist()}")

def check_for_mode_validity(df,mode):
    if mode == "pulldown":
        allowed_columns = ["id","seq","bait_or_target"]
        if not set(allowed_columns).issubset(df.columns):
            raise ValueError(f"Dataframe must contain the following columns: {allowed_columns}. Got {df.columns} instead.")
        if df[~df["bait_or_target"].isin(["bait","target"])].shape[0] > 0:
            raise ValueError(f"Only 'bait' or 'target' are allowed as values for column 'bait_or_target'. Got {df[~df["bait_or_target"].isin(["bait","target"])]["bait_or_target"]} instead.")

#    if df["mode"].unique().shape[0] > 1 and df["mode"].unique()[0] not in ["stoichio-screen"]:
#        raise ValueError(f"Only one mode is allowed for all-vs-all and default, got {df['mode'].unique()} instead")

def check_for_valid_columns(df,mode):
    # Get all columns in the DataFrame
    provided_columns = set(df.columns)
    required_columns = ["id","sequence","type"]

    optional_columns = ["modifications","msa_option","unpaired_msa","paired_msa","templates","model_seeds","bonded_atom_pairs","user_ccd","smiles"]
    if mode == "virtual_drug_screen":
        optional_columns = optional_columns + ["drug_or_target"]
    if mode == "pulldown":
        optional_columns = optional_columns + ["bait_or_target"]
    # Identify unexpected columns
    unexpected_columns = provided_columns - (set(required_columns) | set(optional_columns))

    if unexpected_columns:
        raise ValueError(f"Unexpected columns found: {unexpected_columns}. See --help for more info on allowed columns")


def create_all_vs_all_df(df,msa_option=None):
    """
    Given a DataFrame with columns 'id', 'sequence', and 'type', return a new DataFrame
    that contains the original rows along with all possible pairwise combinations.

    :param df: Input DataFrame with 'id', 'sequence', and 'type' columns.
    :return: New DataFrame with original rows and all vs all combinations.
    """
    all_vs_all = []

    for (id1, seq1), (id2, seq2) in itertools.combinations_with_replacement(df[['id','sequence']].values, 2,):
        type1 = df[df["id"]==id1]["type"].values[0]
        type2 = df[df["id"]==id2]["type"].values[0]
        all_vs_all.append({"id": f"{id1}_{id2}", "sequence": f"{seq1}","type": f"{type1}"})
        all_vs_all.append({"id": f"{id1}_{id2}", "sequence": f"{seq2}","type": f"{type2}"})


    df_all_vs_all = pd.DataFrame(all_vs_all)
    df_all_vs_all_and_original = pd.concat([df, df_all_vs_all], ignore_index=True)
    df_all_vs_all_and_original = df_all_vs_all_and_original.rename(columns={"id": "job_name"})
    counts = df_all_vs_all_and_original['job_name'].value_counts()

    # Assign labels
    df_all_vs_all_and_original['id'] = np.where(
        df_all_vs_all_and_original['job_name'].map(counts) == 1,  # If the group size is 1
        'A',
        np.where(df_all_vs_all_and_original.groupby('job_name').cumcount() % 2 == 0, 'A', 'B')  # Alternate 'A' and 'B' in size-2 groups
    )

    df_all_vs_all_and_original.loc[df_all_vs_all_and_original["type"]=="ligand",'smiles'] = df_all_vs_all_and_original.loc[df_all_vs_all_and_original["type"]=="ligand",'sequence'].copy()
    df_all_vs_all_and_original.loc[df_all_vs_all_and_original["type"] == "ligand", 'sequence'] = ""
    if msa_option in ["auto_template_based","auto_template_free"]:
        df_all_vs_all_and_original["job_name"] = df_all_vs_all_and_original["job_name"] + "_" + msa_option
        return df_all_vs_all_and_original

    return df_all_vs_all_and_original


def create_stoichio_screen_df(df):
    pass


def create_pulldown_df(df):
    pass

def create_virtual_drug_screen_df(df, msa_option=None):
    """
    Simulate a virtual drug screening by pairing ligands (drugs) with protein targets.
    Also includes standalone targets.

    :param df: Input DataFrame with 'id', 'sequence', 'type', and 'drug_or_target' columns.
    :return: DataFrame with ligand-target pairs and standalone targets.
    """
    drug_df = df[df["drug_or_target"] == "drug"].copy()
    target_df = df[df["drug_or_target"] == "target"].copy()

    all_entries = []

    # Add standalone targets
    for _, (target_id, target_seq) in target_df[['id', 'sequence']].iterrows():
        all_entries.append(
            {"job_name": target_id, "id": "A", "type": "protein", "sequence": target_seq, "smiles": ""})

    # Add ligand-target pairs
    for _, (drug_id, drug_smiles) in drug_df[['id', 'sequence']].iterrows():
        for _, (target_id, target_seq) in target_df[['id', 'sequence']].iterrows():
            pair_name = f"{target_id}_{drug_id}"
            all_entries.append(
                {"job_name": pair_name, "id": "A", "type": "protein", "sequence": target_seq, "smiles": ""})
            all_entries.append(
                {"job_name": pair_name, "id": "B", "type": "ligand", "sequence": "", "smiles": drug_smiles})

    df_screen = pd.DataFrame(all_entries)

    if msa_option in ["auto_template_based", "auto_template_free"]:
        df_screen["job_name"] += "_" + msa_option

    return df_screen

def create_df_for_run_mode(df,mode,msa_option):
    """
    Creates mode specific dataframe and writes it to a file.


    :param df_path: Path to CSV table with columns representing parameters:
        - 'id': str
        - 'sequence': str
        - 'type': 'protein', 'rna', 'dna', or 'ligand'
        - 'bait_or_target': str, either 'bait' or 'target', conditionally required with 'pulldown' mode
        - Other optional parameters:
            - 'modifications': list of dicts (as JSON string)
            - 'msa_option': 'auto_template_free','auto_template_based', 'upload', or 'none'
            - 'unpaired_msa': str
            - 'paired_msa': str
            - 'templates': list of dicts (as JSON string)
            - 'model_seeds': list of integers (as string)
            - 'bonded_atom_pairs': list (as JSON string)
            - 'user_ccd': str

    :param mode: one of the following: 'all-vs-all','pulldown', 'stoichio-screen'
    :type df_path: file
    :type mode: string

    """

    check_for_valid_columns(df,mode)

    check_for_empty_values(df)

    df["id"] = df["id"].apply(lambda x: sanitised_name(x))

    if mode == "all-vs-all":
        df=create_all_vs_all_df(df,msa_option)
    if mode == "pulldown":
        df = create_pulldown_df(df)
        check_for_mode_validity(df, mode)
    if mode == "stoichio-screen":
        df = create_stoichio_screen_df(df)
    if mode == "virtual_drug_screen":
        df = create_virtual_drug_screen_df(df,msa_option)

    return df


@click.command()
@click.argument('df_path', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--mode', type=str,default='default', help="Choose run mode: 'default', 'all-vs-all', 'pulldown','virtual_drug_screen', or 'stoichio-screen'")
@click.option('--msa-option', type=str,default='auto_template_free',help="Run template free or template based structure prediction. Choose either 'auto_template_free' or 'auto_template_based'")
def create_tasks_from_dataframe(df_path,output_dir,mode,msa_option):
    """
    Creates batch tasks from a DataFrame.


    :param df_path: Path to CSV table with columns representing parameters:
        - 'job_name': str
        - 'type': 'protein', 'rna', 'dna', or 'ligand'
        - 'id': str or list
        - 'sequence': str
        - 'mode': 'default', 'pulldown', 'all-vs-all','stoichio-screen'
        - Other optional parameters:
            - 'modifications': list of dicts (as JSON string)
            - 'msa_option': 'auto', 'upload', or 'none'
            - 'unpaired_msa': str
            - 'paired_msa': str
            - 'templates': list of dicts (as JSON string)
            - 'model_seeds': list of integers (as string)
            - 'bonded_atom_pairs': list (as JSON string)
            - 'user_ccd': str
    :param mode: one of the following: 'default','all-vs-all','pulldown', 'stoichio-screen'
    :param output_dir: Directory to save json files.
    :type df_path: file
    :type mode: str
    :type output_dir: directory
    :return: Writes one json path per task to output_dir. Returns a list of paths and a list of job_names
    :rtype: tuple of lists
    """

    tasks = []
    job_names = []
    os.makedirs(output_dir,exist_ok=True)
    df=pd.read_csv(df_path)
    if mode != "default":
        df = create_df_for_run_mode(df,mode,msa_option)
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
            msa_option = row.get('msa_option', msa_option)
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

            # Get job-level parameters (assuming they are the same for all entities in the group)
            if model_seeds is None and pd.notna(row.get('model_seeds')):
                model_seeds = parse_list_field(row.get('model_seeds'), data_type=int)
            if bonded_atom_pairs is None and pd.notna(row.get('bonded_atom_pairs')):
                bonded_atom_pairs = parse_json_field(row.get('bonded_atom_pairs'))
            if user_ccd is None and pd.notna(row.get('user_ccd')):
                user_ccd = row.get('user_ccd')

        if model_seeds is None:
            model_seeds = [1]  # Default seed if not provided

        task = create_batch_task(
            job_name=job_name,
            entities=entities,
            model_seeds=model_seeds,
            bonded_atom_pairs=bonded_atom_pairs,
            user_ccd=user_ccd
        )
        input_json_path = os.path.join(output_dir, f"{job_name}.json")
        with open(input_json_path, "w") as outfile:
            json.dump(task, outfile)
        tasks.append(input_json_path)
        job_names.append(f"{job_name}")
    return tasks, job_names


if __name__ == "__main__":
    create_tasks_from_dataframe()
