import pandas as pd
import pdb
import json
import os
from pathlib import Path
import copy
import click

MSA_KEYS = {"unpairedMsa", "unpairedMsaPath", "pairedMsa", "pairedMsaPath", "templates"}
IDENTITY_KEYS = ["sequence", "ccdCodes", "smiles"]

def get_chain_identity(s: dict):
    for k in IDENTITY_KEYS:
        if k in s:
            v = s[k]
            return (k, tuple(v) if isinstance(v, list) else v)
    raise KeyError(f"No identity key found in sequence entry: {s.keys()}")

@click.command()
@click.argument("multimer_file", type=click.Path(exists=True))
@click.argument("monomer_file", type=click.Path(exists=True), nargs=-1)
@click.argument("output_file", type=click.Path())  # where merged JSON will be saved
@click.option("--inference-to-data-map", type=click.Path())  # where merged JSON will be saved
def main(multimer_file, monomer_file, output_file, inference_to_data_map):
    """
    Merge monomer JSONs into a multimer JSON for a given sample.
    """

    if inference_to_data_map:
        inference_to_data_map_df = pd.read_csv(inference_to_data_map,sep="\t")
        job_map_df  = inference_to_data_pipeline_map_df[inference_to_data_pipeline_map_df.multimer_file.str.contains(Path(multimer_file).stem)]
        input_multimer_file = job_map_df.multimer_file.unique()[0]
        grouped = job_map_df.groupby('multimer_file')
        for input_multimer_file_, group in grouped:
            with open(input_multimer_file, "r") as f:
                multimer_data = json.load(f)
    
            merged_multimer = copy.deepcopy(multimer_data)
    
            # 2. Create a reference map for the target chains
            # Key = 'A', 'B', etc. | Value = the inner dictionary reference
            target_map = {}
            for item in merged_multimer['sequences']:
                inner_dict = list(item.values())[0]
                target_map[inner_dict['id']] = inner_dict
    
            # 3. Process each mapping row
            for _, row in group.iterrows():
                monomer_input_file = row["monomer_file"]
                target_chain_id = row["monomer_chain_id"]  # e.g., "B"
    
                with open(monomer_input_file, "r") as f:
                    monomer_data = json.load(f)
                    # Extract the source content (the first sequence in the monomer file)
                    source_content = list(monomer_data['sequences'][0].values())[0]
    
                    if target_chain_id in target_map:
                        # Store the original ID to prevent it from being overwritten
                        original_id = target_map[target_chain_id]['id']
    
                        # INJECT: Update the multimer's dict with monomer's data
                        target_map[target_chain_id].update(source_content)
    
                        # RESTORE: Ensure the ID remains 'B' even if source was 'A'
                        target_map[target_chain_id]['id'] = original_id
                        

    # 1️⃣ Load multimer JSON
    else:
            with open(multimer_file, "r") as f:
                multimer_data = json.load(f)
    
            merged_multimer = copy.deepcopy(multimer_data)
    
            # Build identity -> (monomer_file, monomer_chain_id) lookup from all monomer files
            # Build identity -> (monomer_file, monomer_chain_id, monomer_seq_dict) lookup
            monomer_lookup = {}
            for mf in monomer_file:
                for entry in json.load(open(mf))["sequences"]:
                    s = next(iter(entry.values()))
                    _, identity = get_chain_identity(s)
                    monomer_lookup[identity] = (mf, s["id"], s)
            
            # Map each multimer chain to its matching monomer entry
            chain_map = {}
            for entry in merged_multimer["sequences"]:
                s = next(iter(entry.values()))
                _, identity = get_chain_identity(s)
                if identity in monomer_lookup:
                    chain_map[(multimer_file, s["id"])] = monomer_lookup[identity]
            
            # Merge MSA keys from matched monomer into each multimer chain
            for seq in merged_multimer["sequences"]:
                s = next(iter(seq.values()))
                for k in MSA_KEYS:
                    s.pop(k, None)
                chain_id = s["id"]
                if (multimer_file, chain_id) not in chain_map:
                    continue
                _, _, monomer_s = chain_map[(multimer_file, chain_id)]
                s.update({k: monomer_s[k] for k in MSA_KEYS if k in monomer_s})

    
        # 4️⃣ For each model seed, write a separate merged JSON
    for s in merged_multimer["modelSeeds"]:
        seed_merged_multimer = copy.deepcopy(merged_multimer)
        seed_merged_multimer["modelSeeds"] = [s]
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(seed_merged_multimer, f, indent=4)
    print(f"Merged JSON written to {output_file}")

if __name__ == "__main__":
    main()
