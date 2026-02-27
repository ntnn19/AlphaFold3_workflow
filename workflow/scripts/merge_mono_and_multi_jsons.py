import pandas as pd
import pdb
import json
import os
from pathlib import Path
import copy
import click

@click.command()
@click.argument("multimer_file", type=click.Path(exists=True))
@click.argument("monomer_file", type=click.Path(exists=True), nargs=-1)
@click.argument("output_file", type=click.Path())  # where merged JSON will be saved
@click.option("--inference-to-data-map", type=click.Path())  # where merged JSON will be saved
def main(multimer_file, monomer_file, output_file, inference_to_data_map):
    """
    Merge monomer JSONs into a multimer JSON for a given sample.
    """
    MSA_KEYS = {"unpairedMsa", "unpairedMsaPath", "pairedMsa", "pairedMsaPath", "templates"}
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
        t=pd.DataFrame([multimer_file]*len(monomer_file),columns= ["multimer_file"])
        t["monomer_file"] = monomer_file
        chain2seq_multimer = {list(s.values())[0]['id']: list(s.values())[0]['sequence'] for s in merged_multimer['sequences']}
        t["monomer_chain_id"] = t["monomer_file"].apply(lambda x: (s := next(iter(json.load(open(x))["sequences"][0].values())))["id"])
        t["multimer_chain_id"] = chain2seq_multimer.keys()
        t["multimer_chain_seq"] = chain2seq_multimer.values()
        t["monomer_chain_seq"] = t["monomer_file"].apply(lambda x: (s := next(iter(json.load(open(x))["sequences"][0].values())))["sequence"])
        chain_map = {(row["multimer_file"], row["multimer_chain_id"]): (row["monomer_file"], row["monomer_chain_id"]) for _, row in t[t["multimer_chain_seq"] == t["monomer_chain_seq"]].iterrows()}
        for seq in merged_multimer["sequences"]:
            s = next(iter(seq.values()))
            for k in MSA_KEYS:
                s.pop(k, None)
            chain_id = s["id"]
            monomer_file, _ = chain_map[(multimer_file, chain_id)]
            monomer_seq = next(iter(json.load(open(monomer_file))["sequences"][0].values()))
            s.update({k: monomer_seq[k] for k in MSA_KEYS if k in monomer_seq})

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
