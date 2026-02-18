import glob
import json
import os
from pathlib import Path
import pandas as pd
import copy
import click

@click.command()
@click.argument("inference_to_data_pipeline_map", type=click.Path())
@click.argument("inference_sample_sheet", type=click.Path())
@click.argument("job_name",  type=str)
def main(inference_to_data_pipeline_map,inference_sample_sheet,job_name):
    # Separate monomeric and multimeric files
    inference_to_data_pipeline_map_df = pd.read_csv(inference_to_data_pipeline_map,sep="\t")
    inference_sample_sheet_df = pd.read_csv(inference_sample_sheet,sep="\t")
    job_map_df  = inference_to_data_pipeline_map_df[inference_to_data_pipeline_map_df.inference_samples.str.contains(job_name)]
    job_inference_sample_sheet_df = inference_sample_sheet_df[inference_sample_sheet_df.job_name.str.contains(job_name)]
    input_multimer_file = job_map_df.inference_samples.unique()[0]

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
            monomer_input_file = row["monomer_files"]
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

    for s in merged_multimer["modelSeeds"]:
        seed_merged_multimer = copy.deepcopy(merged_multimer)
        seed_merged_multimer["modelSeeds"] = [s]
        output_path_basename = f"{job_name}_seed-{s}"
        merged_json_path = job_inference_sample_sheet_df.loc[job_inference_sample_sheet_df.job_name==output_path_basename,"inference_samples"].unique()
        os.makedirs(os.path.dirname(merged_json_path[0]), exist_ok=True)
        with open(merged_json_path[0], "w") as merged_json_path_:
            json.dump(seed_merged_multimer, merged_json_path_,indent=4)



if __name__ == "__main__":
    main()
