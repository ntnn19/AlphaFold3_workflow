import pandas as pd
import json
import os
from pathlib import Path
import copy
from collections import defaultdict
import click
from loguru import logger


@click.command()
@click.argument("multimer_file", type=click.Path(exists=True))
@click.argument("monomer_file", type=click.Path(exists=True), nargs=-1)
@click.argument("output_file", type=click.Path())  # where merged JSON will be saved
@click.argument("inference-to-data-map", type=click.Path())
def main(multimer_file, monomer_file, output_file, inference_to_data_map):
    """
    Merge monomer JSONs into a multimer JSON for a given sample.
    The {multi} wildcard already encodes exactly one seed (e.g. job1_seed-10),
    so the input multimer JSON contains modelSeeds=[N] for a single seed.
    We write the merged result once to output_file.
    """

    inference_to_data_map_df = pd.read_csv(inference_to_data_map, sep="\t")
    job_map_df = inference_to_data_map_df[
        inference_to_data_map_df.multimer_file.apply(lambda x: Path(x).stem) == Path(multimer_file).stem
    ]
    input_multimer_file = job_map_df.multimer_file.unique()[0]


    with open(input_multimer_file, "r") as f:
        multimer_data = json.load(f)

    
    for chain_id, mono_file in zip(job_map_df.monomer_chain_id, job_map_df.monomer_file):
        with open(mono_file, "r") as f:
            monomer_data = json.load(f)
        monomer_content = monomer_data["sequences"][0]
        # take monomer data and merge it into multimer_data at chain_id
        for s in multimer_data["sequences"]:
            for k in s:
                if s[k]["id"] == chain_id:
                    original_id = s[k]["id"]
                    s[k] = list(monomer_content.values())[0]
                    s[k]["id"] = original_id
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(multimer_data, f, indent=4)
    logger.info("Merged JSON written to {}", output_file)


if __name__ == "__main__":
    main()
