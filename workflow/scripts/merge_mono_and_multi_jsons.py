import glob
import json
import os
from pathlib import Path
import click

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def extract_monomer_key(path):
    # e.g. "p1_auto_template_free_data.json" -> "p1_auto_template_free"
    return Path(path).stem.replace("_data", "")

def merge_jsons(multimer_data, monomer_data_list):
    """
    Merge monomer data into the multimer data.
    """
    for multimer_entity_dict in multimer_data["sequences"]:
        for monomer_json in monomer_data_list:
            if "protein" in multimer_entity_dict:
                if multimer_entity_dict["protein"]["sequence"] == monomer_json["sequences"][0]["protein"]["sequence"]:
                    multimer_entity_dict["protein"]["unpairedMsa"] = monomer_json["sequences"][0]["protein"]["unpairedMsa"]
                    multimer_entity_dict["protein"]["pairedMsa"] = monomer_json["sequences"][0]["protein"]["pairedMsa"]
                    multimer_entity_dict["protein"]["templates"] = monomer_json["sequences"][0]["protein"].get("templates", [])
            if "rna" in multimer_entity_dict:
                if multimer_entity_dict["rna"]["sequence"]  == monomer_json["sequences"][0]["rna"]["sequence"]:
                    multimer_entity_dict["rna"]["unpairedMsa"] = monomer_json["sequences"][0]["rna"]["unpairedMsa"]
                    multimer_entity_dict["rna"]["pairedMsa"] = monomer_json["sequences"][0]["rna"]["unpairedMsa"]
            if "dna" in multimer_entity_dict:
                if multimer_entity_dict["dna"]["sequence"]  == monomer_json["sequences"][0]["dna"]["sequence"]:
                    multimer_entity_dict["dna"]["unpairedMsa"] = monomer_json["sequences"][0]["dna"]["unpairedMsa"]
                    multimer_entity_dict["dna"]["pairedMsa"] = monomer_json["sequences"][0]["dna"]["unpairedMsa"]
    return multimer_data

@click.command()
@click.argument("monomers_dir", type=click.Path())
@click.argument("multimers_dir", type=click.Path())
@click.argument("output_dir",  type=click.Path())
def main(monomers_dir,multimers_dir,output_dir):
    # Separate monomeric and multimeric files
    monomer_patterns =  os.path.join(monomers_dir, "*_data.json")
    multimer_patterns =  os.path.join(multimers_dir, "*.json")
    monomer_files = list(glob.glob(monomer_patterns))[:1000]
    print("monomer_files=",monomer_files)
    multimer_files = list(glob.glob(multimer_patterns))[:1000]
    print("multimer_files=",multimer_files)
#    monomer_files = [p for p in json_files if "/monomers/" in p]
#    multimer_files = [p for p in json_files if "/multimers/" in p]

    # Create map from monomer key to JSON content
    monomer_data = {}
    for mf in monomer_files:
        key = extract_monomer_key(mf)
        monomer_data[key] = load_json(mf)

    # Merge monomer data into each multimer
    for multimer_path in multimer_files:
        multimer_name = os.path.splitext(os.path.basename(multimer_path))[0]  # e.g. "p1_b1_auto_template_free"
        print("multimer_name=",multimer_name)
        multimer_base = multimer_name + ".json"
        multimer_json = load_json(multimer_path)

        # Match all monomer keys that are part of the multimer name
        #matched_monomers = [
        #    data for key, data in monomer_data.items()
        #    if key in multimer_name
        #]
        matched_monomers = []
        for key, data in monomer_data.items():
            for suffix in ["_auto_template_free", "_auto_template_based"]:
                if suffix in multimer_name and key.split(suffix)[0] in multimer_name:
                    matched_monomers.append(data)
        click.echo(f"matched_monomers= {[m['name'] for m in matched_monomers]}")
        click.echo(f"multimer_json= {multimer_json}")
        if not matched_monomers:
            click.echo(f"Warning: No matching monomers found for {multimer_name}. Skipping...")
            continue
        if len(matched_monomers)==1:
            click.echo(f"Warning: Only one monomer found for {multimer_name}. Skipping...")
            continue
        merged = merge_jsons(multimer_json, matched_monomers)
        sub_output_dir = os.path.join(output_dir,multimer_name)
        os.makedirs(sub_output_dir, exist_ok=True)
        out_path = os.path.join(sub_output_dir,multimer_base.replace(".json","_data.json"))

        write_json(merged, out_path)
        click.echo(f"Wrote merged JSON to {out_path}")

if __name__ == "__main__":
    main()
