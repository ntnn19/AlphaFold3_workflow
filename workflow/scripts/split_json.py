import json
import click

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=True, file_okay=False, writable=True))
def split_json(input_file, output_dir):
    """Splits a JSON file containing a list of dictionaries into separate JSON files."""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of dictionaries.")
    
    for i, subdict in enumerate(data):
        if not isinstance(subdict, dict):
            raise ValueError(f"Element at index {i} is not a dictionary.")
        subdict["modelSeeds"]= [1]
        subdict["dialect"]= "alphafold3"
        subdict["version"]= 1
        subdict["sequences"][-1]["ligand"]["id"]=subdict["sequences"][-1]["ligand"]["id"][0]
        output_path = f"{output_dir}/{subdict['name']}.json"
#        output_path = f"{output_dir}/combination_{i}.json"
        with open(output_path, 'w') as out_f:
            json.dump(subdict, out_f, indent=4)
        click.echo(f"Saved {output_path}")

if __name__ == "__main__":
    split_json()
