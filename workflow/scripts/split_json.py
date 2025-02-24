import click
import json
import os
import string

def sanitised_name(name):
    """Returns sanitised version of the name that can be used as a filename."""
    lower_spaceless_name = name.lower().replace(' ', '_')
    allowed_chars = set(string.ascii_lowercase + string.digits + '_-.')
    return ''.join(l for l in lower_spaceless_name if l in allowed_chars)

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def split_json(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    """Splits a JSON file containing a list of dictionaries into separate JSON files."""
    with open(input_file, 'r') as f:
        data = json.load(f)
    data['name'] = sanitised_name(data["name"])
#    output_path = f"{output_dir}/{os.path.splitext(os.path.basename(input_file))[0]}.json"
    output_path = f"{output_dir}/{data['name']}.json"
    with open(output_path, 'w') as out_f:
        json.dump(data, out_f, indent=4)
    return output_path, data['name']

if __name__ == "__main__":
    split_json()
