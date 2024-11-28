import click
import json
from pathlib import Path

@click.command()
@click.argument("input_files", nargs=-1, type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
def join_json(input_files, output_file):
    """
    Joins multiple JSON files with top-level lists into a single JSON file.
    
    INPUT_FILES: Paths to the input JSON files.
    OUTPUT_FILE: Path to the output JSON file.
    """
    combined_data = []

    # Process each input file
    for input_file in input_files:
        with open(input_file, 'r') as f:
            data = json.load(f)

            # Ensure the top-level is a list
            if not isinstance(data, list):
                raise ValueError(f"The top-level of {input_file} must be a list.")
            
            combined_data.extend(data)  # Add items to the combined list

    # Write the combined list to the output file
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=2)

    click.echo(f"Combined {len(input_files)} files into {output_file}")

if __name__ == "__main__":
    join_json()
