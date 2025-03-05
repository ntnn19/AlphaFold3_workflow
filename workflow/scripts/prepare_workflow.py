import click
import yaml
import os

@click.command()
@click.argument("config", type=click.Path(exists=True))
def setup_directories(config):
    """Reads CONFIG YAML file and creates necessary directories."""
    
    # Load YAML config
    with open(config, "r") as file:
        config_data = yaml.safe_load(file)
    
    # Extract required paths
    output_dir = config_data.get("output_dir")
    tmp_dir = config_data.get("tmp_dir")

    if not output_dir or not tmp_dir:
        click.echo("Error: 'output_dir' or 'tmp_dir' is missing in the config file.", err=True)
        return

    # Create directories
    preprocessing_dir = os.path.join(output_dir, "PREPROCESSING")

    for directory in [output_dir, tmp_dir, preprocessing_dir]:
        os.makedirs(directory, exist_ok=True)
        click.echo(f"Created or verified: {directory}")

if __name__ == "__main__":
    setup_directories()
