#!/usr/bin/env python
# Credit for https://github.com/hlasimpk/af3_mmseqs_scripts

import os
import json
from io import StringIO
import logging
from pathlib import Path

logger = logging.getLogger('logger')

from af3_script_utils import (
    custom_template_argpase_util,
    get_custom_template,
)


def run_custom_template(
    input_json,
    target_id,
    custom_template,
    custom_template_chain,
    output_json=None,
    to_file=True,
):
    # Determine if input_json is a path or dict
    if isinstance(input_json, (str, os.PathLike)) and Path(input_json).is_file():
        af3_json = json.load(open(input_json))
    elif isinstance(input_json, dict):
        af3_json = input_json
    else:
        raise ValueError("input_json must be a path to a JSON file or a dict")

    # Ensure custom template file exists
    if not os.path.exists(custom_template):
        raise FileNotFoundError(f"Custom template file {custom_template} not found")

    # Apply custom template to sequences if present
    if "sequences" in af3_json:
        for sequence in af3_json["sequences"]:
            if "protein" not in sequence:
                continue
            sequence.update(get_custom_template(
                sequence,
                target_id,
                custom_template,
                custom_template_chain,
            ))
    else:
        # If input_json was a single sequence dict
        af3_json = get_custom_template(
            af3_json,
            target_id,
            custom_template,
            custom_template_chain,
        )

    # Handle writing output if requested
    if to_file:
        if not output_json:
            if isinstance(input_json, (str, os.PathLike)):
                output_json = input_json  # overwrite original file
            else:
                raise ValueError("output_json must be provided when input_json is a dict")

        with open(output_json, "w") as f:
            json.dump(af3_json, f, indent=2)

    return af3_json

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add custom template to alphafold input JSON"
    )

    parser.add_argument("--input_json", help="Input alphafold3 json file")
    parser.add_argument("--output_json", help="Output alphafold3 json file")
    parser = custom_template_argpase_util(parser)

    args = parser.parse_args()

    run_custom_template(
        args.input_json,
        args.target_id,
        args.custom_template,
        args.custom_template_chain,
        output_json=args.output_json,
        to_file=False,
    )
