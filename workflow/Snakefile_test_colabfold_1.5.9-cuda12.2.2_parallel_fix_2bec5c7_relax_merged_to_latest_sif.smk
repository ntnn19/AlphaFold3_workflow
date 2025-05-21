configfile: "config/config.yaml"
import os
import pandas as pd
MONOMERS ="input/monomers.csv"
MULTIMERS ="input/multimers.csv"
MONOMERS_TEMPLATE_BASED ="input/monomers_template.csv"
MULTIMERS_TEMPLATE_BASED ="input/multimers_template.csv"

def get_wildcards(path):
    df = pd.read_csv(path)
    return df["id"].tolist()

MONOMERS_WC = get_wildcards(MONOMERS)
MULTIMERS_WC = get_wildcards(MULTIMERS)
MONOMERS_TEMPLATE_BASED_WC = get_wildcards(MONOMERS_TEMPLATE_BASED)
MULTIMERS_TEMPLATE_BASED_WC = get_wildcards(MULTIMERS_TEMPLATE_BASED)

CONTAINER = "/gpfs/cssb/group/cssb-topf/natan/singularity_containers/colabfold/colabfold_1.5.9-cuda12.2.2_parallel_fix_2bec5c7_relax_merged_to_latest.sif"
rule all:
    input:
        expand(os.path.join("output","COLABFOLD_SEARCH_NO_TEMPLATES_MONOMERS","{q}.a3m"),q=MONOMERS_WC),
        expand(os.path.join("output","COLABFOLD_SEARCH_NO_TEMPLATES_MULTIMERS","{q}.a3m"),q=MULTIMERS_WC),
        expand(os.path.join("output","COLABFOLD_SEARCH_TEMPLATES_MONOMERS","{q}.a3m"),q=MONOMERS_TEMPLATE_BASED_WC),
        expand(os.path.join("output","COLABFOLD_SEARCH_TEMPLATES_MULTIMERS","{q}.a3m"),q=MULTIMERS_TEMPLATE_BASED_WC)

rule COLABFOLD_SEARCH_TEMPLATES_MONOMERS_TEMPLATE_BASED:
    input:
        MONOMERS
    output:
        expand(os.path.join("output","COLABFOLD_SEARCH_TEMPLATES_MONOMERS_TEMPLATE_BASED","{q}.a3m"),q=MONOMERS_TEMPLATE_BASED_WC),
    container:
        CONTAINER
    shell:
        """
        colabfold_search --mmseqs /usr/local/envs/colabfold/bin/mmseqs {input} /database /predictions --use-templates 1 --db2 pdb100_230517 --use-env 1
        """


rule COLABFOLD_SEARCH_TEMPLATES_MULTIMERS_TEMPLATE_BASED:
    input:
        MONOMERS
    output:
        expand(os.path.join("output","COLABFOLD_SEARCH_TEMPLATES_MULTIMERS_TEMPLATE_BASED","{q}.a3m"),q=MULTIMERS_TEMPLATE_BASED_WC),
    container:
        CONTAINER
    shell:
        """
        colabfold_search --mmseqs /usr/local/envs/colabfold/bin/mmseqs {input} /database /predictions --use-templates 1 --db2 pdb100_230517 --use-env 1
        """


rule COLABFOLD_SEARCH_NO_TEMPLATES_MONOMERS:
    input:
        MONOMERS
    output:
        expand(os.path.join("output","COLABFOLD_SEARCH_NO_TEMPLATES_MONOMERS","{q}.a3m"),q=MONOMERS_WC),
    container:
        CONTAINER
    shell:
        """
        colabfold_search --mmseqs /usr/local/envs/colabfold/bin/mmseqs {input} /database /predictions
        """

rule COLABFOLD_SEARCH_NO_TEMPLATES_MULTIMERS:
    input:
        MULTIMERS
    output:
        expand(os.path.join("output","COLABFOLD_SEARCH_NO_TEMPLATES_MULTIMERS","{q}.a3m"),q=MULTIMERS_WC),
    container:
        CONTAINER
    shell:
        """
        colabfold_search --mmseqs /usr/local/envs/colabfold/bin/mmseqs {input} /database /predictions
        """
