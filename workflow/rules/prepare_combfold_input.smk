import glob
import subprocess
import os
COMBFOLD_CONTAINER=config["combfold_container"]
DATASET=config["dataset"]
# seed models
#DATASETS_SEED, DATA_DIR_SEED,PAC_DATA_SEED, PREDICTION_SEED,PAC_PREDICTION_SEED, SEED, SAMPLE, SEED_PROTEIN,  = glob_wildcards("{dataset}/af_output/{data_dir}_pac{pac_data}/{prediction_dir}_pac{pac_prediction}/seed-{seed}_sample-{sample}/{p}.cif")
DATA_DIR_SEED,PAC_DATA_SEED, PREDICTION_SEED,PAC_PREDICTION_SEED, SEED, SAMPLE,  = glob_wildcards(DATASET + "/af_output/{data_dir}_pac{pac_data}/{prediction_dir}_pac{pac_prediction}/seed-{seed}_sample-{sample}/model.cif")
#print(list(zip(DATASETS_SEED,DATA_DIR_SEED, PAC_DATA_SEED,PREDICTION_SEED,PAC_PREDICTION_SEED, SEED,SAMPLE))[:10])
#print(list(zip(DATASETS_SEED,DATA_DIR_SEED, PAC_DATA_SEED,PREDICTION_SEED,PAC_PREDICTION_SEED, SEED,SAMPLE,SEED_PROTEIN))[:10])
PAC,SUBUNITS_JSON,  = glob_wildcards(DATASET + "/combfold_input/subunits_jsons_pac{pac}/subunits_{json}.json")
print(DATASET,SUBUNITS_JSON,PAC)
# rename files
# collect pac1 to an empty dir
# collect pac2 to an empty dir
# convert to pdb
# run combfold

#dataset_1/combfold_input/prepare_fastas/pac1/subunits_asymm/

rule all:
    input:
        expand(DATASET + "/combfold_input/all_pdbs_pac{pac}/{prediction_dir}_pac{pac}_seed-{seed}_sample-{sample}_model_no_dna.pdb",zip, prediction_dir=PREDICTION_SEED, seed=SEED, sample=SAMPLE,pac=PAC_DATA_SEED),
#        expand(DATASET + "/combfold_input/all_pdbs_pac{pac}/{prediction_dir}_pac{pac}_seed-{seed}_sample-{sample}_{p}_no_dna.pdb",zip, prediction_dir=PREDICTION_SEED, seed=SEED, sample=SAMPLE, p=SEED_PROTEIN,pac=PAC_DATA_SEED),
        expand(DATASET + "/combfold_output/pac{pac}/subunits_{json}/assembled_results/confidence.txt",  pac=PAC_DATA_SEED, json=SUBUNITS_JSON)

# Rule to create inference inputs and job list
rule RENAME_AND_COLLECT_PREDICTIONS:
    input:
        DATASET + "/af_output/{prediction_dir}_pac{pac}/{prediction_dir}_pac{pac}/seed-{seed}_sample-{sample}/model.cif"
    output:
        temp(DATASET + "/combfold_input/all_pdbs_pac{pac}/{prediction_dir}_pac{pac}_seed-{seed}_sample-{sample}_model.cif")
#    input:
#        DATASET + "/af_output/{prediction_dir}_pac{pac}/{prediction_dir}_pac{pac}/seed-{seed}_sample-{sample}/{p}.cif"
#    output:
#        temp(DATASET + "/combfold_input/all_pdbs_pac{pac}/{prediction_dir}_pac{pac}_seed-{seed}_sample-{sample}_{p}.cif")
    resources:
        slurm_account="cssb",
        slurm_partition="topfgpu",
#        constraint="A100",
        nodes=1,
        runtime=20,
    shell:
        """
        cp {input} {output}
        """

#rule CIF2_PDB:
#    input:
#        rules.RENAME_AND_COLLECT_PREDICTIONS.output
#    output:
#        temp(DATASET + "/combfold_input/all_pdbs_pac{pac}/{prediction_dir}_pac{pac}_seed-{seed}_sample-{sample}_model.pdb")
##        temp(DATASET + "/combfold_input/all_pdbs_pac{pac}/{prediction_dir}_pac{pac}_seed-{seed}_sample-{sample}_{p}.pdb")
#    resources:
#        slurm_account="cssb",
#        slurm_partition="topfgpu",
#        constraint="A100",
#        nodes=1,
#        runtime=20,
#    shell:
#        """
#        gemmi convert {input} {output}
#        """
#
#rule REMOVE_DNA_FROM_PDB:
#    input:
#        rules.CIF2_PDB.output
#    output:
#        DATASET + "/combfold_input/all_pdbs_pac{pac}/{prediction_dir}_pac{pac}_seed-{seed}_sample-{sample}_model_no_dna.pdb"
##        DATASET + "/combfold_input/all_pdbs_pac{pac}/{prediction_dir}_pac{pac}_seed-{seed}_sample-{sample}_{p}_no_dna.pdb"
#    resources:
#        slurm_account="cssb",
#        slurm_partition="topfgpu",
#        constraint="A100",
#        nodes=1,
#        runtime=20,
#    shell:
#        """
#        grep -v "DA" {input} | grep -v "DT" | grep -v "DG" | grep -v "DC" > {output}
#        """

rule REMOVE_DNA_FROM_CIF_AND_SAVE_AS_PDB:
    input:
        rules.RENAME_AND_COLLECT_PREDICTIONS.output
    output:
        DATASET + "/combfold_input/all_pdbs_pac{pac}/{prediction_dir}_pac{pac}_seed-{seed}_sample-{sample}_model_no_dna.pdb"
#        DATASET + "/combfold_input/all_pdbs_pac{pac}/{prediction_dir}_pac{pac}_seed-{seed}_sample-{sample}_{p}_no_dna.pdb"
    resources:
        slurm_account="cssb",
        slurm_partition="topfgpu",
        #constraint="A100",
        nodes=1,
        runtime=20,
    shell:
        """
        python workflow/scripts/remove_dna.py {input}
        """
#
#rule TOUCH:
#    input:
#        DATASET + "/combfold_input/subunits_jsons_pac{pac}/{json}.json"
#    output:
#        touch(DATASET + "/combfold_input/subunits_jsons_pac{pac}/{json}.json")

#rule RUN_COMBFOLD_SELECTION:
checkpoint RUN_COMBFOLD_SELECTION:
    input:
        subunits_json= DATASET + "/combfold_input/subunits_jsons_pac{pac}/subunits_{json}.json",	
        input_pdbs = expand(DATASET + "/combfold_input/all_pdbs_pac{pac}/{prediction_dir}_pac{pac}_seed-{seed}_sample-{sample}_model_no_dna.pdb",zip, prediction_dir=PREDICTION_SEED, seed=SEED, sample=SAMPLE,pac=PAC_DATA_SEED),
#        input_pdbs = expand(DATASET + "/combfold_input/all_pdbs_pac{pac}/{prediction_dir}_pac{pac}_seed-{seed}_sample-{sample}_{p}_no_dna.pdb",zip, prediction_dir=PREDICTION_SEED, seed=SEED, sample=SAMPLE, p=SEED_PROTEIN,pac=PAC_DATA_SEED),
    output:
        outdir=directory(DATASET + "/combfold_input/prepare_fastas_pac{pac}/{json}_fastas"),
    resources:
        slurm_account="cssb",
        slurm_partition="topfgpu",
        #constraint="A100",
        nodes=1,
        runtime=200,
    container:
        COMBFOLD_CONTAINER
#        config["combfold_container"]
    params:
        dataset = DATASET
    shell:
        """
        python /app/CombFold-master/scripts/prepare_fastas.py {input.subunits_json} --stage groups --output-fasta-folder {output.outdir} --max-af-size 7500 --input-pairs-results {params.dataset}/combfold_input/all_pdbs_pac{wildcards.pac}
        """
#        python /app/CombFold-master/scripts/prepare_fastas.py {input.subunits_json} --stage groups --output-fasta-folder {output.outdir} --max-af-size 7500 --input-pairs-results {wildcards.dataset}/combfold_input/all_pdbs_pac{wildcards.pac}
#

#rule INTERMEDIATE_2:
#    input:
#        DATASET + "/combfold_input/combfold_selected_pdbs_pac{pac}/{json}_pdbs/{pdb}.pdb"
#    output:
#        touch(DATASET + "/combfold_input/combfold_selected_pdbs_pac{pac}/{json}_pdbs/{pdb}_renamed.pdb")

def get_prepared_fastas(wildcards):
    #print("WD=",wildcards)
    ck_outputs = checkpoints.RUN_COMBFOLD_SELECTION.get(**wildcards).output[0]
    #print("CK=",ck_outputs)
    FASTA=glob_wildcards(os.path.join(ck_outputs, "{fasta}.fasta")).fasta
    return expand(DATASET + "/combfold_input/prepare_fastas_pac{pac}/{json}_fastas/{fasta}.fasta" ,pac=wildcards.pac,json=wildcards.json, fasta=FASTA)

checkpoint MATCH_COMBFOLD_SELECTED_FASTAS_AND_AF3_PDBS:
    input:
        subunits_json= DATASET + "/combfold_input/subunits_jsons_pac{pac}/subunits_{json}.json",	
        prepared_fastas = get_prepared_fastas,
        input_pdbs = expand(DATASET + "/combfold_input/all_pdbs_pac{pac}/{prediction_dir}_pac{pac}_seed-{seed}_sample-{sample}_model_no_dna.pdb",zip, prediction_dir=PREDICTION_SEED, seed=SEED, sample=SAMPLE,pac=PAC_DATA_SEED),
#        input_pdbs = expand(DATASET + "/combfold_input/all_pdbs_pac{pac}/{prediction_dir}_pac{pac}_seed-{seed}_sample-{sample}_{p}_no_dna.pdb",zip, prediction_dir=PREDICTION_SEED, seed=SEED, sample=SAMPLE, p=SEED_PROTEIN,pac=PAC_DATA_SEED),
    log:
        DATASET + "/logs/{pac}_{json}_rename_combfold_fasta_prep_to_match_af3_and_copy_selected_pdbs.py.log"
    resources:
        slurm_account="cssb",
        slurm_partition="topfgpu",
        #constraint="A100",
        nodes=1,
        runtime=20,
    output:
        directory(DATASET + "/combfold_input/combfold_selected_pdbs_pac{pac}/{json}_pdbs")
    params:
        dataset = DATASET
    shell:
        """
        python workflow/scripts/rename_combfold_fasta_prep_to_match_af3_and_copy_selected_pdbs.py {params.dataset}/combfold_input/prepare_fastas_pac{wildcards.pac}/{wildcards.json}_fastas  {params.dataset}/combfold_input/all_pdbs_pac{wildcards.pac} {output} {log}
        """


def get_matched_pdbs(wildcards):
    #print("WD=",wildcards)
    ck_outputs = checkpoints.MATCH_COMBFOLD_SELECTED_FASTAS_AND_AF3_PDBS.get(**wildcards).output[0]
    #print("CK=",ck_outputs)
    PDB=glob_wildcards(os.path.join(ck_outputs, "{pdb}.pdb")).pdb
    return expand(DATASET + "/combfold_input/combfold_selected_pdbs_pac{pac}/{json}_pdbs/{pdb}.pdb",pac=wildcards.pac,json=wildcards.json, pdb=PDB)

rule RUN_COMBFOLD_ASSEMBLY:
    input:
        subunits_pdbs= get_matched_pdbs,
        subunits_json= DATASET + "/combfold_input/subunits_jsons_pac{pac}/subunits_{json}.json",	
    output:
        DATASET + "/combfold_output/pac{pac}/subunits_{json}/assembled_results/confidence.txt"
    resources:
        slurm_account="cssb",
        slurm_partition="topfgpu",
        #constraint="A100",
        nodes=1,
        runtime=20000,
    container:
        COMBFOLD_CONTAINER
#        config["combfold_container"]
    params:
        dataset = DATASET
    shell:
        """
        echo "Contents of output dir before execution:"
        ls -la {params.dataset}/combfold_output/pac{wildcards.pac}/subunits_{wildcards.json}
        rm -rf {params.dataset}/combfold_output/pac{wildcards.pac}/subunits_{wildcards.json}
        python /app/CombFold-master/scripts/run_on_pdbs.py {input.subunits_json} {params.dataset}/combfold_input/combfold_selected_pdbs_pac{wildcards.pac}/{wildcards.json}_pdbs {params.dataset}/combfold_output/pac{wildcards.pac}/subunits_{wildcards.json}
        """
#rule TEST_CHECKPONT:
#    input:
#        get_matched_pdbs
#    output:
#        touch(DATASET + "/combfold_input/combfold_selected_pdbs_pac{pac}_{json}_done.txt")

