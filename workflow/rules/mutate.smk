#_helper = workflow.source_path("../scripts/mutate.py"), # added this to allow clean deployment
checkpoint MUTATE:
    input:
        json = expand(os.path.join(OUTPUT_DIR,"rule_MERGE_MONOMERS_TO_MULTIMERS","{multi}_data.json"), multi=get_multi_to_monomeric_dict_),
        mutation_list = MUTATION_DF_PATH if MUTATION_DF_PATH is not None else []
        #data = branch(
        #    lookup(query="sample_id == '{multi}'",within=INFERENCE_READY_DF,cols="file"),
        #    then=lookup(query="sample_id == '{multi}'",within=INFERENCE_READY_DF,cols="file"),
        #    otherwise=expand(os.path.join(OUTPUT_DIR,"rule_MERGE_MONOMERS_TO_MULTIMERS","{multi}_data.json"), multi=get_multi_to_monomeric_dict_)
        #),

    output:
        directory(os.path.join(OUTPUT_DIR, "rule_MUTATE")) if MUTATION_DF_PATH is not None else []
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_MUTATE", "MUTATE.log")
    resources:
        mem_mb      = 16000,
        runtime     = 480,
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_MUTATE", "MUTATE.tsv"),
    params:
        output_dir = OUTPUT_DIR,
        data_dir   = os.path.join(OUTPUT_DIR, "rule_MERGE_MONOMERS_TO_MULTIMERS"),
    threads: 8
    conda:
        "../envs/preprocessing.yaml"
    shell:
        """
        find {params.data_dir} -maxdepth 1 -type f -name '*_data.json' -print0 \
        | parallel -0 -j {threads} \
            python {WORKFLOW_DIR}/scripts/mutate.py {{}} {input.mutation_list} {params.output_dir}/rule_MUTATE
        """
