checkpoint MUTATE:
    input:
        _helper = workflow.source_path("../scripts/mutate.py"), # added this to allow clean deployment
        data = branch(
            lookup(query="sample_id == '{multi}'",within=INFERENCE_READY_DF,cols="file"),
            then=lookup(query="sample_id == '{multi}'",within=INFERENCE_READY_DF,cols="file"),
            otherwise=os.path.join(OUTPUT_DIR,"rule_MERGE_MONOMERS_TO_MULTIMERS","{multi}_data.json")
        ),
        mutation_list = MUTATION_DF_PATH if MUTATION_DF_PATH is not None else []
    output:
        directory(os.path.join(OUTPUT_DIR, "rule_MUTATE", "{multi}")) if MUTATION_DF_PATH is not None else []
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_MUTATE", "{multi}.log")
    resources:
        mem_mb      = 16000,
        runtime     = 480,
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_MUTATE", "{multi}.tsv"),
    params:
        output_dir = OUTPUT_DIR 
    conda: 
        "../envs/preprocessing.yaml"
    shell:
        """
        python {input._helper} {input.data} {input.mutation_list} {params.output_dir}/rule_MUTATE/{wildcards.multi}
        """
