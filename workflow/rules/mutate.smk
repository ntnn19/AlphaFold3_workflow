checkpoint MUTATE:
    input:
        json = expand(os.path.join(OUTPUT_DIR,"rule_MERGE_MONOMERS_TO_MULTIMERS","{multi}_data.json"), multi=get_multi_to_monomeric_dict_),
        mutation_list = MUTATION_DF_PATH if MUTATION_DF_PATH is not None else []
    output:
        directory(os.path.join(OUTPUT_DIR, "rule_MUTATE")) if MUTATION_DF_PATH is not None else []
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_MUTATE", "MUTATE.log")
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_MUTATE", "MUTATE.tsv"),
    params:
        output_dir = OUTPUT_DIR,
        data_dir   = os.path.join(OUTPUT_DIR, "rule_MERGE_MONOMERS_TO_MULTIMERS"),
        _helper = workflow.source_path("../scripts/mutate.py")
    conda:
        "../envs/preprocessing.yaml"
    shell:
        """
        find {params.data_dir} -maxdepth 1 -type f -name '*_data.json' -print0 \
        | parallel -0 -j {threads} \
            python {params.helper_} {{}} {input.mutation_list} {params.output_dir}/rule_MUTATE
        """
