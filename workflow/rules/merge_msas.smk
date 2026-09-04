rule MERGE_MONO_AND_MULTI_JSON:
    input:
        unpack(get_merge_inputs)
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_MERGE_MONOMERS_TO_MULTIMERS", "{multi}.log"),
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_MERGE_MONOMERS_TO_MULTIMERS", "{multi}.tsv"),
    params:
        _helper = workflow.source_path("../scripts/merge_mono_and_multi_jsons.py"),
        chain_map = f"{OUTPUT_DIR}/preprocessing/metadata/inference_to_data_pipeline_map.tsv" if DATA_PIPELINE_READY_DF.empty else f"{NORMALIZED_INPUTS_DIR}/inference_to_data_pipeline_map.tsv"
    output:
        os.path.join(OUTPUT_DIR,"rule_MERGE_MONOMERS_TO_MULTIMERS","{multi}_data.json") if MODE in ["custom","all-vs-all","pulldown","virtual-drug-screen","stoichio-screen"] else [],
    conda: "../envs/preprocessing.yaml"
    shell:
        """
        python {params._helper} {input} {output} {params.chain_map} \
        2>&1 | tee {log}
        """
