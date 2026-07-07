rule MERGE_MONO_AND_MULTI_JSON:
    input:
        unpack(get_merge_inputs)
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_MERGE_MONOMERS_TO_MULTIMERS", "{multi}.log"),
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_MERGE_MONOMERS_TO_MULTIMERS", "{multi}.tsv"),
    resources:
        mem_mb  = 2000,
        runtime = 10,
    params:
        _helper = f"{WORKFLOW_DIR}/scripts/merge_mono_and_multi_jsons.py"
    output:
        os.path.join(OUTPUT_DIR,"rule_MERGE_MONOMERS_TO_MULTIMERS","{multi}_data.json") if MODE in ["custom","all-vs-all","pulldown","virtual-drug-screen","stoichio-screen"] else [],
    conda: "../envs/preprocessing.yaml"
    shell:
        """
        python {params._helper} {input} {output} \
        2>&1 | tee {log}
        """
