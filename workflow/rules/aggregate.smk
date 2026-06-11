rule AGGREGATE_RESULTS:
    """Aggregate per-sample AF3 confidence outputs into three TSV tables.

    For each inference job, reads all seed-*/sample-* subdirectories and
    produces:
      - {multi}_global.tsv        one row per sample
      - {multi}_per_chain.tsv     one row per (sample, chain)
      - {multi}_per_chain_pair.tsv one row per (sample, chain_i, chain_j)

    Chain IDs are derived from token_chain_ids in confidences.json, so this
    rule works regardless of the AF3 version installed in the container.
    """
    input:
        cif = os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{multi}", "{multi}_model.cif") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{mut}", "{mut}_model.cif"),
        ipsae_15_15 = os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{multi}", "{multi}_model_15_15.txt") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{mut}", "{mut}_model_15_15.txt"),
        ipsae_10_10 = os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{multi}", "{multi}_model_10_10.txt") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{mut}", "{mut}_model_10_10.txt"),
    output:
        global_tsv    = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "{multi}_global.tsv") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "{mut}_global.tsv"),
        per_chain_tsv = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "{multi}_per_chain.tsv") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "{mut}_per_chain.tsv"),
        per_pair_tsv  = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "{multi}_per_chain_pair.tsv") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "{mut}_per_chain_pair.tsv"),
    params:
        inference_dir = os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{multi}") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{mut}"),
        script        = workflow.source_path("../scripts/aggregate_results.py"),
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_AGGREGATE_RESULTS", "{multi}.log") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "logs", "rule_AGGREGATE_RESULTS", "{mut}.log"),
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_AGGREGATE_RESULTS", "{multi}.tsv") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "benchmarks", "rule_AGGREGATE_RESULTS", "{mut}.tsv"),
    resources:
        mem_mb  = 2000,
        runtime = 10,
    conda: "../envs/preprocessing.yaml"
    shell:
        """
        python {params.script} \
            {params.inference_dir} \
            {output.global_tsv} \
            {output.per_chain_tsv} \
            {output.per_pair_tsv} \
        2>&1 | tee {log}
        """




rule META_AGGREGATE:
    """Concatenate all per-job TSVs into three project-level summary tables.

    Uses a file-of-filenames approach to avoid shell argument-length limits
    (ARG_MAX), which would be hit when concatenating thousands of per-job files.
    Produces:
      - all_global.tsv
      - all_per_chain.tsv
      - all_per_chain_pair.tsv
    """
    input:
        aggregate_outputs,
    output:
        global_tsv    = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all_global.tsv"),
        per_chain_tsv = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all_per_chain.tsv"),
        per_pair_tsv  = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all_per_chain_pair.tsv"),
    params:
        script     = workflow.source_path("../scripts/meta_aggregate.py"),
        agg_dir    = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS"),
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_META_AGGREGATE", "meta_aggregate.log"),
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_META_AGGREGATE", "meta_aggregate.tsv"),
    resources:
        mem_mb  = 4000,
        runtime = 30,
    conda: "../envs/preprocessing.yaml"
    shell:
        """
        # Write filelists using find — no shell glob expansion, no ARG_MAX issue.
        # Exclude the all_*.tsv outputs themselves to avoid self-inclusion on reruns.
        find {params.agg_dir} -maxdepth 1 -name '*_global.tsv'        ! -name 'all_*' | sort > {params.agg_dir}/filelist_global.txt
        find {params.agg_dir} -maxdepth 1 -name '*_per_chain.tsv'     ! -name 'all_*' | sort > {params.agg_dir}/filelist_per_chain.txt
        find {params.agg_dir} -maxdepth 1 -name '*_per_chain_pair.tsv' ! -name 'all_*' | sort > {params.agg_dir}/filelist_per_chain_pair.txt

        python {params.script} {params.agg_dir}/filelist_global.txt        {output.global_tsv}    2>> {log}
        python {params.script} {params.agg_dir}/filelist_per_chain.txt     {output.per_chain_tsv} 2>> {log}
        python {params.script} {params.agg_dir}/filelist_per_chain_pair.txt {output.per_pair_tsv} 2>> {log}
        """
