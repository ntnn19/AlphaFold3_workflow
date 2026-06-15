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
        expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{{multi}}",
                         "seed-{{seed}}_sample-{sample}",
                         "{{multi}}_seed-{{seed}}_sample-{sample}_model_15_15.txt"),
            sample=range(N_SAMPLES)
        ) if MUTATION_DF.empty else
        expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{{mut}}",
                         "seed-{{seed}}_sample-{sample}",
                         "{{mut}}_seed-{{seed}}_sample-{sample}_model_15_15.txt"),
            sample=range(N_SAMPLES)
        ),
        expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{{multi}}",
                         "seed-{{seed}}_sample-{sample}",
                         "{{multi}}_seed-{{seed}}_sample-{sample}_model_10_15.txt"),
            sample=range(N_SAMPLES)
        ) if MUTATION_DF.empty else
        expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{{mut}}",
                         "seed-{{seed}}_sample-{sample}",
                         "{{mut}}_seed-{{seed}}_sample-{sample}_model_10_15.txt"),
            sample=range(N_SAMPLES)
        ),
        expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{{multi}}",
                            "seed-{{seed}}_sample-{sample}",
                            "{{multi}}_seed-{{seed}}_sample-{sample}_model.cif"),
            sample=range(N_SAMPLES)
        ) if MUTATION_DF.empty else
        expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{{mut}}",
                            "seed-{{seed}}_sample-{sample}",
                            "{{mut}}_seed-{{seed}}_sample-{sample}_model.cif"),
            sample=range(N_SAMPLES)
        ),
        expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{{multi}}",
                            "seed-{{seed}}_sample-{sample}",
                            "{{multi}}_seed-{{seed}}_sample-{sample}_confidences.json"),
            sample=range(N_SAMPLES)
        ) if MUTATION_DF.empty else
        expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{{mut}}",
                            "seed-{{seed}}_sample-{sample}",
                            "{{mut}}_seed-{{seed}}_sample-{sample}_confidences.json"),
            sample=range(N_SAMPLES)
        ),
    output:
        global_tsv    = expand(os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "{{multi}}", "{{multi}}_seed-{{seed}}_sample-{sample}_global.tsv"), sample=range(N_SAMPLES)) if MUTATION_DF.empty else expand(os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS","{{mul}}", "{{mut}}_seed-{{seed}}_sample-{sample}_global.tsv"), sample=range(N_SAMPLES)),
        per_chain_tsv = expand(os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "{{multi}}", "{{multi}}_seed-{{seed}}_sample-{sample}_per_chain.tsv"), sample=range(N_SAMPLES)) if MUTATION_DF.empty else expand(os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "{{mul}}", "{{mut}}_seed-{{seed}}_sample-{sample}_per_chain.tsv"), sample=range(N_SAMPLES)),
        per_pair_tsv  = expand(os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "{{multi}}", "{{multi}}_seed-{{seed}}_sample-{sample}_per_chain_pair.tsv"), sample=range(N_SAMPLES)) if MUTATION_DF.empty else expand(os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "{{mul}}", "{{mut}}_seed-{{seed}}_sample-{sample}_per_chain_pair.tsv"), sample=range(N_SAMPLES)),
#        touch(os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS","{multi}_seed-{seed}_done.txt" if MUTATION_DF.empty else "{mut}_seed-{seed}_done.txt"))
    params:
        inference_dir = os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{multi}") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{mut}"),
        out_dir = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "{multi}") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "{mut}"),
        script = workflow.source_path("../scripts/aggregate_results.py"),
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_AGGREGATE_RESULTS", "{multi}", "{multi}_seed-{seed}.log") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "logs", "rule_AGGREGATE_RESULTS",{mut}, "{mut}_seed-{seed}.log"),
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_AGGREGATE_RESULTS", "{multi}", "{multi}_seed-{seed}.tsv") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "benchmarks", "rule_AGGREGATE_RESULTS",{mut}, "{mut}_seed-{seed}.tsv"),            
    resources:
        mem_mb  = 2000,
        runtime = 10,
    conda: "../envs/preprocessing.yaml"
    shell:
        """
        python {params.script} \
            --inference_dir {params.inference_dir} \
            --out_dir {params.out_dir} \
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
    #resources:
    #    mem_mb  = 4000,
    #    runtime = 30,
    conda: "../envs/preprocessing.yaml"
    shell:
        """
        find {params.agg_dir} -maxdepth 1 -name '*_global.tsv'        ! -name 'all_*' | sort > {params.agg_dir}/filelist_global.txt
        find {params.agg_dir} -maxdepth 1 -name '*_per_chain.tsv'     ! -name 'all_*' | sort > {params.agg_dir}/filelist_per_chain.txt
        find {params.agg_dir} -maxdepth 1 -name '*_per_chain_pair.tsv' ! -name 'all_*' | sort > {params.agg_dir}/filelist_per_chain_pair.txt

        python {params.script} {params.agg_dir}/filelist_global.txt        {output.global_tsv}    2>> {log}
        python {params.script} {params.agg_dir}/filelist_per_chain.txt     {output.per_chain_tsv} 2>> {log}
        python {params.script} {params.agg_dir}/filelist_per_chain_pair.txt {output.per_pair_tsv} 2>> {log}
        """
