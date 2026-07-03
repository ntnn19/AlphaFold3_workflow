rule EXTRACT_SCORES:
    input:
        rules.IPSAE.output.ipsae_15_15,
        rules.IPSAE.output.ipsae_10_15,
        rules.AF3_INFERENCE.output.model,
        rules.AF3_INFERENCE.output.scores
    output:
        global_tsv    = expand(os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES", "{{multi}}", "{{multi}}_seed-{{seed}}_sample-{sample}_af_global.tsv"), sample=range(N_SAMPLES)) if MUTATION_DF.empty else expand(os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES","{{mut}}", "{{mut}}_seed-{{seed}}_sample-{sample}_af_global.tsv"), sample=range(N_SAMPLES)),
        per_chain_tsv = expand(os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES", "{{multi}}", "{{multi}}_seed-{{seed}}_sample-{sample}_af_per_chain.tsv"), sample=range(N_SAMPLES)) if MUTATION_DF.empty else expand(os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES", "{{mut}}", "{{mut}}_seed-{{seed}}_sample-{sample}_af_per_chain.tsv"), sample=range(N_SAMPLES)),
        per_pair_tsv  = expand(os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES", "{{multi}}", "{{multi}}_seed-{{seed}}_sample-{sample}_af_per_chain_pair.tsv"), sample=range(N_SAMPLES)) if MUTATION_DF.empty else expand(os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES", "{{mut}}", "{{mut}}_seed-{{seed}}_sample-{sample}_af_per_chain_pair.tsv"), sample=range(N_SAMPLES)),
        ipsae  = expand(os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES", "{{multi}}", "{{multi}}_seed-{{seed}}_sample-{sample}_ipsae.tsv"), sample=range(N_SAMPLES)) if MUTATION_DF.empty else expand(os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES", "{{mut}}", "{{mut}}_seed-{{seed}}_sample-{sample}_ipsae.tsv"), sample=range(N_SAMPLES)),
    params:
        inference_dir = os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{multi}") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{mut}"),
        out_dir = os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES", "{multi}") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES", "{mut}"),
        script = workflow.source_path("../scripts/extract_scores.py"),
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_EXTRACT_SCORES", "{multi}", "{multi}_seed-{seed}.log") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "logs", "rule_EXTRACT_SCORES","{mut}", "{mut}_seed-{seed}.log"),
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_EXTRACT_SCORES", "{multi}", "{multi}_seed-{seed}.tsv") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "benchmarks", "rule_EXTRACT_SCORES","{mut}", "{mut}_seed-{seed}.tsv"),
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


rule AGGREGATE_RESULTS:
    input:
        aggregate_outputs,
    output:
        global_tsv    = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all_af_global.tsv"),
        per_chain_tsv = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all_af_per_chain.tsv"),
        per_pair_tsv  = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all_af_per_chain_pair.tsv"),
        ipsae_tsv     = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all_ipsae.tsv"),
    params:
        script     = workflow.source_path("../scripts/aggregate_scores.py"),
        agg_dir    = os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES"),
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_AGGREGATE_RESULTS", "aggregate.log"),
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_AGGREGATE_RESULTS", "aggregate.tsv")
    conda: "../envs/preprocessing.yaml"
    shell:
        """
        find {params.agg_dir} -name '*_af_global.tsv'        ! -name 'all_*' | sort > {params.agg_dir}/filelist_af_global.txt
        find {params.agg_dir} -name '*_af_per_chain.tsv'     ! -name 'all_*' | sort > {params.agg_dir}/filelist_af_per_chain.txt
        find {params.agg_dir} -name '*_af_per_chain_pair.tsv' ! -name 'all_*' | sort > {params.agg_dir}/filelist_af_per_chain_pair.txt
        find {params.agg_dir} -name '*_ipsae.tsv' ! -name 'all_*' | sort > {params.agg_dir}/filelist_ipsae.txt

        python {params.script} {params.agg_dir}/filelist_af_global.txt        {output.global_tsv}    2>> {log}
        python {params.script} {params.agg_dir}/filelist_af_per_chain.txt     {output.per_chain_tsv} 2>> {log}
        python {params.script} {params.agg_dir}/filelist_af_per_chain_pair.txt {output.per_pair_tsv} 2>> {log}
        python {params.script} {params.agg_dir}/filelist_ipsae.txt            {output.ipsae_tsv}    2>> {log}
        """
