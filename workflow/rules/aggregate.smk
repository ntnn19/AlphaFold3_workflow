rule EXTRACT_SCORES:
    input:
        rules.IPSAE.output.ipsae_15_15,
        rules.IPSAE.output.ipsae_10_15,
        rules.IPSAE.input.model,
        rules.IPSAE.input.confidences
    output:
        global_tsv    = temp(os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES","batch_{batch}", "{job_name}", "{job_name}_seed-{seed}_sample-{sample}_af_global.tsv")) if MUTATION_DF.empty else temp(os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES","batch_{batch}","{mut}", "{mut}_seed-{seed}_sample-{sample}_af_global.tsv")),
        per_chain_tsv = temp(os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES","batch_{batch}", "{job_name}", "{job_name}_seed-{seed}_sample-{sample}_af_per_chain.tsv")) if MUTATION_DF.empty else temp(os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES","batch_{batch}", "{mut}", "{mut}_seed-{seed}_sample-{sample}_af_per_chain.tsv")),
        per_pair_tsv  = temp(os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES","batch_{batch}", "{job_name}", "{job_name}_seed-{seed}_sample-{sample}_af_per_chain_pair.tsv")) if MUTATION_DF.empty else temp(os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES","batch_{batch}", "{mut}", "{mut}_seed-{seed}_sample-{sample}_af_per_chain_pair.tsv")),
        ipsae  = temp(os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES","batch_{batch}", "{job_name}", "{job_name}_seed-{seed}_sample-{sample}_ipsae.tsv")) if MUTATION_DF.empty else temp(os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES","batch_{batch}", "{mut}", "{mut}_seed-{seed}_sample-{sample}_ipsae.tsv"))
    params:
        inference_dir = os.path.join(OUTPUT_DIR, "rule_ALPHAFAST", "batch_{batch}", "{job_name}") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "rule_ALPHAFAST","batch_{batch}", "{mut}"),
        out_dir = os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES","batch_{batch}", "{job_name}") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES","batch_{batch}", "{mut}"),
        helper_ = workflow.source_path("../scripts/extract_scores.py")
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_EXTRACT_SCORES","batch_{batch}", "{job_name}", "{job_name}_seed-{seed}_sample-{sample}.log") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "logs", "rule_EXTRACT_SCORES","batch_{batch}", "{mut}", "{mut}_seed-{seed}.log"),
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_EXTRACT_SCORES","batch_{batch}", "{job_name}", "{job_name}_seed-{seed}_sample-{sample}.tsv") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "benchmarks", "rule_EXTRACT_SCORES","batch_{batch}", "{mut}", "{mut}_seed-{seed}.tsv"),
    conda: "../envs/preprocessing.yaml"
    shell:
        """
        (
        set -euo pipefail
        set -x
        python {params.helper_} \
            --inference_dir {params.inference_dir} \
            --seed {wildcards.seed} \
            --sample {wildcards.sample} \
            --global_tsv {output.global_tsv} \
            --per_chain_tsv {output.per_chain_tsv} \
            --per_pair_tsv {output.per_pair_tsv} \
            --ipsae_tsv {output.ipsae}
        ) > {log} 2>&1
        """

rule AGGREGATE_WITHIN_BATCHES:
    input:
        get_extract_scores_outputs,
    output:
        global_tsv    = temp(os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS","batch_{batch}", "all_af_global.tsv")),
        per_chain_tsv = temp(os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS","batch_{batch}", "all_af_per_chain.tsv")),
        per_pair_tsv  = temp(os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS","batch_{batch}", "all_af_per_chain_pair.tsv")),
        ipsae_tsv     = temp(os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS","batch_{batch}", "all_ipsae.tsv")),
    params:
        helper_     = workflow.source_path("../scripts/aggregate_scores.py"),
        agg_dir    = os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES","batch_{batch}")
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_AGGREGATE_RESULTS","batch_{batch}", "aggregate.log"),
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_AGGREGATE_RESULTS","batch_{batch}", "aggregate.tsv")
    conda: "../envs/preprocessing.yaml"
    shell:
        """
        (
        set -euo pipefail
        set -x
        find {params.agg_dir} -name '*_af_global.tsv'        ! -name 'all_*' | sort > {params.agg_dir}/filelist_af_global.txt
        find {params.agg_dir} -name '*_af_per_chain.tsv'     ! -name 'all_*' | sort > {params.agg_dir}/filelist_af_per_chain.txt
        find {params.agg_dir} -name '*_af_per_chain_pair.tsv' ! -name 'all_*' | sort > {params.agg_dir}/filelist_af_per_chain_pair.txt
        find {params.agg_dir} -name '*_ipsae.tsv' ! -name 'all_*' | sort > {params.agg_dir}/filelist_ipsae.txt

        python {params.helper_} {params.agg_dir}/filelist_af_global.txt        {output.global_tsv}
        python {params.helper_} {params.agg_dir}/filelist_af_per_chain.txt     {output.per_chain_tsv}
        python {params.helper_} {params.agg_dir}/filelist_af_per_chain_pair.txt {output.per_pair_tsv}
        python {params.helper_} {params.agg_dir}/filelist_ipsae.txt            {output.ipsae_tsv}
        ) > {log} 2>&1
        """


rule AGGREGATE_ACROSS_BATCHES:
    input:
        unpack(get_aggregate_results_outputs)
    output:
        af_global         = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all", "all_af_global.tsv"),
        af_per_chain      = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all", "all_af_per_chain.tsv"),
        af_per_chain_pair = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all", "all_af_per_chain_pair.tsv"),
        ipsae             = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all", "all_ipsae.tsv"),
    params:
        base = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS"),
        out  = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all"),
        helper_     = workflow.source_path("../scripts/aggregate_batches.sh"),
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_AGGREGATE_RESULTS", "all", "aggregate.log"),
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_AGGREGATE_RESULTS", "all", "aggregate.tsv")
    shell:
        """
        (
        set -euo pipefail
        set -x
        bash {params.helper_} \
            {params.base} {params.out} \
            af_global af_per_chain af_per_chain_pair ipsae
        ) > {log} 2>&1
        """
        
        #rule MOCK:
#    input:
#        expand(os.path.join(OUTPUT_DIR, "flags", "alphafast_batch_{batch}.done"), batch=get_alphafast_inputs)
#    output: touch("done.txt")