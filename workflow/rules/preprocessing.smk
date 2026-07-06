has_seeds = (
    "model_seeds" in RAW_DATA_DF.columns
    and RAW_DATA_DF["model_seeds"].notna().any()
    and RAW_DATA_DF["model_seeds"].astype(str).str.strip().ne("").any()
)

if N_SEEDS:
    N_SEEDS_ = f"--n-seeds {N_SEEDS}"
elif has_seeds:
    N_SEEDS_ = ""
else:
    N_SEEDS_ = "--n-seeds 1"

checkpoint PREPROCESSING:
    input:
        sample_sheet = RAW_DATA_PATH if not RAW_DATA_DF.empty else [],
        _helper = workflow.source_path("../scripts/prepare_af3_templates.py"),
    output:
        directory(os.path.join(OUTPUT_DIR,"rule_PREPROCESSING")) if not RAW_DATA_DF.empty else []
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_PREPROCESSING", "preprocessing.log"),
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_PREPROCESSING", "preprocessing.tsv"),
    params:
        mode       = MODE,
        n_seeds    = N_SEEDS_,
        n_samples  = N_SAMPLES,
        msa_option = MSA_OPTION,
        out_dir    = lambda w, output: str(Path(output[0]).parent),
        predict_individual_components = PREDICT_INDIVIDUAL_COMPONENTS,
        script     = workflow.source_path("../scripts/preprocessing.py"),
    resources:
        mem_mb = 4000,
        runtime = 60,
    conda: "../envs/preprocessing.yaml"
    shell:
        """
        python {params.script} \
        {input.sample_sheet} \
        {params.out_dir} \
        --mode={params.mode} \
        {params.n_seeds} \
        --n-samples {params.n_samples} {params.predict_individual_components} \
        2>&1 | tee {log}
        """