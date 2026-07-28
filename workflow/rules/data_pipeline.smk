rule AF3_DATA_SPEEDY_PIPELINE:
    input:
        data = os.path.join(OUTPUT_DIR,"rule_PREPROCESSING","monomers","{mono}.json")
    params:
        mode = MODE,
        models_dir = MODELS_DIR,
        databases_dir = DB_DIR,
        extra_af3_data_flags = EXTRA_AF3_DATA_FLAGS,
        output_dir = lambda w, output: str(Path(output[0]).parents[2])
    output:
        data_pipeline_monomers=os.path.join(OUTPUT_DIR,"rule_AF3_DATA_PIPELINE","{mono}","{mono}_data.json"),
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_AF3_DATA_PIPELINE", "{mono}.log"),
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_AF3_DATA_PIPELINE", "{mono}.tsv"),
    container:
        AF3_CONTAINER
    shell:
        """
        python /app/alphafold/run_alphafold.py --json_path={input.data} \
        --model_dir={params.models_dir} \
        --output_dir={params.output_dir}/rule_AF3_DATA_PIPELINE \
        --db_dir={params.databases_dir} \
        --run_data_pipeline=true \
        --run_inference=false \
        {params.extra_af3_data_flags} \
        2>&1 | tee {log}
        """
