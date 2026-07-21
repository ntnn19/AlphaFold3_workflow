#_helper = workflow.source_path("../scripts/mutate.py"), # added this to allow clean deployment
#
# MUTATE takes the UNION of every canonical single-seed JSON that could be mutated:
#   1. merged multimers  -> OUTPUT_DIR/rule_MERGE_MONOMERS_TO_MULTIMERS/*_data.json
#        (raw_data / data_pipeline_ready / merge_ready streams; already one seed
#         per multimer by construction/contract)
#   2. exploded inference_ready single-seed JSONs -> OUTPUT_DIR/normalized_inputs/
#        inference_ready/*.json   (written at load time in common.smk)
#
# Because mutation scope is "all streams", both sets are fed to scripts/mutate.py.
# mutate.py reads each JSON's own name, strips the seed, and only emits output when
# the (seed-stripped) base name matches a row in the mutations table — so listing a
# non-matching file in the manifest is a cheap no-op. mutate.py deep-copies the
# input, so the single modelSeeds value is preserved into every WT/variant output.
#
# The file list is streamed to GNU parallel via a manifest file (`parallel -a`)
# rather than argv, so the job is safe regardless of how many thousands of inputs
# exist (no ARG_MAX limit).
checkpoint MUTATE:
    input:
        # Merged multimers from raw_data / data_pipeline_ready / merge_ready
        # (forces the MERGE step to finish first). get_mutatable_merged_multimers
        # unions the raw_data stream (post-PREPROCESSING) with the load-time
        # merge_ready/dp_ready stems and, crucially, returns [] cleanly for an
        # inference_ready-only run (where PREPROCESSING produces no output).
        json = get_mutatable_merged_multimers,
        # Exploded inference_ready single-seed JSONs (written at load time). Declared
        # as inputs so MUTATE re-runs if any of them change; empty when no
        # inference_ready sheet was supplied.
        inference_ready = INFERENCE_READY_DF["file"].tolist() if not INFERENCE_READY_DF.empty else [],
        mutation_list = MUTATION_DF_PATH if MUTATION_DF_PATH is not None else []

    output:
        directory(os.path.join(OUTPUT_DIR, "rule_MUTATE")) if MUTATION_DF_PATH is not None else []
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_MUTATE", "MUTATE.log")
    resources:
        mem_mb      = 16000,
        runtime     = 480,
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_MUTATE", "MUTATE.tsv"),
    params:
        output_dir = OUTPUT_DIR,
        merge_dir  = os.path.join(OUTPUT_DIR, "rule_MERGE_MONOMERS_TO_MULTIMERS"),
        norm_dir   = NORMALIZED_INPUTS_DIR,
    threads: 8
    conda:
        "../envs/preprocessing.yaml"
    shell:
        r"""
        set -euo pipefail
        mkdir -p {params.output_dir}/rule_MUTATE
        manifest="$(mktemp)"
        trap 'rm -f "$manifest"' EXIT

        # Union of both canonical single-seed sources. Merged multimers use the
        # *_data.json convention; exploded inference_ready files are plain *.json.
        # `-print0 | sort -zu` de-duplicates and is null-safe for odd filenames.
        {{
            find {params.merge_dir} -maxdepth 1 -type f -name '*_data.json' -print0 2>/dev/null || true
            if [ -d "{params.norm_dir}" ]; then
                find {params.norm_dir} -maxdepth 1 -type f -name '*.json' -print0 2>/dev/null || true
            fi
        }} | sort -zu > "$manifest"

        n=$(tr -cd '\0' < "$manifest" | wc -c)
        echo "MUTATE: $n candidate JSON(s) in manifest (merged multimers + exploded inference_ready)" | tee {log}

        if [ "$n" -eq 0 ]; then
            echo "MUTATE: no candidate inputs found; nothing to mutate." | tee -a {log}
            exit 0
        fi

        parallel -0 -a "$manifest" -j {threads} \
            python {WORKFLOW_DIR}/scripts/mutate.py {{}} {input.mutation_list} {params.output_dir}/rule_MUTATE \
            2>&1 | tee -a {log}
        """
