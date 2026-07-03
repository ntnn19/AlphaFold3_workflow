checkpoint MUTATE:
    input:
        _helper = workflow.source_path("../scripts/mutate.py"), # added this to allow clean deployment
        data = branch(
            lookup(query="sample_id == '{multi}'",within=INFERENCE_READY_DF,cols="file"),
            then=lookup(query="sample_id == '{multi}'",within=INFERENCE_READY_DF,cols="file"),
            otherwise=os.path.join(OUTPUT_DIR,"rule_MERGE_MONOMERS_TO_MULTIMERS","{multi}_data.json")
        ),
        mutation_list = MUTATION_DF_PATH if MUTATION_DF_PATH is not None else []
    output:
        directory(os.path.join(OUTPUT_DIR, "rule_MUTATE", "{multi}")) if MUTATION_DF_PATH is not None else []
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_MUTATE", "{multi}.log")
    resources:
        mem_mb      = 16000,
        runtime     = 480,
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_MUTATE", "{multi}.tsv"),
    params:
        output_dir = OUTPUT_DIR 
    conda: 
        "../envs/preprocessing.yaml"
    shell:
        """
        python {input._helper} {input.data} {input.mutation_list} {params.output_dir}/rule_MUTATE/{wildcards.multi}
        """


# ─────────────────────────────────────────────────────────────────────────────
# Per-multimer aggregator rules for the mutation stream.
#
# Motivation
# ----------
# Historically, the mutation branch of _collect_inference_targets() in
# common.smk expanded every mutated multimer's downstream targets in a Python
# `for` loop that called `checkpoints.MUTATE.get(multi=multi)` inline.  Because
# `rule all` (and `AGGREGATE_RESULTS`) is a single job with a single wildcards
# object, the loop's first `checkpoints.MUTATE.get()` call raised
# IncompleteCheckpointException, aborting the whole input function; Snakemake
# then re-invoked the input function only after that specific MUTATE instance
# finished — so DP + MERGE for multimer k+1 could not begin until MUTATE for
# multimer k had completed.  That is what produced the "AF3_DATA_SPEEDY_PIPELINE
# runs serially per multimer" behavior under mutations.
#
# Fix
# ----
# Move the `checkpoints.MUTATE.get(multi=wildcards.multi)` call out of the
# `rule all` input function and into these per-multimer aggregator rules,
# each of which carries its own `{multi}` wildcard.  Because each aggregator
# instance has a different wildcards object, each checkpoint call is
# independent — Snakemake schedules every MUTATE(multi=k) concurrently, and
# every chain's DP job across all multimers becomes visible in the DAG at
# the same time.  The rules themselves are trivial: they only assemble file
# lists and touch a sentinel marker so that `rule all` and
# `AGGREGATE_RESULTS` can request work on a per-multimer basis.
# ─────────────────────────────────────────────────────────────────────────────

def _mutant_inference_targets(wildcards):
    """Return every AF3_INFERENCE output (model.cif + ipsae txt files) for
    a single mutated multimer.

    Invoking `checkpoints.MUTATE.get(multi=wildcards.multi)` inside this
    per-`{multi}` input function is what unlocks parallel DAG expansion:
    each aggregator instance blocks only on its own MUTATE checkpoint.
    """
    mutate_dir = checkpoints.MUTATE.get(multi=wildcards.multi).output[0]
    muts, = glob_wildcards(os.path.join(mutate_dir, "{mut}.json"))
    if not muts:
        return []
    versioned = AF3_VERSION not in ["v3.0.0", "v3.0.1"]
    paths = []
    for mut in muts:
        m = re.search(r'seed-(\d+)', mut)
        if m is None:
            continue
        seed = m.group(1)
        for sample in range(N_SAMPLES):
            base_dir = os.path.join(
                OUTPUT_DIR, "rule_AF3_INFERENCE", mut,
                f"seed-{seed}_sample-{sample}",
            )
            if versioned:
                paths.append(os.path.join(base_dir, f"{mut}_seed-{seed}_sample-{sample}_model.cif"))
                paths.append(os.path.join(base_dir, f"{mut}_seed-{seed}_sample-{sample}_model_10_15.txt"))
                paths.append(os.path.join(base_dir, f"{mut}_seed-{seed}_sample-{sample}_model_15_15.txt"))
            else:
                paths.append(os.path.join(base_dir, "model.cif"))
                paths.append(os.path.join(base_dir, "model_10_15.txt"))
                paths.append(os.path.join(base_dir, "model_15_15.txt"))
    return paths


def _mutant_extract_scores_targets(wildcards):
    """Return every EXTRACT_SCORES per-sample TSV for a single mutated
    multimer. Mirrors `_mutant_inference_targets` but resolves to the
    aggregate-scores paths consumed by AGGREGATE_RESULTS."""
    mutate_dir = checkpoints.MUTATE.get(multi=wildcards.multi).output[0]
    muts, = glob_wildcards(os.path.join(mutate_dir, "{mut}.json"))
    if not muts:
        return []
    paths = []
    for mut in muts:
        m = re.search(r'seed-(\d+)', mut)
        if m is None:
            continue
        seed = m.group(1)
        for sample in range(N_SAMPLES):
            stem = f"{mut}_seed-{seed}_sample-{sample}"
            paths.append(os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES", mut, f"{stem}_af_global.tsv"))
            paths.append(os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES", mut, f"{stem}_af_per_chain.tsv"))
            paths.append(os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES", mut, f"{stem}_af_per_chain_pair.tsv"))
            paths.append(os.path.join(OUTPUT_DIR, "rule_EXTRACT_SCORES", mut, f"{stem}_ipsae.tsv"))
    return paths


rule MUTANT_INFERENCE_SET:
    """Per-mutated-multimer sentinel: fan out to every AF3_INFERENCE output
    for chains derived from `{multi}`.  Consumed by `rule all` in place of
    the previous inline checkpoint expansion.  Local rule — only touches
    a marker file.

    NOTE: sentinel is created via an explicit `shell` action rather than
    Snakemake's `touch()` output flag.  On Snakemake 9.x, using `touch()`
    for a rule defined in an included .smk file when wildcard_constraints
    are declared at the Snakefile level silently drops the touch action,
    causing MissingOutputException with "parent dir not present" (upstream
    issue snakemake/snakemake#3645).  The explicit `mkdir + touch` shell
    bypasses that interaction and is version-agnostic."""
    input:
        _mutant_inference_targets
    output:
        os.path.join(OUTPUT_DIR, "rule_AF3_MUTANT_SET_DONE", "{multi}.inference.done")
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_AF3_MUTANT_SET_DONE", "{multi}.inference.log")
    resources:
        mem_mb  = 100,
        runtime = 1,
    shell:
        "mkdir -p $(dirname {output}) $(dirname {log}) && touch {output} 2> {log}"


rule MUTANT_EXTRACT_SCORES_SET:
    """Per-mutated-multimer sentinel: fan out to every EXTRACT_SCORES output
    for chains derived from `{multi}`.  Consumed by AGGREGATE_RESULTS in
    place of the previous inline checkpoint expansion in
    `aggregate_outputs()`.  Local rule — only touches a marker file.

    See NOTE on MUTANT_INFERENCE_SET re: shell vs. touch() and upstream
    issue snakemake/snakemake#3645."""
    input:
        _mutant_extract_scores_targets
    output:
        os.path.join(OUTPUT_DIR, "rule_AF3_MUTANT_SET_DONE", "{multi}.scores.done")
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_AF3_MUTANT_SET_DONE", "{multi}.scores.log")
    resources:
        mem_mb  = 100,
        runtime = 1,
    shell:
        "mkdir -p $(dirname {output}) $(dirname {log}) && touch {output} 2> {log}"
