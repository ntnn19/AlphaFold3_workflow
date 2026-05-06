# Snakefile
# Generate AlphaFold3 reports from a samplesheet.
#
# Samplesheet TSV (header required):
# sample_id    af3_dir
# S1           /path/to/af3/output/S1_jobdir
# S2           /path/to/af3/output/S2_jobdir
#
# Usage:
# snakemake -j 20 --config samplesheet=samples.tsv script=workflow/scripts/af3_report.py
#
# Note: This assumes af3_report.py produces:
#   report.html, predictions.tsv, chains.tsv, chain_pairs.tsv
# and writes plots under the output directory.

import pandas as pd
from pathlib import Path

def get_targets(wc):
    d = Path(AF3_DIR[wc.sample_id])
    print("USALIGN sample_id:",wc.sample_id)
    print("USALIGN AF3 dir:",d)
    print("Exists:",d.exists())
    print("CIFs:",list(d.rglob("*.cif")))
    return sorted(str(p) for p in d.rglob("*.cif"))


SAMPLESHEET = config.get("samplesheet","samples.tsv")
LAYOUT = config.get("layout","nagarnat")  # "nagarnat" (default) or "dm"
OUTPUT_DIR = config.get("output_dir","results")

df = pd.read_csv(SAMPLESHEET,sep="\t",dtype=str).fillna("")
if not {"sample_id", "af3_dir"}.issubset(df.columns):
    raise ValueError("Samplesheet must be a TSV with columns: sample_id, af3_dir")

SAMPLES = df["sample_id"].tolist()
AF3_DIR = dict(zip(df["sample_id"],df["af3_dir"]))
GROUND_TRUTH = dict(zip(df["sample_id"],df["ground_truth"])) if "ground_truth" in df.columns else {}
WORKFLOW_DIR = os.path.dirname(os.path.abspath(workflow.snakefile))
report: f"{WORKFLOW_DIR}/report/workflow.rst"

print("WORKFLOW_DIR=",WORKFLOW_DIR)
print("SAMPLESHEET=",SAMPLESHEET)
print("DF=",df.head())
print("SAMPLES=",SAMPLES[:10])
print("AF3_DIR=",list(AF3_DIR.items())[:10])
print("GROUND_TRUTH=",list(GROUND_TRUTH.items())[:10])
OUTPUT_TABLES = ["chain_pairs", "chains", "master", "predictions", "sample_status"]
OUTPUT_PLOTS = ["chain_iptm_distribution"]
if GROUND_TRUTH != {}:
    OUTPUT_TABLES = OUTPUT_TABLES + ["usalign"]
    OUTPUT_PLOTS = OUTPUT_PLOTS + ["tm_score_distribution"]

rule all:
    input:
        expand(f"{OUTPUT_DIR}/reports/alphafold3/{{sample_id}}/af3_report.done.txt",sample_id=SAMPLES),
        expand(f"{OUTPUT_DIR}/reports/usalign/{{sample_id}}/usalign_report.done.txt",sample_id=SAMPLES) if GROUND_TRUTH != {} else [], 
        expand(f"{OUTPUT_DIR}/reports/all/all_{{i}}.tsv", i=OUTPUT_TABLES),
        expand(f"{OUTPUT_DIR}/reports/all/plots/{{i}}.html", i=OUTPUT_PLOTS)


rule AF3_REPORT:
    input:
        af3_dir=lambda wc: AF3_DIR[wc.sample_id]
    output:
        html=report(
            directory(f"{OUTPUT_DIR}/reports/alphafold3/{{sample_id}}/plots"),
            patterns=["{name}.html", "{name}.png", ],
            #caption="report/af3_plots.rst",
            category="AlphaFold3",
            subcategory="{sample_id}",
        ),
        pred=report(f"{OUTPUT_DIR}/reports/alphafold3/{{sample_id}}/predictions.tsv",category="AlphaFold3",subcategory="{sample_id}"),
        chains=report(f"{OUTPUT_DIR}/reports/alphafold3/{{sample_id}}/chains.tsv",category="AlphaFold3",subcategory="{sample_id}"),
        pairs=report(f"{OUTPUT_DIR}/reports/alphafold3/{{sample_id}}/chain_pairs.tsv",category="AlphaFold3",subcategory="{sample_id}"),
        molstar=report(
               directory(f"{OUTPUT_DIR}/reports/alphafold3/{{sample_id}}/molstar"),patterns=["{name}.html"],category="AlphaFold3",subcategory="{sample_id}"),
        done_flag=touch(f"{OUTPUT_DIR}/reports/alphafold3/{{sample_id}}/af3_report.done.txt")

    params:
        outdir=lambda wc: f"{OUTPUT_DIR}/reports/alphafold3/{wc.sample_id}",
        layout=LAYOUT,
        raw_data = config["raw_data"]
    shell:
        """
          python {WORKFLOW_DIR}/scripts/af3_report.py {input.af3_dir} \
            -o {params.outdir} \
            --layout {params.layout} --input-tsv {params.raw_data}
          """

rule USALIGN:
    input:
        targets=get_targets,
        ref_list=lambda wc: GROUND_TRUTH[wc.sample_id]
    output:
        done_flag=touch(f"{OUTPUT_DIR}/rule_USALIGN/{{sample_id}}/usalign.done.txt")
    params:
        outdir=lambda wc: f"{OUTPUT_DIR}/rule_USALIGN/{wc.sample_id}"
    shell:
        """
        # Read ground truth paths from the text file, skip blank lines
        while IFS= read -r ref || [ -n "$ref" ]; do
            # Skip empty/whitespace-only lines
            ref=$(echo "$ref" | xargs)
            [ -z "$ref" ] && continue

            # Derive ground truth label from filename (without extension)
            ref_name=$(basename "$ref")
            ref_name="${{ref_name%%.*}}"

            for target in {input.targets}; do
                # Derive prediction sample name from parent directory
                sample=$(basename $(dirname "$target"))

                USalign "$target" "$ref" -ter 0 -mm 1 -outfmt 2 \
                    > {params.outdir}/${{sample}}_ref-${{ref_name}}.usalign.tsv
            done
        done < {input.ref_list}
        """

rule USALIGN_REPORT:
    input:
        done_flg1=f"{OUTPUT_DIR}/rule_USALIGN/{{sample_id}}/usalign.done.txt",
        done_flg2=f"{OUTPUT_DIR}/reports/alphafold3/{{sample_id}}/af3_report.done.txt",
        af3_dir=lambda wc: AF3_DIR[wc.sample_id]
    output:
        html=report(
            directory(f"{OUTPUT_DIR}/reports/usalign/{{sample_id}}/plots"),
            patterns=["{name}.html"],
            #caption="report/af3_plots.rst",
            category="USalign",
            subcategory="{sample_id}",
        ),
        summary=report(f"{OUTPUT_DIR}/reports/usalign/{{sample_id}}/usalign_summary.tsv",category="USalign",subcategory="{sample_id}"),
        done_flag=touch(f"{OUTPUT_DIR}/reports/usalign/{{sample_id}}/usalign_report.done.txt")
    params:
        outdir=lambda wc: f"{OUTPUT_DIR}/reports/usalign/{wc.sample_id}",
        layout=LAYOUT
    shell:
        """
          python {WORKFLOW_DIR}/scripts/usalign_report.py {OUTPUT_DIR}/rule_USALIGN/{wildcards.sample_id} -o {params.outdir} --predictions-tsv {OUTPUT_DIR}/reports/alphafold3/{wildcards.sample_id}/predictions.tsv
          """


rule SUMMARY_REPORT:
    input:
        done_flgs=expand(f"{OUTPUT_DIR}/reports/usalign/{{sample_id}}/usalign_report.done.txt",sample_id=SAMPLES) if GROUND_TRUTH != {} else [],
        af3_dir = expand(f"{OUTPUT_DIR}/reports/alphafold3/{{sample_id}}/af3_report.done.txt",sample_id=SAMPLES)
    output:
        html=report(
            directory(f"{OUTPUT_DIR}/reports/all"),
            patterns=["{name}.tsv"],
            #caption="report/af3_plots.rst",
            category="All",
        ),
        tables = expand(f"{OUTPUT_DIR}/reports/all/all_{{i}}.tsv",i=["chain_pairs", "chains", "master", "predictions", "sample_status", "usalign"]),
    params:
        outdir= f"{OUTPUT_DIR}/reports/all",
        layout=LAYOUT,
        usalign_base=f"--usalign-base {OUTPUT_DIR}/reports/usalign" if GROUND_TRUTH != {} else ""
    shell:
        """
        python {WORKFLOW_DIR}/scripts/all_report.py \
        --report-samples {SAMPLESHEET} \
        --af3-base {OUTPUT_DIR}/reports/alphafold3 \
        {params.usalign_base} \
        -o {params.outdir}
        """

rule SUMMARY_PLOTS:
    input:
        tables = expand(f"{OUTPUT_DIR}/reports/all/all_{{i}}.tsv",i=OUTPUT_TABLES),
    output:
        report_plots=report(
            directory(f"{OUTPUT_DIR}/reports/all/plots"),
            patterns=["{name}.html"],
            #caption="report/af3_plots.rst",
            category="All",
            ),
        actual_plots = expand(f"{OUTPUT_DIR}/reports/all/plots/{{i}}.html", i=OUTPUT_PLOTS)
    params:
        outdir= f"{OUTPUT_DIR}/reports/all",
        layout=LAYOUT,
        tm_plot = f"--tm-plot {OUTPUT_DIR}/reports/all/plots/tm_score_distribution.html" if GROUND_TRUTH != {} else ""
    shell:
        """
        python {WORKFLOW_DIR}/scripts/all_plot.py \
        --pair-tsv {OUTPUT_DIR}/reports/all/all_chain_pairs.tsv \
        --master-tsv {OUTPUT_DIR}/reports/all/all_master.tsv \
        -o {OUTPUT_DIR}/reports/all/plots/chain_iptm_distribution.html \
        {params.tm_plot}
        """

