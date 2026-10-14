import os
import sys
import re
import json
import logging
import argparse
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

#  Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("run_models.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# IMPORTS FROM st2_retrieval_generation
# Import shared infrastructure
try:
    from st2_retrieval_generation import (
        parse_permit_chunks,
        load_embedder_and_embeddings,
        retrieve_hybrid_rerank_chimney_aware,
        build_event_prompt,
        build_period_prompt,
        check_citations,
        citation_diversity_summary,
        SYSTEM_ROLE,
        EMBED_MODEL_LARGE,
        CROSS_ENCODER_MODEL,
        RERANK_CANDIDATE_POOL,
        LLM_BASE_URL,
        LLM_MAX_TOKENS,
        PermitChunk,
    )
    log.info("Imports from st2_retrieval_generation.py: OK")
except ImportError as e:
    log.error(f"Could not import from st2_retrieval_generation.py: {e}")
    log.error("Ensure st2_retrieval_generation.py is in the same directory.")
    sys.exit(1)

# CONFIG

PERMITS_PATH       = "./Data/env_permits.txt"
EVENTS_PARQUET     = "./Processed/aggregated_events.parquet"
PARAGRAPHS_PARQUET = "./Processed/event_paragraphs.parquet"
BASE_OUTPUT_DIR    = "./Generation_Output/ModelComparison"

DATE_FROM = "2025-03-01"
DATE_TO   = "2025-03-31"
CHIMNEY   = None   # None = both KGF1 and KGF2

LLM_API_KEY = open("./Data/API.txt").readline().strip()

# VariantD fixed configuration
VARIANT_NAME      = "VariantD"
TOP_K             = 8
RETRIEVAL_MODE    = "hybrid_rerank_chimney"
EMBED_MODEL       = EMBED_MODEL_LARGE
CANDIDATE_POOL    = RERANK_CANDIDATE_POOL * 2   # larger pool for chimney filtering

# gpt-5.1 is excluded: its results are taken from the VariantD retrieval
# evaluation run (temperature=0, identical configuration). Re-running it
# here would produce stochastic duplicates that cannot be compared cleanly.
MODELS = [
    "gpt-5-mini",
    "gpt-5-nano",
]

# LLM CALL

def call_llm_model(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Call the LLM with an explicit model parameter."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# MODEL RUNNER

def run_model(
    model: str,
    retrieved_per_event: list,
    merged: pd.DataFrame,
    permit_chunks: list,
    llm_client: OpenAI,
    period_slug: str,
    skip_done: bool = False,
) -> dict:
    """
    Generate reports for one model using pre-retrieved VariantD contexts.
    Returns summary dict.
    """
    #  Output directory 
    model_slug = re.sub(r"[^\w\-]", "-", model)   # gpt-5.1 → gpt-5-1
    output_dir = Path(BASE_OUTPUT_DIR) / model_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    events_path = output_dir / f"event_reports_{period_slug}.json"

    #  Skip if already done 
    if skip_done and events_path.exists():
        log.info(f"[{model}] Output already exists — skipping (--skip-done set).")
        with open(events_path, encoding="utf-8") as f:
            event_reports = json.load(f)
        return {
            "model":   model,
            "status":  "skipped",
            "n_events": len(event_reports),
        }

    log.info(f"\n{'='*70}")
    log.info(f"MODEL: {model}")
    log.info(f"Variant: {VARIANT_NAME} | k={TOP_K} | embedder={EMBED_MODEL}")
    log.info(f"Output → {output_dir}")
    log.info(f"{'='*70}")

    #  Generate per-event reports 
    log.info(f"[{model}] Generating {len(merged)} event reports...")
    event_reports = []
    for i, (_, row) in enumerate(merged.iterrows()):
        hits        = retrieved_per_event[i]
        user_prompt = build_event_prompt(row, hits, TOP_K)
        try:
            llm_report = call_llm_model(
                llm_client, model, SYSTEM_ROLE, user_prompt
            )
        except Exception as e:
            log.warning(f"[{model}] Event {i+1} generation failed: {e}")
            llm_report = f"[GENERATION FAILED: {e}]"

        event_reports.append({
            "event_idx":          int(row["event_idx"]),
            "chimney":            str(row["chimney"]),
            "event_datetime":     str(row["event_datetime"]),
            "user_input":         user_prompt,
            "retrieved_contexts": [h["content"]         for h in hits],
            "llm_report":         llm_report,
            "retrieved_ids":      [h["voorschrift_id"]  for h in hits],
            "retrieval_scores":   [round(float(h["score"]), 4) for h in hits],
        })

        if (i + 1) % 10 == 0:
            log.info(f"[{model}] Event reports: {i+1}/{len(merged)}")

            #  Checkpoint every 10 events 
            ckpt_path = output_dir / f"checkpoint_{period_slug}.json"
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(event_reports, f, ensure_ascii=False, indent=2,
                          default=str)

    log.info(f"[{model}] All {len(event_reports)} event reports generated.")

    #  Generate period report 
    log.info(f"[{model}] Generating period report...")
    try:
        period_prompt  = build_period_prompt(
            event_reports, merged, DATE_FROM, DATE_TO
        )
        period_report  = call_llm_model(
            llm_client, model, SYSTEM_ROLE, period_prompt
        )
    except Exception as e:
        log.warning(f"[{model}] Period report failed: {e}")
        period_report = f"[PERIOD REPORT GENERATION FAILED: {e}]"
    log.info(f"[{model}] Period report done.")

    #  Citation check 
    log.info(f"[{model}] Running citation validity check...")
    citation_check = check_citations(event_reports, period_report, permit_chunks)
    diversity      = citation_diversity_summary(citation_check, len(event_reports))
    log.info(
        f"[{model}] Hallucination rate: "
        f"{citation_check['hallucination_rate']:.1%}"
    )

    #  Save outputs 
    log.info(f"[{model}] Saving outputs...")

    with open(events_path, "w", encoding="utf-8") as f:
        json.dump(event_reports, f, ensure_ascii=False, indent=2, default=str)

    period_path = output_dir / f"period_report_{period_slug}.txt"
    with open(period_path, "w", encoding="utf-8") as f:
        f.write(f"Period  : {DATE_FROM} → {DATE_TO}\n")
        f.write(f"Variant : {VARIANT_NAME}\n")
        f.write(f"Model   : {model}\n")
        f.write(f"Embedder: {EMBED_MODEL}\n")
        f.write(f"Top-k   : {TOP_K}\n\n")
        f.write(period_report)

    citation_path = output_dir / f"citation_check_{period_slug}.json"
    with open(citation_path, "w", encoding="utf-8") as f:
        json.dump(
            {**citation_check, "diversity": diversity},
            f, ensure_ascii=False, indent=2,
        )

    summary_rows = []
    for er, cc in zip(event_reports, citation_check["events"]):
        row_data = merged[merged["event_idx"] == er["event_idx"]]
        row      = row_data.iloc[0] if not row_data.empty else pd.Series()
        summary_rows.append({
            "event_idx":        er["event_idx"],
            "chimney":          er["chimney"],
            "event_datetime":   er["event_datetime"],
            "aq_composite_z":   row.get("aq_composite_z", None),
            "retrieved_ids":    " | ".join(er["retrieved_ids"]),
            "retrieval_scores": " | ".join(str(s) for s in er["retrieval_scores"]),
            "hallucination":    cc["hallucination"],
            "fabricated_ids":   " | ".join(cc["fabricated_ids"]),
            "not_retrieved":    " | ".join(cc["not_retrieved"]),
        })
    pd.DataFrame(summary_rows).to_csv(
        output_dir / f"summary_table_{period_slug}.csv",
        index=False, encoding="utf-8",
    )

    # Remove checkpoint now that we have the full output
    ckpt_path = output_dir / f"checkpoint_{period_slug}.json"
    if ckpt_path.exists():
        ckpt_path.unlink()

    log.info(f"[{model}] Outputs saved → {output_dir}")

    return {
        "model":             model,
        "model_slug":        model_slug,
        "status":            "done",
        "n_events":          len(event_reports),
        "hallucination_rate": citation_check["hallucination_rate"],
        "diversity":         diversity,
    }


# MAIN

def main():
    parser = argparse.ArgumentParser(
        description="Model comparison generation — VariantD only"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"Run one model only. Options: {MODELS}",
    )
    parser.add_argument(
        "--skip-done", action="store_true",
        help="Skip models whose event_reports output already exists.",
    )
    args = parser.parse_args()

    models_to_run = MODELS
    if args.model:
        if args.model not in MODELS:
            log.error(f"Unknown model '{args.model}'. Valid: {MODELS}")
            sys.exit(1)
        models_to_run = [args.model]
        log.info(f"--model flag set: running {args.model} only.")

    period_slug = f"{DATE_FROM}_to_{DATE_TO}" + (f"_{CHIMNEY}" if CHIMNEY else "")

    log.info("=" * 70)
    log.info("Model Comparison — VariantD")
    log.info(f"Period  : {DATE_FROM} → {DATE_TO} | Chimney: {CHIMNEY or 'both'}")
    log.info(f"Variant : {VARIANT_NAME} | k={TOP_K} | embedder={EMBED_MODEL}")
    log.info(f"Models  : {models_to_run}")
    log.info("=" * 70)

    #  Load permit corpus 
    log.info("Loading permit corpus...")
    permit_chunks = parse_permit_chunks(PERMITS_PATH)
    log.info(f"  {len(permit_chunks)} active permit chunks loaded.")

    #  Load event data 
    log.info("Loading event data...")
    events_df     = pd.read_parquet(EVENTS_PARQUET)
    paragraphs_df = pd.read_parquet(PARAGRAPHS_PARQUET)

    events_df["event_datetime"] = pd.to_datetime(events_df["event_datetime"])
    paragraphs_df["datetime"]   = pd.to_datetime(paragraphs_df["datetime"])

    mask = (
        (events_df["event_datetime"] >= DATE_FROM) &
        (events_df["event_datetime"] <= DATE_TO + " 23:59:59")
    )
    if CHIMNEY:
        mask &= (events_df["chimney"] == CHIMNEY)

    events_filtered     = events_df[mask].reset_index(drop=True)
    para_mask           = paragraphs_df["event_idx"].isin(
        events_filtered["event_idx"]
    )
    paragraphs_filtered = paragraphs_df[para_mask].reset_index(drop=True)

    merge_cols = ["event_idx", "paragraph", "structured_json"]
    if "retrieval_query" in paragraphs_filtered.columns:
        merge_cols.append("retrieval_query")
        log.info("  retrieval_query field found — using clean paragraphs for retrieval.")
    else:
        log.warning(
            "  retrieval_query not found — falling back to full paragraph. "
            "Re-run Stage 1 Cells 21-22 to generate clean retrieval queries."
        )

    merged = events_filtered.merge(
        paragraphs_filtered[merge_cols],
        on="event_idx", how="left",
    )
    log.info(f"  {len(merged)} events in period.")

    #  Shared retrieval infrastructure (built once) 
    log.info("Initialising LLM client...")
    llm_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    log.info("Building BM25 index...")
    from rank_bm25 import BM25Okapi
    bm25_corpus = [c.content.lower().split() for c in permit_chunks]
    bm25_index  = BM25Okapi(bm25_corpus)
    log.info(f"  BM25 index built over {len(bm25_corpus)} chunks.")

    log.info(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    log.info("  Cross-encoder loaded.")

    log.info(f"Loading embedder: {EMBED_MODEL}")
    embedder, permit_embeddings = load_embedder_and_embeddings(
        EMBED_MODEL, permit_chunks
    )
    log.info("  Embedder loaded.")

    #  Retrieve ONCE for VariantD — reused for all models 
    log.info(f"Retrieving VariantD contexts for {len(merged)} events (once)...")
    retrieved_per_event = []
    for _, row in merged.iterrows():
        para = str(row.get("retrieval_query", row.get("paragraph", "")))
        hits = retrieve_hybrid_rerank_chimney_aware(
            para, str(row["chimney"]),
            embedder, permit_embeddings, permit_chunks,
            bm25_index, cross_encoder, TOP_K,
            CANDIDATE_POOL,
        )
        retrieved_per_event.append(hits)
    log.info("  Retrieval complete — contexts will be reused for all models.")

    #  Run generation for each model 
    all_results = []
    import shutil


    GPT51_SRC = Path("./Generation_Output/VariantD")
    GPT51_DST = Path("./Generation_Output/ModelComparison/gpt-5-1")
    GPT51_DST.mkdir(parents=True, exist_ok=True)

    for fname in [
        f"event_reports_{DATE_FROM}_to_{DATE_TO}.json",
        f"period_report_{DATE_FROM}_to_{DATE_TO}.txt",
    ]:
        src_path = GPT51_SRC / fname
        dst_path = GPT51_DST / fname
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            log.info(f"  Copied {fname} → ModelComparison/gpt-5-1/")
        else:
            log.warning(f"  VariantD source not found: {src_path} — run generation first")
    for model in models_to_run:
        try:
            result = run_model(
                model             = model,
                retrieved_per_event = retrieved_per_event,
                merged            = merged,
                permit_chunks     = permit_chunks,
                llm_client        = llm_client,
                period_slug       = period_slug,
                skip_done         = args.skip_done,
            )
            all_results.append(result)
        except Exception as e:
            log.error(f"[{model}] FAILED: {e}")
            log.error(traceback.format_exc())
            all_results.append({"model": model, "status": "failed", "error": str(e)})

    # Cross-model summary
    log.info("\n" + "=" * 70)
    log.info("MODEL COMPARISON SUMMARY")
    log.info("=" * 70)
    log.info(f"  {'Model':<14} {'Status':<10} {'Halluc%':<10} {'UniqueIDs':<12} {'TotalCitations'}")
    log.info(f"  {'-'*60}")
    for r in all_results:
        if r.get("status") in ("done", "skipped"):
            d = r.get("diversity", {})
            log.info(
                f"  {r['model']:<14} {r['status']:<10} "
                f"{r.get('hallucination_rate', float('nan')):<10.1%} "
                f"{d.get('unique_ids_cited', 'N/A'):<12} "
                f"{d.get('total_citations', 'N/A')}"
            )
        else:
            log.info(f"  {r['model']:<14} FAILED — {r.get('error', '')}")

    summary_path = Path(BASE_OUTPUT_DIR) / "model_comparison_summary.json"
    Path(BASE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"\nSummary saved → {summary_path}")
    log.info(
        "Run run_ragas_only.py and run_llm_judge.py next, pointing "
        "BASE_OUTPUT_DIR to ./Generation_Output/ModelComparison/{model_slug}/ "
        "for each model."
    )


if __name__ == "__main__":
    main()
