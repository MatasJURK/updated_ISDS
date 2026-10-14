import os
import sys
import re
import json
import math
import asyncio
import hashlib
import logging
import argparse
import traceback
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("run_ragas_models.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# CONFIG

OUTPUT_DIR   = "./Generation_Output/ModelComparison"
DATE_FROM    = "2025-03-01"
DATE_TO      = "2025-03-31"
CHIMNEY      = None   # None = both KGF1 and KGF2

LLM_BASE_URL = "https://llmproxy.uva.nl/v1"
LLM_API_KEY  = open("./Data/API.txt").readline().strip()
LLM_MODEL    = "gpt-oss-120b"   # evaluator model free

RAGAS_MAX_TOKENS    = 16384
RAGAS_SLEEP_SEC     = 3
RAGAS_MAX_RETRIES   = 1
CHECKPOINT_INTERVAL = 10

TRANSLATE_TO_ENGLISH = False # Tested - no difference

MODELS = [
    ("gpt-5-1",    8),   # gpt-5.1 output (folder named gpt-5-1)
    ("gpt-5-mini", 8),
    ("gpt-5-nano", 8),
]

# HELPERS

def json_serialisable(obj):
    if hasattr(obj, "item"):
        val = obj.item()
        if isinstance(val, float) and math.isnan(val):
            return None
        return val
    if isinstance(obj, float) and math.isnan(obj):
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def extract_paragraph_from_prompt(user_input: str) -> str:
    """Extract the natural-language event paragraph from the saved prompt."""
    import re
    m = re.search(
        r'---\s*Natural language summary\s*---\s*(.*?)\s*---\s*Structured data',
        user_input, re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def extract_sections_34(report: str) -> str:
    """Extract sections 3 and 4 from a compliance mini-report."""
    import re
    boundaries = []
    for n in range(1, 6):
        m = re.search(rf'(?m)^{n}\.\s', report)
        if m:
            boundaries.append((n, m.start()))
    if len(boundaries) < 5:
        return report
    boundaries.sort(key=lambda x: x[1])
    sections: dict = {}
    for i, (num, start) in enumerate(boundaries):
        end = boundaries[i + 1][1] if i + 1 < len(boundaries) else len(report)
        sections[num] = report[start:end].strip()
    if 3 not in sections or 4 not in sections:
        return report
    return sections[3] + "\n\n" + sections[4]


def _checkpoint_path(output_dir: Path, period_slug: str) -> Path:
    return output_dir / f"ragas_checkpoint_{period_slug}.json"


def _load_checkpoint(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_checkpoint(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=json_serialisable)


# BATCHED METRIC SCORING WITH CHECKPOINT/RESUME

async def score_metric_batched(
    metric, samples, metric_name, sleep_sec, max_retries,
    checkpoint_path, existing_scores=None,
):
    """Score all samples for one metric, saving checkpoint every N events."""
    import asyncio

    n = len(samples)
    scores = list(existing_scores) if existing_scores and len(existing_scores) == n \
             else [float("nan")] * n

    # Find first incomplete index
    start_idx = next(
        (i for i, s in enumerate(scores) if math.isnan(s)), n
    )
    if start_idx == n:
        log.info(f"    {metric_name}: all {n} scores already in checkpoint.")
        return scores

    log.info(
        f"    {metric_name}: scoring {n - start_idx} remaining samples "
        f"(resuming from {start_idx})..."
    )

    for i in range(start_idx, n):
        for attempt in range(max_retries + 1):
            try:
                result = await metric.single_turn_ascore(samples[i])
                scores[i] = float(result) if result is not None else float("nan")
                break
            except Exception as e:
                if attempt < max_retries:
                    await asyncio.sleep(sleep_sec)
                else:
                    log.warning(
                        f"    {metric_name} event {i}: failed after "
                        f"{max_retries + 1} attempts — {e}"
                    )
                    scores[i] = float("nan")

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            cp = _load_checkpoint(checkpoint_path)
            cp[metric_name] = scores
            _save_checkpoint(checkpoint_path, cp)
            log.info(
                f"    {metric_name}: checkpoint saved at event {i + 1}/{n}"
            )

        await asyncio.sleep(sleep_sec)

    # Final checkpoint save
    cp = _load_checkpoint(checkpoint_path)
    cp[metric_name] = scores
    _save_checkpoint(checkpoint_path, cp)

    nan_count = sum(1 for s in scores if math.isnan(s))
    valid     = [s for s in scores if not math.isnan(s)]
    mean_val  = sum(valid) / len(valid) if valid else float("nan")
    log.info(
        f"    {metric_name}: mean={mean_val:.4f} | nan={nan_count}/{n}"
    )
    return scores


# RAGAS EVALUATION FOR ONE MODEL

async def run_ragas_for_model(
    model_slug: str,
    top_k: int,
    event_reports_path: Path,
    output_dir: Path,
):
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithoutReference,
    )

    period_slug = f"{DATE_FROM}_to_{DATE_TO}" + (f"_{CHIMNEY}" if CHIMNEY else "")

    log.info(f"\n{'='*70}")
    log.info(f"RAGAS — {model_slug} (top_k={top_k})")
    log.info(f"Reading: {event_reports_path}")
    log.info(f"{'='*70}")

    # Load event reports 
    with open(event_reports_path, encoding="utf-8") as f:
        event_reports = json.load(f)
    log.info(f"  Loaded {len(event_reports)} event reports.")

    # Filter out GENERATION FAILED entries
    failed = [
        er for er in event_reports
        if str(er.get("llm_report", "")).startswith("[GENERATION FAILED")
    ]
    if failed:
        log.warning(
            f"  {len(failed)} events have GENERATION FAILED reports — "
            f"these will produce NaN scores."
        )

    # Verify required fields
    missing = [
        er["event_idx"] for er in event_reports
        if "retrieved_contexts" not in er or "llm_report" not in er
    ]
    if missing:
        log.error(f"  {len(missing)} events missing required fields — aborting.")
        return None

    # Use actual k from data
    actual_k = len(event_reports[0]["retrieved_contexts"])
    if actual_k != top_k:
        log.warning(
            f"  Expected top_k={top_k} but reports contain {actual_k} contexts "
            f"— using actual value."
        )
        top_k = actual_k

    # Initialise evaluator
    log.info("  Initialising evaluator LLM and embeddings...")
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=0,
            max_tokens=RAGAS_MAX_TOKENS,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-large",
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )
    )

    faithfulness = Faithfulness(llm=evaluator_llm)
    response_rel = ResponseRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        strictness=1,
    )
    context_prec = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

    # Build samples
    def _make_sample(er, response=None, inject_paragraph=False):
        paragraph = extract_paragraph_from_prompt(er.get("user_input", ""))
        contexts  = list(er["retrieved_contexts"])
        if inject_paragraph and paragraph:
            contexts = [f"[Event evidence]\n{paragraph}"] + contexts
        return SingleTurnSample(
            user_input=paragraph or (
                f"Compliance assessment for {er['chimney']} "
                f"at {er['event_datetime']}"
            ),
            response=response if response is not None else er["llm_report"],
            retrieved_contexts=contexts,
        )

    samples_faith_legal   = [
        _make_sample(er, extract_sections_34(er["llm_report"]), inject_paragraph=False)
        for er in event_reports
    ]
    samples_faith_factual = [
        _make_sample(er, inject_paragraph=True) for er in event_reports
    ]
    samples_full = [_make_sample(er) for er in event_reports]
    log.info(f"  {len(samples_full)} samples built.")

    # Score metrics with checkpoint/resume
    cp_path = _checkpoint_path(output_dir, period_slug)
    cp      = _load_checkpoint(cp_path)
    if cp:
        log.info(
            f"  Checkpoint found — resuming "
            f"(completed: {[k for k in cp if not k.startswith('_')]})"
        )

    # Check if final result already exists
    ragas_path = output_dir / f"ragas_results_{period_slug}.json"
    if ragas_path.exists():
        log.info(
            f"  Final result already exists — skipping. "
            f"Delete {ragas_path.name} to force rerun."
        )
        with open(ragas_path, encoding="utf-8") as f:
            return json.load(f)["aggregate"]

    log.info("  Scoring faithfulness_legal (s3-4 vs permit chunks)...")
    faith_legal_scores = await score_metric_batched(
        faithfulness, samples_faith_legal, "faithfulness_legal",
        RAGAS_SLEEP_SEC, RAGAS_MAX_RETRIES,
        checkpoint_path=cp_path,
        existing_scores=cp.get("faithfulness_legal"),
    )

    log.info("  Scoring faithfulness_factual (full report + event paragraph)...")
    faith_factual_scores = await score_metric_batched(
        faithfulness, samples_faith_factual, "faithfulness_factual",
        RAGAS_SLEEP_SEC, RAGAS_MAX_RETRIES,
        checkpoint_path=cp_path,
        existing_scores=cp.get("faithfulness_factual"),
    )

    log.info("  Scoring response_relevancy...")
    relevancy_scores = await score_metric_batched(
        response_rel, samples_full, "response_relevancy",
        RAGAS_SLEEP_SEC, RAGAS_MAX_RETRIES,
        checkpoint_path=cp_path,
        existing_scores=cp.get("response_relevancy"),
    )

    log.info("  Scoring context_precision...")
    precision_scores = await score_metric_batched(
        context_prec, samples_full, "context_precision",
        RAGAS_SLEEP_SEC, RAGAS_MAX_RETRIES,
        checkpoint_path=cp_path,
        existing_scores=cp.get("context_precision"),
    )

    # aggregate
    def safe_mean(scores):
        valid = [s for s in scores if not math.isnan(s)]
        return sum(valid) / len(valid) if valid else float("nan")

    def safe_nan_count(scores):
        return int(sum(1 for s in scores if math.isnan(s)))

    ragas_output = {
        "meta": {
            "period":           f"{DATE_FROM} to {DATE_TO}",
            "model":            model_slug,
            "evaluator_model":  LLM_MODEL,
            "top_k":            int(top_k),
            "n_samples":        int(len(samples_full)),
            "n_failed_reports": int(len(failed)),
            "translate":        TRANSLATE_TO_ENGLISH,
            "note": (
                "Dual faithfulness: faithfulness_legal (s3-4 vs permit chunks) "
                "and faithfulness_factual (full report vs paragraph+permits). "
                "user_input is the full event paragraph. "
                "Evaluator: gpt-oss-120b. Raw Dutch evaluated (TRANSLATE=False)."
            ),
        },
        "aggregate": {
            "faithfulness_legal":                       safe_mean(faith_legal_scores),
            "faithfulness_legal_nan_count":             safe_nan_count(faith_legal_scores),
            "faithfulness_factual":                     safe_mean(faith_factual_scores),
            "faithfulness_factual_nan_count":           safe_nan_count(faith_factual_scores),
            "answer_relevancy":                         safe_mean(relevancy_scores),
            "relevancy_nan_count":                      safe_nan_count(relevancy_scores),
            "llm_context_precision_without_reference":  safe_mean(precision_scores),
            "precision_nan_count":                      safe_nan_count(precision_scores),
        },
        "per_event": [
            {
                "event_idx":      int(er["event_idx"]),
                "chimney":        str(er["chimney"]),
                "event_datetime": str(er["event_datetime"]),
                "faithfulness_legal":  (
                    None if math.isnan(faith_legal_scores[i])
                    else float(faith_legal_scores[i])
                ),
                "faithfulness_factual": (
                    None if math.isnan(faith_factual_scores[i])
                    else float(faith_factual_scores[i])
                ),
                "answer_relevancy": (
                    None if math.isnan(relevancy_scores[i])
                    else float(relevancy_scores[i])
                ),
                "llm_context_precision_without_reference": (
                    None if math.isnan(precision_scores[i])
                    else float(precision_scores[i])
                ),
            }
            for i, er in enumerate(event_reports)
        ],
    }

    # Save
    with open(ragas_path, "w", encoding="utf-8") as f:
        json.dump(ragas_output, f, ensure_ascii=False, indent=2,
                  default=json_serialisable)
    log.info(f"  Saved → {ragas_path}")

    # Remove checkpoint
    if cp_path.exists():
        cp_path.unlink()
        log.info("  Checkpoint removed.")

    agg = ragas_output["aggregate"]
    log.info(
        f"  faith_legal={agg['faithfulness_legal']:.4f} | "
        f"faith_factual={agg['faithfulness_factual']:.4f} | "
        f"rel={agg['answer_relevancy']:.4f} | "
        f"prec={agg['llm_context_precision_without_reference']:.4f}"
    )
    return agg


# MAIN

async def main():
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation for ModelComparison outputs"
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help=f"ModelComparison directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Run one model slug only (e.g. gpt-5-1, gpt-5-mini, gpt-5-nano)",
    )
    args = parser.parse_args()

    base_dir    = Path(args.output_dir)
    period_slug = f"{DATE_FROM}_to_{DATE_TO}" + (f"_{CHIMNEY}" if CHIMNEY else "")

    models_to_run = MODELS
    if args.model:
        match = [m for m in MODELS if m[0] == args.model]
        if not match:
            log.error(f"Unknown model '{args.model}'. Valid: {[m[0] for m in MODELS]}")
            sys.exit(1)
        models_to_run = match

    log.info("=" * 70)
    log.info("RAGAS Model Comparison Evaluation")
    log.info(f"Period    : {DATE_FROM} to {DATE_TO} | Chimney: {CHIMNEY or 'both'}")
    log.info(f"Output dir: {base_dir}")
    log.info(f"Models    : {[m[0] for m in models_to_run]}")
    log.info(f"Evaluator : {LLM_MODEL}")
    log.info("=" * 70)

    all_results = []

    for model_slug, top_k in models_to_run:
        output_dir         = base_dir / model_slug
        event_reports_path = output_dir / f"event_reports_{period_slug}.json"

        if not event_reports_path.exists():
            log.error(
                f"[{model_slug}] event_reports not found: {event_reports_path}"
            )
            all_results.append({"model": model_slug, "error": "event_reports not found"})
            continue

        try:
            result = await run_ragas_for_model(
                model_slug         = model_slug,
                top_k              = top_k,
                event_reports_path = event_reports_path,
                output_dir         = output_dir,
            )
            if result is not None:
                all_results.append({"model": model_slug, "ragas": result})
            else:
                all_results.append({"model": model_slug, "error": "returned None"})
        except Exception as e:
            log.error(f"[{model_slug}] FAILED: {e}\n{traceback.format_exc()}")
            all_results.append({"model": model_slug, "error": str(e)})

    # Summary
    log.info("\n" + "=" * 70)
    log.info("FINAL MODEL COMPARISON SUMMARY")
    log.info("=" * 70)
    log.info(
        f"  {'Model':<14} {'faith_legal':>12} {'faith_fact':>12} "
        f"{'rel':>8} {'prec':>8}"
    )
    log.info(f"  {'-'*58}")
    for r in all_results:
        if "ragas" in r:
            s = r["ragas"]
            log.info(
                f"  {r['model']:<14} "
                f"{s['faithfulness_legal']:>12.4f} "
                f"{s['faithfulness_factual']:>12.4f} "
                f"{s['answer_relevancy']:>8.4f} "
                f"{s['llm_context_precision_without_reference']:>8.4f}"
            )
        else:
            log.info(f"  {r['model']:<14} FAILED — {r.get('error', '')}")

    summary_path = base_dir / "ragas_model_comparison.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2,
                  default=json_serialisable)
    log.info(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
