import os
import sys
import math
import json
import hashlib
import asyncio
import logging
import traceback
import numpy as np
from pathlib import Path

# Windows event loop fix
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("run_ragas_only.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# CONFIGURATION

BASE_OUTPUT_DIR  = "./Generation_Output"
DATE_FROM        = "2025-03-01"
DATE_TO          = "2025-03-31"
CHIMNEY          = None          # "KGF1", "KGF2", or None for both

LLM_BASE_URL     = "https://llmproxy.uva.nl/v1"
LLM_API_KEY      = open("./Data/API.txt").readline().strip()
LLM_MODEL        = "gpt-oss-120b"   # evaluator and translator
EMBED_MODEL      = "intfloat/multilingual-e5-base"

RAGAS_MAX_TOKENS  = 16384
RAGAS_SLEEP_SEC   = 3
RAGAS_MAX_RETRIES  = 1
CHECKPOINT_INTERVAL = 10   # save progress every N events within a metric

TRANSLATE_TO_ENGLISH = False

VARIANTS = [
    ("VariantA", 3),
    ("VariantB", 8),
    ("VariantC", 8),
    ("VariantD", 8),
]

# JSON SERIALISATION HELPER

def json_serialisable(obj):
    if hasattr(obj, "item"):
        val = obj.item()
        if isinstance(val, float) and math.isnan(val):
            return None
        return val
    if isinstance(obj, float) and math.isnan(obj):
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")

# =============================================================================
# TRANSLATION
# Translates Dutch permit text and compliance reports to English.
# Uses an in-memory dict keyed by SHA-256 hash — repeated permit chunks
# (e.g. 1.2.19 appearing in 60%+ of events) are translated only once.
# =============================================================================


def extract_paragraph_from_prompt(user_input: str) -> str:
    """Extract the natural-language event paragraph from the saved prompt string."""
    import re
    m = re.search(
        r'---\s*Natural language summary\s*---\s*(.*?)\s*---\s*Structured data',
        user_input, re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def extract_sections_34(report: str) -> str:
    """
    Extract sections 3 and 4 from a compliance mini-report.
    Uses first-occurrence anchoring to avoid matching numbered lists
    inside section 3 (e.g. "1. Voorschrift 1.2.19 ...").
    """
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


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def translate_text(text: str, client, cache: dict) -> str:
    """
    Translate a single Dutch string to English using gpt-oss-120b.
    Returns cached result if the same text has been seen before.
    """
    key = _hash(text)
    if key in cache:
        return cache[key]

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=4096,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional Dutch-to-English translator "
                        "specialising in environmental regulation and industrial "
                        "permit law. Translate the following text accurately. "
                        "Preserve all permit condition identifiers (e.g. 1.2.19, "
                        "III/2.26, EL.107), numeric values, chemical names, and "
                        "technical terminology exactly as written. "
                        "Output only the translated text with no preamble."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        translated = response.choices[0].message.content.strip()
    except Exception as e:
        log.warning(f"  [translate] Failed for hash {key[:8]}…: {e} — using original")
        translated = text   # fall back to original on failure

    cache[key] = translated
    return translated


def translate_event_reports(event_reports: list, client) -> tuple:
    """
    Translate llm_report and retrieved_contexts for every event.
    Returns (translated_reports, cache_stats_dict).
    Cache is shared across all events so duplicate permit chunks cost one call.
    """
    cache: dict = {}
    translated  = []
    total_reports   = 0
    total_contexts  = 0
    cache_hits      = 0

    log.info("  Translating event reports and permit contexts (Dutch > English)...")

    for i, er in enumerate(event_reports):
        # Translate llm_report
        tr_report = translate_text(er["llm_report"], client, cache)
        was_cached_r = _hash(er["llm_report"]) in cache and \
                       cache[_hash(er["llm_report"])] != er["llm_report"]
        total_reports += 1

        # Translate each retrieved context
        tr_contexts = []
        for ctx in er["retrieved_contexts"]:
            before_size = len(cache)
            tr_ctx = translate_text(ctx, client, cache)
            if len(cache) == before_size:   # no new entry = cache hit
                cache_hits += 1
            tr_contexts.append(tr_ctx)
            total_contexts += 1

        translated.append({
            **er,
            "llm_report":         tr_report,
            "retrieved_contexts": tr_contexts,
            "llm_report_original":         er["llm_report"],
            "retrieved_contexts_original": er["retrieved_contexts"],
        })

        if (i + 1) % 10 == 0:
            log.info(
                f"  Translated {i+1}/{len(event_reports)} events "
                f"| cache size: {len(cache)} unique texts"
            )

    unique_translated = len(cache)
    stats = {
        "total_report_translations":  total_reports,
        "total_context_lookups":      total_contexts,
        "unique_texts_translated":    unique_translated,
        "cache_hits":                 cache_hits,
        "model_calls_made":           unique_translated,
    }
    log.info(
        f"  Translation complete — {unique_translated} unique texts translated, "
        f"{cache_hits} cache hits saved ({cache_hits} redundant calls avoided)"
    )
    return translated, stats

# METRIC SCORING

async def score_metric_batched(
    metric, samples, metric_name, sleep_sec, max_retries,
    checkpoint_path=None, existing_scores=None
):
    """
    Score all samples sequentially with retry, sleep, and checkpoint support.
    """
    scores = list(existing_scores) if existing_scores else []
    start  = len(scores)
    n      = len(samples)

    if start > 0:
        log.info(f"  [{metric_name}] Resuming from event {start}/{n} "
                 f"({start} already completed)")

    for i in range(start, n):
        sample = samples[i]
        score  = float("nan")
        for attempt in range(max_retries + 1):
            try:
                score = float(await metric.single_turn_ascore(sample))
                break
            except Exception as e:
                if attempt < max_retries:
                    log.warning(
                        f"  [{metric_name}] Event {i+1}/{n} attempt {attempt+1} "
                        f"failed: {e} — retrying in {sleep_sec*2}s"
                    )
                    await asyncio.sleep(sleep_sec * 2)
                else:
                    log.error(
                        f"  [{metric_name}] Event {i+1}/{n} failed after "
                        f"{max_retries+1} attempts: {e}"
                    )
        scores.append(score)

        # Save checkpoint every CHECKPOINT_INTERVAL events
        if checkpoint_path and (len(scores) % CHECKPOINT_INTERVAL == 0 or i == n - 1):
            try:
                cp = _load_checkpoint(checkpoint_path)
                cp[metric_name] = scores
                cp["_partial"] = True
                _save_checkpoint(checkpoint_path, cp)
            except Exception as e:
                log.warning(f"  [{metric_name}] Checkpoint save failed: {e}")

        if i < n - 1:
            await asyncio.sleep(sleep_sec)

    log.info(f"  [{metric_name}] done — "
             f"{sum(1 for s in scores if not math.isnan(s))}/{n} valid scores")
    return scores


def _checkpoint_path(output_dir, period_slug):
    return Path(output_dir) / f"ragas_checkpoint_{period_slug}.json"


def _load_checkpoint(path):
    if Path(path).exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_checkpoint(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=json_serialisable)


async def run_ragas_for_variant(variant_name, top_k, event_reports_path, output_dir):
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithoutReference,
    )
    from openai import OpenAI

    log.info(f"\n{'='*70}")
    log.info(f"RAGAS — {variant_name} (top_k={top_k})")
    log.info(f"Reading: {event_reports_path}")
    log.info(f"Translate to English: {TRANSLATE_TO_ENGLISH}")
    log.info(f"{'='*70}")

    # Load event reports
    with open(event_reports_path, encoding="utf-8") as f:
        event_reports = json.load(f)
    log.info(f"  Loaded {len(event_reports)} event reports.")

    # Verify required fields
    missing = [
        er["event_idx"] for er in event_reports
        if "retrieved_contexts" not in er or "llm_report" not in er
    ]
    if missing:
        log.error(f"  {len(missing)} events missing required fields — aborting.")
        return None

    # Verify k matches expected
    actual_k = len(event_reports[0]["retrieved_contexts"])
    if actual_k != top_k:
        log.warning(
            f"  Expected top_k={top_k} but event reports contain "
            f"{actual_k} contexts — using actual value {actual_k}"
        )
        top_k = actual_k

    # Optional translation pre-pass
    translation_stats = None
    if TRANSLATE_TO_ENGLISH:
        sync_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        event_reports, translation_stats = translate_event_reports(
            event_reports, sync_client
        )

    # Initialise LLM and embeddings
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

    # Initialise metrics
    faithfulness = Faithfulness(llm=evaluator_llm)
    response_rel = ResponseRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        strictness=1,
    )
    context_prec = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

    # Build samples 
    # Two faithfulness sample sets:
    #   faith_legal  — sections 3-4 only, permit chunks only
    #                  → measures legal grounding
    #   faith_factual — full report, paragraph + permit chunks prepended
    #                  → measures absence of hallucination
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

    # Score metrics
    period_slug = f"{DATE_FROM}_to_{DATE_TO}" + (f"_{CHIMNEY}" if CHIMNEY else "")

    # Load checkpoint
    cp_path = _checkpoint_path(output_dir, period_slug)
    cp      = _load_checkpoint(cp_path)
    if cp:
        log.info(f"  Checkpoint found — resuming partial run "
                 f"(completed metrics: {[k for k in cp if not k.startswith('_')]})")

    log.info("  Scoring faithfulness_legal (s3-4, permit chunks only)...")
    faith_legal_scores = await score_metric_batched(
        faithfulness, samples_faith_legal, "faithfulness_legal",
        RAGAS_SLEEP_SEC, RAGAS_MAX_RETRIES,
        checkpoint_path=cp_path,
        existing_scores=cp.get("faithfulness_legal"),
    )

    log.info("  Scoring faithfulness_factual (full report, para+permits)...")
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

    # Build output dict
    def safe_mean(scores):
        valid = [s for s in scores if not math.isnan(s)]
        return sum(valid) / len(valid) if valid else float("nan")

    def safe_nan_count(scores):
        return int(sum(1 for s in scores if math.isnan(s)))

    note = (
        "Dual faithfulness: faithfulness_legal (s3-4 vs permit chunks) and "
        "faithfulness_factual (full report vs paragraph+permits). "
        "user_input is now the full event paragraph. "
        "Generator (gpt-5.1) and evaluator (gpt-oss-120b) are different models. "
        + ("Text translated to English before evaluation." if TRANSLATE_TO_ENGLISH
           else "Raw Dutch text evaluated (TRANSLATE_TO_ENGLISH=False).")
    )

    ragas_output = {
        "meta": {
            "period":               f"{DATE_FROM} to {DATE_TO}",
            "variant":              variant_name,
            "generator_model":      "gpt-5.1",
            "evaluator_model":      LLM_MODEL,
            "embedder":             EMBED_MODEL,
            "top_k":                int(top_k),
            "n_samples":            int(len(samples_full)),
            "translated_to_english": TRANSLATE_TO_ENGLISH,
            "translation_model":    LLM_MODEL if TRANSLATE_TO_ENGLISH else None,
            "translation_stats":    translation_stats,
            "note":                 note,
        },
        "aggregate": {
            "faithfulness_legal":                      safe_mean(faith_legal_scores),
            "faithfulness_legal_nan_count":            safe_nan_count(faith_legal_scores),
            "faithfulness_factual":                    safe_mean(faith_factual_scores),
            "faithfulness_factual_nan_count":          safe_nan_count(faith_factual_scores),
            "answer_relevancy":                        safe_mean(relevancy_scores),
            "llm_context_precision_without_reference": safe_mean(precision_scores),
            "relevancy_nan_count":                     safe_nan_count(relevancy_scores),
            "precision_nan_count":                     safe_nan_count(precision_scores),
        },
        "per_event": [
            {
                "event_idx":      int(er["event_idx"]),
                "chimney":        str(er["chimney"]),
                "event_datetime": str(er["event_datetime"]),
                "retrieved_ids":  list(er["retrieved_ids"]),
                "faithfulness_legal": (
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
    suffix = "_translated" if TRANSLATE_TO_ENGLISH else ""
    ragas_path = Path(output_dir) / f"ragas_results_{period_slug}{suffix}.json"
    with open(ragas_path, "w", encoding="utf-8") as f:
        json.dump(ragas_output, f, ensure_ascii=False, indent=2,
                  default=json_serialisable)
    log.info(f"  Saved → {ragas_path}")

    # Clean up checkpoint file
    cp_path = _checkpoint_path(output_dir, period_slug)
    if Path(cp_path).exists():
        Path(cp_path).unlink()
        log.info("  Checkpoint file removed (variant complete)")
    agg = ragas_output["aggregate"]
    log.info(
        f"  faith_legal={agg['faithfulness_legal']:.4f} | "
        f"faith_factual={agg['faithfulness_factual']:.4f} "
        f"(nan={agg['faithfulness_factual_nan_count']}) | "
        f"rel={agg['answer_relevancy']:.4f} "
        f"(nan={agg['relevancy_nan_count']}) | "
        f"prec={agg['llm_context_precision_without_reference']:.4f} "
        f"(nan={agg['precision_nan_count']})"
    )

    return agg

# MAIN

async def main():
    log.info("=" * 70)
    log.info("RAGAS-only evaluation from saved event reports")
    log.info(f"Period: {DATE_FROM} to {DATE_TO} | Chimney: {CHIMNEY or 'both'}")
    log.info(f"Variants: {[v[0] for v in VARIANTS]}")
    log.info(f"Translate to English: {TRANSLATE_TO_ENGLISH}")
    log.info("=" * 70)

    period_slug = f"{DATE_FROM}_to_{DATE_TO}" + (f"_{CHIMNEY}" if CHIMNEY else "")
    all_results = []

    for variant_name, top_k in VARIANTS:
        output_dir         = Path(BASE_OUTPUT_DIR) / variant_name
        event_reports_path = output_dir / f"event_reports_{period_slug}.json"

        if not event_reports_path.exists():
            log.error(
                f"[{variant_name}] event_reports not found: {event_reports_path}"
            )
            all_results.append({
                "variant": variant_name,
                "error":   "event_reports not found"
            })
            continue

        # Level 1 failsafe: skip if final result already exists
        suffix     = "_translated" if TRANSLATE_TO_ENGLISH else ""
        ragas_path = output_dir / f"ragas_results_{period_slug}{suffix}.json"
        if ragas_path.exists():
            log.info(f"[{variant_name}] Final result already exists — skipping. "
                     f"Delete {ragas_path.name} to force rerun.")
            try:
                with open(ragas_path, encoding="utf-8") as f_in:
                    existing = json.load(f_in)
                all_results.append({"variant": variant_name, "ragas": existing["aggregate"]})
            except Exception:
                all_results.append({"variant": variant_name, "skipped": True})
            continue

        try:
            result = await run_ragas_for_variant(
                variant_name=variant_name,
                top_k=top_k,
                event_reports_path=event_reports_path,
                output_dir=output_dir,
            )
            if result is not None:
                all_results.append({"variant": variant_name, "ragas": result})
            else:
                all_results.append({
                    "variant": variant_name,
                    "error":   "evaluation returned None"
                })
        except Exception as e:
            log.error(
                f"[{variant_name}] failed: {e}\n{traceback.format_exc()}"
            )
            all_results.append({"variant": variant_name, "error": str(e)})

    # Final summary
    log.info("\n" + "=" * 70)
    log.info("FINAL RAGAS SUMMARY")
    log.info("=" * 70)
    for r in all_results:
        if "error" in r:
            log.info(f"  {r['variant']}: FAILED — {r['error']}")
        else:
            s = r["ragas"]
            log.info(
                f"  {r['variant']} | "
                f"faith_legal={s['faithfulness_legal']:.4f} | faith_factual={s['faithfulness_factual']:.4f} (nan={s['faithfulness_factual_nan_count']}) | "
                f"rel={s['answer_relevancy']:.4f} (nan={s['relevancy_nan_count']}) | "
                f"prec={s['llm_context_precision_without_reference']:.4f} "
                f"(nan={s['precision_nan_count']})"
            )

    suffix      = "_translated" if TRANSLATE_TO_ENGLISH else ""
    summary_path = Path(BASE_OUTPUT_DIR) / f"ragas_comparison{suffix}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2,
                  default=json_serialisable)
    log.info(f"\nComparison saved → {summary_path}")
    log.info("All done.")


if __name__ == "__main__":
    asyncio.run(main())
