import os
import argparse
import sys
import re
import json
import logging
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from collections import Counter

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("run_generation.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# CONFIG


PERMITS_PATH       = "./Data/env_permits.txt"
EVENTS_PARQUET     = "./Processed/aggregated_events.parquet"
PARAGRAPHS_PARQUET = "./Processed/event_paragraphs.parquet"
BASE_OUTPUT_DIR    = "./Generation_Output"

DATE_FROM = "2025-03-01" # Period you wish LLM to investigate
DATE_TO   = "2025-03-31"
CHIMNEY   = None           # "KGF1", "KGF2", or None for both

# Embedder per variant group
EMBED_MODEL_BASE  = "intfloat/multilingual-e5-base"   # Variant A
EMBED_MODEL_LARGE = "intfloat/multilingual-e5-large"  # Variants B and C

# Generator LLM
LLM_BASE_URL   = "https://llmproxy.uva.nl/v1"
LLM_API_KEY    = open("./Data/API.txt").readline().strip() # Replace with your personal one, alongside the endpoint
LLM_MODEL      = "gpt-5.1" 
LLM_MAX_TOKENS = 7000 # Amount is approximated, testing shows that it's supposed to run with no missing generations or failures

INCLUDE_REVOKED = False

# Cross-encoder model for Variant C
CROSS_ENCODER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATE_POOL = 16   # 2 * TOP_K

# Variants: (name, top_k, retrieval_mode, embed_model)
VARIANTS = [
    ("VariantA", 3, "dense",                 EMBED_MODEL_BASE),
    ("VariantB", 8, "hybrid",                EMBED_MODEL_LARGE),
    ("VariantC", 8, "hybrid_rerank",         EMBED_MODEL_LARGE),
    ("VariantD", 8, "hybrid_rerank_chimney", EMBED_MODEL_LARGE),
]

# SENSOR DISCLAIMER
# Documented here for thesis data limitations section.
# NOT passed to the LLM — used for documentation purposes only.

SENSOR_DISCLAIMER = (
    "Note: the compounds cited in OD NZKG enforcement proceedings against "
    "KGF1/KGF2 (PAKs, benzeen, benzo(a)pyreen, and heavy metals at battery "
    "chimneys EL.107/EL.216) are not measured by the available sensor network. "
    "Elevated readings in tracked chemicals indicate a plume event physically "
    "consistent with KGF operations but do not constitute measurement of the "
    "compounds subject to active permit enforcement. Stack emission permit limits "
    "are source limits measured at the chimney and cannot be directly compared "
    "to ambient sensor readings at 1-4 km distance."
)

# PERMIT PARSING

def classify_chimney_scope(content: str) -> str:
    """
    Classify a permit chunk as KGF1-specific, KGF2-specific, or shared.

    Matches both EL/AV emission point codes and Dutch facility name references.
    Chunks referencing both facilities (e.g. 'Kooksfabriek 1 en 2') resolve
    to 'shared'. Chunks with neither code nor name resolve to 'shared'.

    References:
      - OD NZKG omgevingsvergunning 2007-00001 emission point register
      - Tata Steel IJmuiden facility layout documents
    """
    KGF1_CODE = re.compile(r'\bEL\.?\s*1\d{2}\b', re.IGNORECASE)
    KGF2_CODE = re.compile(r'\b(?:EL\.?\s*|AV\.?\s*)2\d{2}\b', re.IGNORECASE)
    KGF1_NAME = re.compile(r'\bkooks(?:gas)?fabriek\s*1\b', re.IGNORECASE)
    KGF2_NAME = re.compile(r'\bkooks(?:gas)?fabriek\s*2\b', re.IGNORECASE)

    has_kgf1 = bool(KGF1_CODE.search(content)) or bool(KGF1_NAME.search(content))
    has_kgf2 = bool(KGF2_CODE.search(content)) or bool(KGF2_NAME.search(content))

    if has_kgf1 and not has_kgf2:
        return "KGF1"
    if has_kgf2 and not has_kgf1:
        return "KGF2"
    return "shared"


@dataclass
class PermitChunk:
    voorschrift_id: str
    content: str
    theme: str
    status: str
    datum_status: str
    embed_text:    str = field(init=False)
    chimney_scope: str = field(init=False)   # "KGF1" | "KGF2" | "shared"

    def __post_init__(self):
        self.embed_text    = f"passage: [{self.voorschrift_id}] {self.content}"
        self.chimney_scope = classify_chimney_scope(self.content)


def parse_permit_chunks(permit_path: str) -> List[PermitChunk]:
    with open(permit_path, encoding="utf-8") as f:
        raw = f.read()

    FIELD_TRIGGERS = {
        "Inhoud":             "content",
        "Milieuthema tata":   "theme",
        "Werkeenheid tata":   "skip",
        "Status voorschrift": "status",
        "Datum status":       "datum",
    }

    chunks = []
    for raw_chunk in [c.strip() for c in raw.split("Voorschriftnummer") if c.strip()]:
        lines = [l.strip() for l in raw_chunk.splitlines() if l.strip()]
        if not lines:
            continue

        current_id      = lines[0]
        current_content = []
        current_theme   = ""
        current_status  = ""
        current_datum   = ""
        reading_field   = None

        for line in lines[1:]:
            if line in FIELD_TRIGGERS:
                reading_field = FIELD_TRIGGERS[line]
                continue
            if reading_field == "content":
                current_content.append(line)
            elif reading_field == "theme":
                current_theme = line
                reading_field = None
            elif reading_field == "status":
                current_status = line
                reading_field = None
            elif reading_field == "datum":
                current_datum = line
                reading_field = None

        if not current_id or not current_content:
            continue
        if not INCLUDE_REVOKED and "Ingetrokken" in current_status:
            continue

        chunks.append(PermitChunk(
            voorschrift_id=current_id.strip(),
            content=" ".join(current_content).strip(),
            theme=current_theme.strip(),
            status=current_status.strip(),
            datum_status=current_datum.strip(),
        ))

    return chunks

# CITATION VALIDITY

def _normalise_id(raw_id: str) -> Optional[str]:
    s = raw_id.strip()
    s = re.sub(r'^[Vv]oorschrift\s+', '', s)
    s = s.split(':')[0].strip()
    s = s.split(' ')[0].strip()
    if re.match(r'^[0-9]$', s) or re.match(r'^[A-Za-z]$', s):
        return None
    return s


def build_corpus_ids(permit_chunks: List[PermitChunk]) -> set:
    return {
        nid for c in permit_chunks
        if (nid := _normalise_id(c.voorschrift_id)) is not None
    }


def extract_cited_ids(text: str, corpus_ids: set) -> set:
    cited = set()
    for norm_id in corpus_ids:
        if re.search(rf'\b{re.escape(norm_id)}\b', text, re.IGNORECASE):
            cited.add(norm_id)
    return cited


def check_citations(event_reports, period_report_text, permit_chunks):
    corpus_ids     = build_corpus_ids(permit_chunks)
    results        = {"events": [], "period": {}, "hallucination_rate": 0.0}
    n_hallucinated = 0

    for er in event_reports:
        cited         = extract_cited_ids(er["llm_report"], corpus_ids)
        valid         = {_normalise_id(rid) for rid in er["retrieved_ids"]
                         if _normalise_id(rid)}
        fabricated    = cited - corpus_ids
        not_retrieved = (cited & corpus_ids) - valid

        has_hallucination = len(fabricated) > 0
        if has_hallucination:
            n_hallucinated += 1

        results["events"].append({
            "event_idx":      er["event_idx"],
            "cited_ids":      sorted(cited),
            "retrieved_ids":  sorted(valid),
            "fabricated_ids": sorted(fabricated),
            "not_retrieved":  sorted(not_retrieved),
            "hallucination":  has_hallucination,
        })

    period_cited = extract_cited_ids(period_report_text, corpus_ids)
    period_fabr  = period_cited - corpus_ids

    results["period"] = {
        "cited_ids":      sorted(period_cited),
        "fabricated_ids": sorted(period_fabr),
        "hallucination":  len(period_fabr) > 0,
    }
    results["hallucination_rate"] = round(
        n_hallucinated / max(len(event_reports), 1), 4
    )
    return results


def citation_diversity_summary(citation_check, n_events):
    all_ids = [iid for e in citation_check["events"] for iid in e["cited_ids"]]
    counter = Counter(all_ids)
    top5    = counter.most_common(5)
    return {
        "unique_ids_cited": len(set(all_ids)),
        "total_citations":  len(all_ids),
        "top_5": [
            {"id": iid, "count": cnt, "pct": round(cnt / n_events * 100, 1)}
            for iid, cnt in top5
        ],
    }

# LLM

from openai import OpenAI

SYSTEM_ROLE = (
    "You are a compliance assessment assistant for OD NZKG. "
    "You may only cite permit conditions and evidence present in the provided context. "
    "Do not infer rules not explicitly retrieved. "
    "If a relevant permit condition is absent from context, state this explicitly. "
    "All output must be written in English regardless of the language of the source documents."
)

HALLUCINATION_GUARD = (
    "After generating your response, verify that every voorschriftnummer "
    "you cite exists in the retrieved context above. "
    "If any cited ID is not present in context, remove it and note the omission."
)


def call_llm(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        temperature=0,          # deterministic — reproducible across reruns
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def format_permit_context(hits: list) -> str:
    lines = []
    for h in hits:
        lines.append(
            f"--- Permit condition [{h['voorschrift_id']}] "
            f"(theme: {h['theme']} | status: {h['status']}) ---\n"
            f"{h['content']}"
        )
    return "\n\n".join(lines)


def build_event_prompt(row: pd.Series, hits: list, top_k: int) -> str:
    paragraph  = str(row.get("retrieval_query", row.get("paragraph", "")))
    structured = row.get("structured_json", "{}")
    permit_ctx = format_permit_context(hits)

    return f"""=== EMISSION EVENT ===
Chimney   : {row['chimney']}
Timestamp : {row['event_datetime']}
Event type: {row.get('soort_voorval', 'N/A')}

--- Natural language summary ---
{paragraph}

--- Structured data (JSON) ---
{structured}

=== RETRIEVED PERMIT CONDITIONS (top {top_k}) ===
{permit_ctx}

=== TASK ===
Write a concise compliance mini-report for this single event. Structure it as:
1. Event summary (1-2 sentences)
2. Severity assessment (reference aq_severity_z, citizen descriptor counts, offshore_flag)
3. Permit conditions potentially relevant to this event (cite each by voorschriftnummer)
4. Preliminary compliance observations (what may or may not align with retrieved permits)
5. Data gaps (any missing evidence or absent permit conditions needed for full assessment)
Respond entirely in English.
{HALLUCINATION_GUARD}
"""


def build_period_prompt(
    event_reports: list,
    merged: pd.DataFrame,
    date_from: str,
    date_to: str,
) -> str:
    summaries = []
    for er in event_reports:
        summaries.append(
            f"[Event {er['event_idx']} | {er['chimney']} | {er['event_datetime']}]\n"
            f"Permits cited: {', '.join(er['retrieved_ids'])}\n"
            f"{er['llm_report']}"
        )
    all_summaries = f"\n\n{'='*60}\n\n".join(summaries)

    aq_scores    = merged["aq_composite_z"].dropna().tolist()
    n_offshore   = int(merged["offshore_flag"].sum()) \
                   if "offshore_flag" in merged.columns else "N/A"
    n_concurrent = int((merged["concurrent_share"] < 1.0).sum()) \
                   if "concurrent_share" in merged.columns else "N/A"

    period_stats = ""
    if aq_scores:
        period_stats = (
            f"Period         : {date_from} to {date_to}\n"
            f"Total events   : {len(event_reports)}\n"
            f"Offshore events: {n_offshore}\n"
            f"Concurrent events (shared attribution): {n_concurrent}\n"
            f"AQ z-score     : mean={np.mean(aq_scores):.2f}, "
            f"max={np.max(aq_scores):.2f}\n"
        )

    return f"""=== PERIOD COMPLIANCE REVIEW ===
{period_stats}
=== PER-EVENT MINI-REPORTS ===
{all_summaries}

=== TASK ===
Based on the per-event reports above, write a consolidated compliance
assessment for this period. Structure it as:
1. Executive summary (3-5 sentences)
2. Recurring permit conditions of concern (group events by voorschriftnummer)
3. Highest-severity events (rank by aq_composite_z, note any with citizen descriptor evidence)
4. Overall recommendation: does this period warrant a formal investigation and why?
5. Data limitations for this assessment

Only cite voorschriftnummers that appear in the per-event reports above.
Write the full assessment in English.
{HALLUCINATION_GUARD}
"""

# RETRIEVAL

def retrieve_dense(
    query_text: str,
    embedder,
    permit_embeddings: np.ndarray,
    permit_chunks: List[PermitChunk],
    k: int,
) -> list:
    query_embed = embedder.encode(
        f"query: {query_text}", normalize_embeddings=True
    )
    scores  = permit_embeddings @ query_embed
    top_idx = np.argsort(scores)[::-1][:k]
    return [
        {
            "voorschrift_id": permit_chunks[i].voorschrift_id,
            "content":        permit_chunks[i].content,
            "theme":          permit_chunks[i].theme,
            "status":         permit_chunks[i].status,
            "score":          float(scores[i]),
        }
        for i in top_idx
    ]


def rrf_fuse(rankings: list, k: int = 60) -> list:
    scores = {}
    for weight, ranked_list in rankings:
        for rank, idx in enumerate(ranked_list):
            scores[idx] = scores.get(idx, 0.0) + weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def retrieve_hybrid(
    query_text: str,
    embedder,
    permit_embeddings: np.ndarray,
    permit_chunks: List[PermitChunk],
    bm25_index,
    k: int,
    semantic_weight: float = 0.7,
    lexical_weight: float  = 0.3,
) -> list:
    query_embed  = embedder.encode(
        f"query: {query_text}", normalize_embeddings=True
    )
    dense_scores = permit_embeddings @ query_embed
    dense_ranked = list(np.argsort(dense_scores)[::-1])

    query_tokens = query_text.lower().split()
    bm25_scores  = bm25_index.get_scores(query_tokens)
    bm25_ranked  = list(np.argsort(bm25_scores)[::-1])

    fused = rrf_fuse(
        [(semantic_weight, dense_ranked), (lexical_weight, bm25_ranked)],
        k=60,
    )
    return [
        {
            "voorschrift_id": permit_chunks[idx].voorschrift_id,
            "content":        permit_chunks[idx].content,
            "theme":          permit_chunks[idx].theme,
            "status":         permit_chunks[idx].status,
            "score":          rrf_score,
            "dense_score":    float(dense_scores[idx]),
            "bm25_score":     float(bm25_scores[idx]),
        }
        for idx, rrf_score in fused[:k]
    ]


def retrieve_hybrid_rerank(
    query_text: str,
    embedder,
    permit_embeddings: np.ndarray,
    permit_chunks: List[PermitChunk],
    bm25_index,
    cross_encoder,
    k: int,
    candidate_pool: int,
) -> list:
    query_embed  = embedder.encode(
        f"query: {query_text}", normalize_embeddings=True
    )
    dense_scores = permit_embeddings @ query_embed
    dense_ranked = list(np.argsort(dense_scores)[::-1])

    query_tokens = query_text.lower().split()
    bm25_scores  = bm25_index.get_scores(query_tokens)
    bm25_ranked  = list(np.argsort(bm25_scores)[::-1])

    fused      = rrf_fuse([(0.7, dense_ranked), (0.3, bm25_ranked)], k=60)
    candidates = fused[:candidate_pool]

    pairs     = [
        (query_text, permit_chunks[idx].content)
        for idx, _ in candidates
    ]
    ce_scores = cross_encoder.predict(pairs)

    def _minmax(arr):
        lo, hi = arr.min(), arr.max()
        return np.ones_like(arr) if hi == lo else (arr - lo) / (hi - lo)

    ce_arr  = np.array([float(s) for s in ce_scores])
    rrf_arr = np.array([rrf_s for _, rrf_s in candidates])
    ce_norm  = _minmax(ce_arr)
    rrf_norm = _minmax(rrf_arr)
    combined = 0.8 * ce_norm + 0.2 * rrf_norm

    ranked = sorted(
        zip(candidates, ce_scores, ce_norm, rrf_norm, combined),
        key=lambda x: x[4],
        reverse=True,
    )
    return [
        {
            "voorschrift_id": permit_chunks[idx].voorschrift_id,
            "content":        permit_chunks[idx].content,
            "theme":          permit_chunks[idx].theme,
            "status":         permit_chunks[idx].status,
            "score":          float(comb),
            "rrf_score":      rrf_s,
            "ce_score":       float(ce_s),
            "dense_score":    float(dense_scores[idx]),
            "bm25_score":     float(bm25_scores[idx]),
        }
        for (idx, rrf_s), ce_s, ce_n, rrf_n, comb in ranked[:k]
    ]


# EMBEDDER LOADER

def retrieve_hybrid_rerank_chimney_aware(
    query_text: str,
    event_chimney: str,
    embedder,
    permit_embeddings: np.ndarray,
    permit_chunks: List[PermitChunk],
    bm25_index,
    cross_encoder,
    k: int,
    candidate_pool: int,
) -> list:

    wrong_chimney = "KGF2" if event_chimney == "KGF1" else "KGF1"

    query_embed  = embedder.encode(
        f"query: {query_text}", normalize_embeddings=True
    )
    dense_scores = permit_embeddings @ query_embed
    dense_ranked = list(np.argsort(dense_scores)[::-1])

    query_tokens = query_text.lower().split()
    bm25_scores  = bm25_index.get_scores(query_tokens)
    bm25_ranked  = list(np.argsort(bm25_scores)[::-1])

    fused      = rrf_fuse([(0.7, dense_ranked), (0.3, bm25_ranked)], k=60)
    candidates = fused[:candidate_pool]

    pairs     = [
        (query_text, permit_chunks[idx].content)
        for idx, _ in candidates
    ]
    ce_scores = cross_encoder.predict(pairs)

    def _minmax(arr):
        lo, hi = arr.min(), arr.max()
        return np.ones_like(arr) if hi == lo else (arr - lo) / (hi - lo)

    ce_arr  = np.array([float(s) for s in ce_scores])
    rrf_arr = np.array([rrf_s for _, rrf_s in candidates])
    ce_norm  = _minmax(ce_arr)
    rrf_norm = _minmax(rrf_arr)
    combined = 0.8 * ce_norm + 0.2 * rrf_norm

    ranked = sorted(
        zip(candidates, ce_scores, ce_norm, rrf_norm, combined),
        key=lambda x: x[4],
        reverse=True,
    )

    valid        = []
    contaminated = []

    for (idx, rrf_s), ce_s, ce_n, rrf_n, comb in ranked:
        scope = permit_chunks[idx].chimney_scope
        entry = {
            "voorschrift_id":   permit_chunks[idx].voorschrift_id,
            "content":          permit_chunks[idx].content,
            "theme":            permit_chunks[idx].theme,
            "status":           permit_chunks[idx].status,
            "score":            float(comb),
            "rrf_score":        rrf_s,
            "ce_score":         float(ce_s),
            "dense_score":      float(dense_scores[idx]),
            "bm25_score":       float(bm25_scores[idx]),
            "chimney_scope":    scope,
            "chimney_filtered": False,
        }
        if scope != wrong_chimney:
            valid.append(entry)
        else:
            entry["chimney_filtered"] = True
            contaminated.append(entry)

    result = valid[:k]
    if len(result) < k:
        needed = k - len(result)
        log.debug(
            f"  [VariantD] Chimney filter: {len(contaminated)} wrong-chimney "
            f"chunks removed, {needed} fallback(s) added"
        )
        result.extend(contaminated[:needed])

    return result[:k]


def load_embedder_and_embeddings(model_name: str, permit_chunks: List[PermitChunk]):
    from sentence_transformers import SentenceTransformer
    log.info(f"Loading embedder: {model_name}")
    embedder          = SentenceTransformer(model_name)
    permit_texts      = [c.embed_text for c in permit_chunks]
    permit_embeddings = embedder.encode(
        permit_texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    permit_embeddings = np.array(permit_embeddings)
    log.info(f"  Embeddings shape: {permit_embeddings.shape}")
    return embedder, permit_embeddings


# VARIANT RUNNER

def run_variant(
    variant_name: str,
    top_k: int,
    retrieval_mode: str,
    embed_model: str,
    permit_chunks: List[PermitChunk],
    permit_embeddings: np.ndarray,
    embedder,
    merged: pd.DataFrame,
    llm_client: OpenAI,
    bm25_index=None,
    cross_encoder=None,
):
    output_dir  = Path(BASE_OUTPUT_DIR) / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)
    period_slug = f"{DATE_FROM}_to_{DATE_TO}" + (f"_{CHIMNEY}" if CHIMNEY else "")

    log.info(f"\n{'='*70}")
    log.info(
        f"VARIANT {variant_name} | mode={retrieval_mode} | "
        f"k={top_k} | embedder={embed_model} | model={LLM_MODEL}"
    )
    log.info(f"Output → {output_dir}")
    log.info(f"{'='*70}")

    # 1. Retrieve
    log.info(f"[{variant_name}] Retrieving permits for {len(merged)} events...")
    retrieved_per_event = []
    for _, row in merged.iterrows():
        # Use retrieval_query (clean paragraph without boilerplate) if available
        para = str(row.get("retrieval_query", row.get("paragraph", "")))

        if retrieval_mode == "dense":
            hits = retrieve_dense(
                para, embedder, permit_embeddings, permit_chunks, top_k
            )
        elif retrieval_mode == "hybrid":
            hits = retrieve_hybrid(
                para, embedder, permit_embeddings, permit_chunks,
                bm25_index, top_k,
            )
        elif retrieval_mode == "hybrid_rerank":
            hits = retrieve_hybrid_rerank(
                para, embedder, permit_embeddings, permit_chunks,
                bm25_index, cross_encoder, top_k, RERANK_CANDIDATE_POOL,
            )
        elif retrieval_mode == "hybrid_rerank_chimney":
            hits = retrieve_hybrid_rerank_chimney_aware(
                para, str(row["chimney"]),
                embedder, permit_embeddings, permit_chunks,
                bm25_index, cross_encoder, top_k,
                RERANK_CANDIDATE_POOL * 2,   # larger pool to cover filtering losses
            )
        retrieved_per_event.append(hits)
    log.info(f"[{variant_name}] Retrieval complete.")

    # 2. Generate per-event reports
    log.info(f"[{variant_name}] Generating {len(merged)} event reports...")
    event_reports = []
    for i, (_, row) in enumerate(merged.iterrows()):
        hits        = retrieved_per_event[i]
        user_prompt = build_event_prompt(row, hits, top_k)
        try:
            llm_report = call_llm(llm_client, SYSTEM_ROLE, user_prompt)
        except Exception as e:
            log.warning(f"[{variant_name}] Event {i+1} LLM call failed: {e}")
            llm_report = f"[GENERATION FAILED: {e}]"

        event_reports.append({
            "event_idx":          int(row["event_idx"]),
            "chimney":            str(row["chimney"]),
            "event_datetime":     str(row["event_datetime"]),
            "user_input":         user_prompt,
            "retrieved_contexts": [h["content"] for h in hits],
            "llm_report":         llm_report,
            "retrieved_ids":      [h["voorschrift_id"] for h in hits],
            "retrieval_scores":   [round(float(h["score"]), 4) for h in hits],
        })

        if (i + 1) % 10 == 0:
            log.info(f"[{variant_name}] Event reports: {i+1}/{len(merged)}")

    log.info(f"[{variant_name}] All event reports generated.")

    # 3. Generate period report
    log.info(f"[{variant_name}] Generating period report...")
    try:
        period_prompt = build_period_prompt(
            event_reports, merged, DATE_FROM, DATE_TO
        )
        period_report = call_llm(llm_client, SYSTEM_ROLE, period_prompt)
    except Exception as e:
        log.warning(f"[{variant_name}] Period report failed: {e}")
        period_report = f"[PERIOD REPORT GENERATION FAILED: {e}]"
    log.info(f"[{variant_name}] Period report done.")

    # 4. Citation check
    log.info(f"[{variant_name}] Running citation validity check...")
    citation_check = check_citations(event_reports, period_report, permit_chunks)
    diversity      = citation_diversity_summary(citation_check, len(event_reports))
    log.info(
        f"[{variant_name}] Hallucination rate: "
        f"{citation_check['hallucination_rate']:.1%}"
    )
    log.info(
        f"[{variant_name}] Diversity: "
        f"{diversity['unique_ids_cited']} unique IDs, "
        f"{diversity['total_citations']} total"
    )

    # 5. Save outputs
    log.info(f"[{variant_name}] Saving outputs...")

    events_path = output_dir / f"event_reports_{period_slug}.json"
    with open(events_path, "w", encoding="utf-8") as f:
        json.dump(event_reports, f, ensure_ascii=False, indent=2, default=str)

    period_path = output_dir / f"period_report_{period_slug}.txt"
    with open(period_path, "w", encoding="utf-8") as f:
        f.write(f"Period  : {DATE_FROM} → {DATE_TO}\n")
        f.write(f"Variant : {variant_name}\n")
        f.write(f"Model   : {LLM_MODEL}\n")
        f.write(f"Embedder: {embed_model}\n")
        f.write(f"Top-k   : {top_k}\n\n")
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
    log.info(f"[{variant_name}] Outputs saved → {output_dir}")

    return {
        "variant":            variant_name,
        "embed_model":        embed_model,
        "hallucination_rate": citation_check["hallucination_rate"],
        "diversity":          diversity,
    }


# MAIN

def main():
    # CLI argument parsing 
    # Allows targeting a single variant without editing the VARIANTS list.
    # Example: python run_generation.py --variant VariantD
    # Included during the testing phase
    parser = argparse.ArgumentParser(description="Stage 2 report generation")
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Run only this variant (VariantA / VariantB / VariantC / VariantD). "
             "Omit to run all variants in VARIANTS list.",
    )
    args = parser.parse_args()

    valid_names   = [v[0] for v in VARIANTS]
    variants_to_run = VARIANTS
    if args.variant:
        if args.variant not in valid_names:
            log.error(
                f"Unknown variant '{args.variant}'. "
                f"Valid options: {valid_names}"
            )
            sys.exit(1)
        variants_to_run = [v for v in VARIANTS if v[0] == args.variant]
        log.info(f"--variant flag set: running {args.variant} only.")

    log.info("=" * 70)
    log.info("Stage 2 — Report Generation")
    log.info(f"Period  : {DATE_FROM} → {DATE_TO} | Chimney: {CHIMNEY or 'both'}")
    log.info(f"Model   : {LLM_MODEL}")
    log.info(f"Variants: {[(v[0], v[3]) for v in variants_to_run]}")
    log.info("=" * 70)

    # Parse permit corpus
    log.info("Loading permit corpus...")
    permit_chunks = parse_permit_chunks(PERMITS_PATH)
    log.info(f"  {len(permit_chunks)} active permit chunks loaded.")

    # Load event data
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

    # Merge — include retrieval_query if present, fall back gracefully
    merge_cols = ["event_idx", "paragraph", "structured_json"]
    if "retrieval_query" in paragraphs_filtered.columns:
        merge_cols.append("retrieval_query")
        log.info("  retrieval_query field found — using clean paragraphs for retrieval.")
    else:
        log.warning(
            "  retrieval_query field not found in parquet — "
            "falling back to full paragraph for retrieval. "
            "Re-run Stage 1 Cells 21-22 to generate clean retrieval queries."
        )

    merged = events_filtered.merge(
        paragraphs_filtered[merge_cols],
        on="event_idx", how="left",
    )
    log.info(f"  {len(merged)} events in period.")

    # Shared components
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

    # Run variants
    # Embedder cache — load each model only once
    embedder_cache = {}

    def get_embedder(model_name):
        if model_name not in embedder_cache:
            embedder, embeddings = load_embedder_and_embeddings(
                model_name, permit_chunks
            )
            embedder_cache[model_name] = (embedder, embeddings)
            log.info(
                f"  Embedder cached: {model_name} "
                f"(dim={embeddings.shape[1]})"
            )
        return embedder_cache[model_name]

    all_results = []
    for variant_name, top_k, retrieval_mode, embed_model in variants_to_run:
        embedder, permit_embeddings = get_embedder(embed_model)
        try:
            result = run_variant(
                variant_name=variant_name,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                embed_model=embed_model,
                permit_chunks=permit_chunks,
                permit_embeddings=permit_embeddings,
                embedder=embedder,
                merged=merged,
                llm_client=llm_client,
                bm25_index=bm25_index,
                cross_encoder=cross_encoder,
            )
            all_results.append(result)
        except Exception as e:
            log.error(
                f"VARIANT {variant_name} FAILED: {e}\n{traceback.format_exc()}"
            )
            all_results.append({"variant": variant_name, "error": str(e)})

    # Summary
    log.info("\n" + "=" * 70)
    log.info("GENERATION COMPLETE — SUMMARY")
    log.info("=" * 70)
    for r in all_results:
        if "error" in r:
            log.info(f"  {r['variant']}: FAILED — {r['error']}")
        else:
            log.info(
                f"  {r['variant']} ({r['embed_model']}) | "
                f"halluc={r['hallucination_rate']:.1%} | "
                f"unique_ids={r['diversity']['unique_ids_cited']} | "
                f"total_citations={r['diversity']['total_citations']}"
            )

    summary_path = Path(BASE_OUTPUT_DIR) / "generation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"\nSummary saved → {summary_path}")
    log.info("Run run_ragas_only.py next for evaluation.")


if __name__ == "__main__":
    main()
