"""
Three ablation conditions:
  no_aq        — AQ sensor readings removed from event paragraph
  no_citizen   — Citizen complaint data removed from event paragraph
  no_weather   — Atmospheric context (wind, stability, plume geometry) removed

Comparison matrix records per event (vs baseline):
  - Whether the event's preliminary compliance language changes notably
  - Which voorschriften are cited in ablated vs baseline output
  - Which voorschriften are dropped / newly introduced

Period-level comparison:
  - Does the overall investigation recommendation (Yes/No) change?
  - Which permit conditions disappear from the period summary?
"""

import os
import re
import json
import logging
import traceback
import numpy as np
from pathlib import Path
from openai import OpenAI

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("run_sub3_ablation.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# CONFIGURATION  — edit these to match your local paths

DATE_FROM = "2025-03-01"
DATE_TO   = "2025-03-31"

# Path to VariantD event_reports JSON
VARIANT_D_EVENTS_JSON = (
    f"./Generation_Output/VariantD/"
    f"event_reports_{DATE_FROM}_to_{DATE_TO}.json"
)

BASE_OUTPUT_DIR = "./Generation_Output"
PERIOD_SLUG     = f"{DATE_FROM}_to_{DATE_TO}"

LLM_BASE_URL   = "https://llmproxy.uva.nl/v1"
LLM_API_KEY    = open("./Data/API.txt").readline().strip()
LLM_MODEL      = "gpt-5.1"
LLM_MAX_TOKENS = 7000

CHECKPOINT_EVERY = 10   # save progress every N events

# CONSTANTS

SYSTEM_ROLE = (
    "You are a compliance assessment assistant for OD NZKG. "
    "You may only cite permit conditions and evidence that are present "
    "in the provided context. "
    "Do not infer rules that are not explicitly retrieved. "
    "If a relevant permit condition is absent from context, "
    "state this explicitly. "
    "Write your entire response in English."
)

HALLUCINATION_GUARD = (
    "After generating your response, verify that every voorschriftnummer "
    "you cite exists in the retrieved context above. "
    "If any cited ID is not present in context, remove it and note the omission."
)

ABLATION_CONDITIONS = ["no_aq", "no_citizen", "no_weather"]

# Placeholder text inserted where stripped content was
ABLATION_LABELS = {
    "no_aq":      "[Air quality sensor data not available for this assessment.]",
    "no_citizen": "[No citizen complaint data available for this assessment.]",
    "no_weather": "[Atmospheric context and dispersion geometry not available for this assessment.]",
}

# PARAGRAPH STRIPPING
# The paragraph structure (from Stage 1) is:
#   1. Header       — "Emission event at KGF[X] reported on..."
#   2. Weather      — "Wind [dir] at [speed] m/s. Atmospheric stability: ... 
#                      Estimated plume direction: [X]°. 
#                      Note: [offshore / concurrent info]"
#   3. AQ           — "Monitor NL[XXXXX] (primary, in plume cone): ..."
#                      OR "No in-cone monitor available..."
#   4. Sensor note  — "Note: the compounds cited in OD NZKG enforcement..."
#                      (kept for all conditions — it is a data limitation, not evidence)
#   5. Citizen      — "[N] citizen report(s) within plume window..."
#                      OR "No citizen reports within plume window."

def strip_aq(paragraph: str) -> str:
    # Pattern 1: monitor WAS in cone — "Monitor NL..." through "...may be higher."
    pattern_data = re.compile(
        r'Monitor\s+NL\w+.*?(?:may be higher\.|may exceed the hourly mean\.)',
        re.DOTALL | re.IGNORECASE,
    )
    # Pattern 2: monitor was NOT in cone — covers both:
    #   "No monitor fell within the plume cone for this event. AQ evidence absent."
    #   "No in-cone monitor available..."
    pattern_nodata = re.compile(
        r'(?:No monitor fell within[^.]+\.\s*AQ evidence absent\.'
        r'|No in-cone monitor[^.]*\.)',
        re.DOTALL | re.IGNORECASE,
    )

    stripped = pattern_data.sub(ABLATION_LABELS["no_aq"], paragraph, count=1)
    stripped = pattern_nodata.sub("", stripped, count=1)

    # Remove the concurrent attribution note — meaningless without AQ
    stripped = re.sub(
        r'Note:\s+\d+\s+concurrent\s+KGF\s+event\(s\)\s+within\s+[^;]+;\s+'
        r'AQ\s+readings\s+attributed\s+[^.]+\.',
        'Note: concurrent KGF event(s) present within ±3 hours '
        '(AQ attribution not applicable — sensor data removed).',
        stripped,
        flags=re.IGNORECASE,
    )
    stripped = re.sub(r'\s{2,}', ' ', stripped).strip()
    return stripped


def strip_citizen(paragraph: str) -> str:
    """
    Remove citizen complaint data from the paragraph.
    Removes from '[N] citizen report(s) within plume window' or
    'No citizen reports within plume window' through end of paragraph.
    """
    pattern = re.compile(
        r'(?:\d+\s+citizen\s+report\(s\)\s+within\s+plume\s+window|'
        r'No\s+citizen\s+reports?\s+(?:within|matched|available)[^.]*\.).*$',
        re.DOTALL | re.IGNORECASE,
    )
    stripped = pattern.sub(ABLATION_LABELS["no_citizen"], paragraph, count=1)
    return stripped


def strip_weather(paragraph: str) -> str:
    """
    Remove atmospheric context and dispersion geometry.
    Removes from 'Wind [dir] at [speed]' through the end of the plume/
    concurrent note section, i.e. up to the first mention of 'Monitor' 
    or 'No in-cone' or 'citizen'.
    
    Keeps the event header (timestamp, chimney, event type).
    """
    # Pattern: from "Wind " (start of weather block) through the last sentence
    # of the atmospheric context block, which ends just before "Monitor" or
    # "No in-cone" or the citizen section or the sensor disclaimer.
    #
    # The weather block typically contains:
    #   Wind [dir] at [X] m/s. Atmospheric stability: ... Estimated plume
    #   direction: [X]°. Note: [offshore info OR concurrent info].
    #
    # We match from "Wind " through either:
    #   - "...to this event." (concurrent attribution note)
    #   - "...community exposure unlikely." (offshore note)
    #   - any sentence ending in "." before "Monitor" or digit+citizen

    pattern = re.compile(
        r'Wind\s+\w.*?'
        r'(?:'
        r'attributed\s+(?:fully\s+)?(?:\d+%\s+)?to\s+this\s+event\.|'
        r'community\s+exposure\s+(?:from\s+[^.]+\s+)?unlikely\.|'
        r'plume\s+directed\s+offshore[^.]*\.'
        r')',
        re.DOTALL | re.IGNORECASE,
    )
    stripped = pattern.sub(ABLATION_LABELS["no_weather"], paragraph, count=1)

    if ABLATION_LABELS["no_weather"] not in stripped:
        pattern_fallback = re.compile(
            r'Wind\s+\w.*?(?=Monitor\s+NL|No\s+in-cone|\d+\s+citizen|'
            r'No\s+citizen|Note:\s+the\s+compounds)',
            re.DOTALL | re.IGNORECASE,
        )
        stripped = pattern_fallback.sub(
            ABLATION_LABELS["no_weather"] + " ", paragraph, count=1
        )

    return stripped


def strip_structured_json(json_str: str, condition: str) -> str:
    """
    Remove condition-relevant fields from the structured JSON.
    Conservative: only removes fields clearly belonging to the ablated modality.
    Returns a valid JSON string, or the original string if parsing fails.
    """
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return json_str   # can't parse — return as-is

    if condition == "no_aq":
        aq_keys = [k for k in data if any(
            kw in k.lower() for kw in [
                "aq_", "z_score", "severity_z", "monitor", "fingerprint",
                "composite_z", "general_z", "coke_z", "sensor",
            ]
        )]
        for k in aq_keys:
            data[k] = "[removed — AQ ablation]"

    elif condition == "no_citizen":
        citizen_keys = [k for k in data if any(
            kw in k.lower() for kw in [
                "citizen", "complaint", "odour", "report_count",
                "coke_specific", "particulate_desc", "generic_desc",
                "postcode_count",
            ]
        )]
        for k in citizen_keys:
            data[k] = "[removed — citizen ablation]"

    elif condition == "no_weather":
        weather_keys = [k for k in data if any(
            kw in k.lower() for kw in [
                "wind", "atmospheric", "stability", "pasquill", "plume",
                "offshore", "dispersion", "direction", "speed_extrapolated",
                "concurrent", "attribution_share",
            ]
        )]
        for k in weather_keys:
            data[k] = "[removed — weather ablation]"

    return json.dumps(data, ensure_ascii=False)


# PROMPT MANIPULATION
#
# Rather than re-reading the parquet, we parse the stored user_input from
# VariantD event_reports. Structure of user_input (from build_event_prompt):
#
#   === EMISSION EVENT ===
#   Chimney   : ...
#   Timestamp : ...
#   Event type: ...
#
#   --- Natural language summary ---
#   [paragraph]
#
#   --- Structured data (JSON) ---
#   [structured_json]
#
#   === RETRIEVED PERMIT CONDITIONS (top 8) ===
#   [permit context]
#
#   === TASK ===
#   ...
# =============================================================================

SECTION_SUMMARY    = "--- Natural language summary ---"
SECTION_STRUCTURED = "--- Structured data (JSON) ---"
SECTION_PERMITS    = "=== RETRIEVED PERMIT CONDITIONS"
SECTION_TASK       = "=== TASK ==="


def parse_user_input(user_input: str) -> dict:
    """
    Extract components from a stored user_input string.
    Returns dict with keys: header, paragraph, structured_json, permit_context, task.
    """
    def extract_between(text, start_marker, end_marker):
        s = text.find(start_marker)
        if s == -1:
            return ""
        s += len(start_marker)
        e = text.find(end_marker, s)
        if e == -1:
            return text[s:].strip()
        return text[s:e].strip()

    header         = extract_between(user_input, "=== EMISSION EVENT ===", SECTION_SUMMARY)
    paragraph      = extract_between(user_input, SECTION_SUMMARY, SECTION_STRUCTURED)
    structured_json= extract_between(user_input, SECTION_STRUCTURED, SECTION_PERMITS)
    permit_context = extract_between(user_input, SECTION_PERMITS, SECTION_TASK)
    task           = user_input[user_input.find(SECTION_TASK):]

    return {
        "header":          header,
        "paragraph":       paragraph,
        "structured_json": structured_json,
        "permit_context":  permit_context,
        "task":            task,
    }


STRIP_FUNCS = {
    "no_aq":      strip_aq,
    "no_citizen": strip_citizen,
    "no_weather": strip_weather,
}


def build_ablated_prompt(user_input: str, condition: str, top_k: int = 8) -> str:
    """
    Parse stored user_input, apply modality stripping, rebuild prompt.
    """
    parts = parse_user_input(user_input)
    if not parts["paragraph"]:
        log.warning("Could not parse paragraph from user_input — returning original.")
        return user_input

    stripped_para  = STRIP_FUNCS[condition](parts["paragraph"])
    stripped_json  = strip_structured_json(parts["structured_json"], condition)

    # Rebuild with the same structure build_event_prompt uses
    prompt = (
        f"=== EMISSION EVENT ===\n"
        f"{parts['header']}\n\n"
        f"--- Natural language summary ---\n"
        f"{stripped_para}\n\n"
        f"--- Structured data (JSON) ---\n"
        f"{stripped_json}\n\n"
        f"{SECTION_PERMITS} (top {top_k}) ===\n"
        f"{parts['permit_context']}\n\n"
        f"{SECTION_TASK}\n"
        f"Write a concise compliance mini-report for this single event. Structure it as:\n"
        f"1. Event summary (1-2 sentences)\n"
        f"2. Severity assessment (reference available evidence; note any missing modalities)\n"
        f"3. Permit conditions potentially relevant to this event (cite each by voorschriftnummer)\n"
        f"4. Preliminary compliance observations (what may or may not align with retrieved permits)\n"
        f"5. Data gaps (any missing evidence or absent permit conditions needed for full assessment)\n\n"
        f"{HALLUCINATION_GUARD}\n"
    )
    return prompt


# PERIOD REPORT BUILDER
# (same structure as run_generation.py but takes pre-built event_reports)

def build_period_prompt_ablation(
    event_reports: list,
    condition: str,
    date_from: str,
    date_to: str,
) -> str:
    # Filter out events where generation failed — their error messages
    # would inject garbage into the period context
    valid_reports = [
        er for er in event_reports
        if not str(er.get("llm_report", "")).startswith(
            ("[SKIPPED", "[GENERATION FAILED", "[PERIOD REPORT GENERATION FAILED")
        )
    ]
    n_skipped = len(event_reports) - len(valid_reports)

    summaries = []
    for er in valid_reports:
        summaries.append(
            f"[Event {er['event_idx']} | {er['chimney']} | {er['event_datetime']}]\n"
            f"Permits cited: {', '.join(er['retrieved_ids'])}\n"
            f"{er['llm_report']}"
        )
    all_summaries = f"\n\n{'='*60}\n\n".join(summaries)

    condition_note = {
        "no_aq": (
            "DATA CONSTRAINT: Air quality sensor readings (ambient PM10, PM2.5, SO2, "
            "z-scores, monitor fingerprints) were NOT available for any event in this "
            "period. All severity assessments rely on meteorological context, citizen "
            "complaints, and permit/event metadata only."
        ),
        "no_citizen": (
            "DATA CONSTRAINT: Citizen complaint data (reports, odour descriptors, "
            "postcode counts) were NOT available for any event in this period. "
            "All severity assessments rely on ambient AQ sensor readings, "
            "meteorological context, and permit/event metadata only."
        ),
        "no_weather": (
            "DATA CONSTRAINT: Atmospheric context and dispersion geometry (wind speed, "
            "wind direction, Pasquill-Gifford stability, plume direction, offshore/onshore "
            "classification) were NOT available for any event in this period. "
            "All severity assessments rely on ambient AQ sensor readings, citizen "
            "complaints, and permit/event metadata only."
        ),
    }[condition]

    skipped_note = (
        f"\nNote: {n_skipped} event report(s) were excluded due to generation failures.\n"
        if n_skipped else ""
    )

    return (
        f"=== PERIOD COMPLIANCE REVIEW ===\n"
        f"Period    : {date_from} to {date_to}\n"
        f"Ablation  : {condition}\n"
        f"{condition_note}\n"
        f"Total events assessed: {len(valid_reports)} of {len(event_reports)}{skipped_note}\n\n"
        f"=== PER-EVENT MINI-REPORTS ===\n"
        f"{all_summaries}\n\n"
        f"=== TASK ===\n"
        f"Based on the per-event reports above, write a consolidated compliance\n"
        f"assessment for this period in English. Structure it as:\n"
        f"1. Executive summary (3-5 sentences)\n"
        f"2. Recurring permit conditions of concern (group events by voorschriftnummer)\n"
        f"3. Highest-severity events (rank by the available evidence under this data\n"
        f"   constraint; justify your ranking explicitly)\n"
        f"4. Overall recommendation: does this period warrant a formal investigation?\n"
        f"   Weigh only the evidence available under this data constraint.\n"
        f"   A formal investigation is warranted only if the available evidence\n"
        f"   (without the missing modality) is sufficient to justify one on its own.\n"
        f"   If the remaining evidence is insufficient, say so explicitly and state\n"
        f"   what additional data would be required.\n"
        f"   Begin your answer with one of: Yes / No / Inconclusive.\n"
        f"5. Data limitations specific to this ablation condition\n\n"
        f"Only cite voorschriftnummers that appear in the per-event reports above.\n"
        f"{HALLUCINATION_GUARD}\n"
    )


# COMPARISON MATRIX BUILDERS

def extract_cited_ids_from_report(text: str) -> set:
    """
    Extract voorschrift IDs from LLM output text.
    Uses a whitelist of known permit IDs for this dossier to avoid matching
    decimal measurement values (PM10 concentrations, z-scores, etc.) as IDs.
    """
    # Canonical permit IDs derived from retrieved_ids across all conditions
    CANONICAL_IDS = {
        "0.4.13.H",
        "1.2.7", "1.2.13.H",
        "1.2.20", "1.2.20.H", "1.2.20-A",
        "1.2.20.H.a", "1.2.20.H.b", "1.2.20.H.c",
        "1.2.20.H.d", "1.2.20.H.e", "1.2.20.H.f",
        "1.2.25", "1.2.26",
        "I", "II", "III",
        "2",
    }
    NORMALIZE = {
        "I.": "I", "II.": "II", "III.": "III",
        "1.2.20 H": "1.2.20.H", "1.2.20H": "1.2.20.H",
    }

    def _try_extract(t, found):
        m = re.search(
            r'\b((?:0\.4|1\.2)(?:\.\d+)*(?:[-\.][A-Za-z](?:\.[a-z])?)?)\b', t
        )
        if m:
            found.add(NORMALIZE.get(m.group(1), m.group(1)))
        m2 = re.search(r'\b(I{1,3}V?|VI{0,3})\b', t)
        if m2:
            val = NORMALIZE.get(m2.group(1) + ".", m2.group(1))
            if val in CANONICAL_IDS:
                found.add(val)
        if re.search(r'[Vv]oorschrift\s+2\b', t):
            found.add("2")

    found = set()

    # Bracketed citations: [1.2.26], [I.], [Voorschrift 1.2.20.H.d: ...]
    for bc in re.findall(r'\[([^\]]+)\]', text):
        _try_extract(bc, found)

    # After "Voorschrift " keyword
    for ref in re.findall(
        r'[Vv]oorschrift\s+([\d\.]+[-\w]*(?:\.[A-Za-z](?:\.[a-z])?)?)', text
    ):
        found.add(NORMALIZE.get(ref.strip('.'), ref.strip('.')))

    # After "Besluitonderdeel "
    for ref in re.findall(r'[Bb]esluitonderdeel\s+([IVX]+)', text):
        found.add(NORMALIZE.get(ref + ".", ref))

    # Bare 1.2.x and 0.4.x
    for bid in re.findall(
        r'\b((?:0\.4|1\.2)(?:\.\d+)*(?:[-\.][A-Za-z](?:\.[a-z])?)?)\b', text
    ):
        found.add(NORMALIZE.get(bid, bid))

    # Standalone roman numerals
    for rm in re.findall(
        r'(?:^|[\s\[\(\-–])([IVX]{1,4})(?:\.)?(?=\s|$|\]|\)|\,|\:)',
        text, re.MULTILINE
    ):
        norm = NORMALIZE.get(rm.strip() + ".", rm.strip())
        if norm in CANONICAL_IDS:
            found.add(norm)

    # Voorschrift 2 (NOx)
    if re.search(r'[Vv]oorschrift\s+2\b', text):
        found.add("2")

    found = {NORMALIZE.get(f, f) for f in found}
    return {f for f in found if f in CANONICAL_IDS}


def classify_recommendation(report_text: str) -> str:
    """
    Classify the investigation recommendation from an event or period report.
    Returns: 'yes', 'no', 'inconclusive', or 'unclear'

    For event reports, looks at sections 4-5.
    For period reports, looks for the section 4 opening word (Yes/No/Inconclusive).
    """
    text_lower = report_text.lower()

    # Period report: section 4 should start with Yes/No/Inconclusive
    sec4_match = re.search(
        r'(?:overall recommendation[^:]*:|4\.\s+overall)[^\n]*\n+([^\n]+)',
        report_text, re.IGNORECASE
    )
    if sec4_match:
        opening = sec4_match.group(1).strip().lower()
        if opening.startswith('yes'):           return 'yes'
        if opening.startswith('no'):            return 'no'
        if opening.startswith('inconclusive'): return 'inconclusive'
        if opening.startswith('insufficient'): return 'inconclusive'

    # Keyword-based fallback for both event and period reports
    yes_signals = [
        'warrants further investigation', 'warrants formal investigation',
        'recommends investigation', 'formal investigation is warranted',
        'should be investigated', 'warrants scrutiny', 'investigation is warranted',
        'further regulatory engagement', 'formal investigation warranted',
    ]
    no_signals = [
        'does not warrant', 'no investigation warranted', 'insufficient evidence',
        'cannot be assessed', 'no formal investigation', 'not warranted',
    ]
    inconclusive_signals = [
        'inconclusive', 'insufficient evidence to determine',
        'cannot determine whether', 'evidence is insufficient',
    ]

    yes_count  = sum(1 for s in yes_signals          if s in text_lower)
    no_count   = sum(1 for s in no_signals           if s in text_lower)
    inc_count  = sum(1 for s in inconclusive_signals if s in text_lower)

    if max(yes_count, no_count, inc_count) == 0:
        return 'unclear'
    if yes_count >= no_count and yes_count >= inc_count:
        return 'yes'
    if no_count >= yes_count and no_count >= inc_count:
        return 'no'
    return 'inconclusive'


def build_comparison_matrix(
    baseline_events: list,
    ablated_events_by_condition: dict,
    baseline_period: str,
    ablated_periods_by_condition: dict,
) -> dict:
    """
    Build the structured Sub-RQ 3 comparison matrix.
    
    Returns a dict with:
      - per_event: list of per-event comparisons
      - period_summary: overall recommendation change per condition
      - aggregate: summary statistics
    """
    # Build baseline lookup
    baseline_lookup = {er["event_idx"]: er for er in baseline_events}
    baseline_rec    = classify_recommendation(baseline_period)
    baseline_ids_period = extract_cited_ids_from_report(baseline_period)

    per_event = []
    for event_idx, base_er in baseline_lookup.items():
        base_ids = extract_cited_ids_from_report(base_er["llm_report"])
        base_rec = classify_recommendation(base_er["llm_report"])

        event_row = {
            "event_idx":          event_idx,
            "chimney":            base_er["chimney"],
            "event_datetime":     base_er["event_datetime"],
            "baseline_rec":       base_rec,
            "baseline_ids":       sorted(base_ids),
            "conditions":         {},
        }

        for condition in ABLATION_CONDITIONS:
            ablated_events = ablated_events_by_condition.get(condition, [])
            abl_er = next(
                (e for e in ablated_events if e["event_idx"] == event_idx),
                None,
            )
            if abl_er is None:
                event_row["conditions"][condition] = {"status": "missing"}
                continue

            abl_ids = extract_cited_ids_from_report(abl_er["llm_report"])
            abl_rec = classify_recommendation(abl_er["llm_report"])

            event_row["conditions"][condition] = {
                "recommendation":    abl_rec,
                "rec_changed":       abl_rec != base_rec,  # fixed: no longer suppressed for 'unclear' baseline
                "cited_ids":         sorted(abl_ids),
                "ids_dropped":       sorted(base_ids - abl_ids),
                "ids_added":         sorted(abl_ids - base_ids),
                "ids_unchanged":     sorted(base_ids & abl_ids),
                "n_ids_dropped":     len(base_ids - abl_ids),
                "n_ids_added":       len(abl_ids - base_ids),
            }

        per_event.append(event_row)

    # Period-level summary
    period_summary = {
        "baseline": {
            "recommendation": baseline_rec,
            "cited_ids":      sorted(baseline_ids_period),
        }
    }
    for condition in ABLATION_CONDITIONS:
        abl_period = ablated_periods_by_condition.get(condition, "")
        abl_rec    = classify_recommendation(abl_period)
        abl_ids    = extract_cited_ids_from_report(abl_period)
        period_summary[condition] = {
            "recommendation": abl_rec,
            "rec_changed":    abl_rec != baseline_rec,
            "cited_ids":      sorted(abl_ids),
            "ids_dropped":    sorted(baseline_ids_period - abl_ids),
            "ids_added":      sorted(abl_ids - baseline_ids_period),
        }

    # Aggregate statistics
    aggregate = {}
    for condition in ABLATION_CONDITIONS:
        events_with_condition = [
            e for e in per_event
            if condition in e["conditions"] and "rec_changed" in e["conditions"][condition]
        ]
        n_changed = sum(
            1 for e in events_with_condition
            if e["conditions"][condition]["rec_changed"]
        )
        total_dropped = sum(
            e["conditions"][condition]["n_ids_dropped"]
            for e in events_with_condition
        )
        # Most frequently dropped voorschrift IDs
        all_dropped = [
            iid
            for e in events_with_condition
            for iid in e["conditions"][condition]["ids_dropped"]
        ]
        from collections import Counter
        top_dropped = Counter(all_dropped).most_common(5)

        aggregate[condition] = {
            "n_events_evaluated":      len(events_with_condition),
            "n_rec_changed":           n_changed,
            "pct_rec_changed":         round(n_changed / max(len(events_with_condition), 1) * 100, 1),
            "mean_ids_dropped":        round(total_dropped / max(len(events_with_condition), 1), 2),
            "top_dropped_voorschriften": [{"id": k, "count": v} for k, v in top_dropped],
            "period_rec_changed":      period_summary.get(condition, {}).get("rec_changed", None),
        }

    return {
        "period":    PERIOD_SLUG,
        "baseline_variant": "VariantD",
        "model":     LLM_MODEL,
        "per_event": per_event,
        "period_summary": period_summary,
        "aggregate": aggregate,
    }


# LLM CALL

def call_llm(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_ROLE},
            {"role": "user",   "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# CHECKPOINT HELPERS

def load_checkpoint(path: Path) -> list:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        log.info(f"  Resumed from checkpoint: {len(data)} events already done.")
        return data
    return []


def save_checkpoint(path: Path, data: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


# MAIN

def main():
    log.info("=" * 70)
    log.info("Sub-RQ 3 Modality Ablation")
    log.info(f"Period   : {DATE_FROM} → {DATE_TO}")
    log.info(f"Baseline : VariantD  |  Model: {LLM_MODEL}")
    log.info(f"Conditions: {ABLATION_CONDITIONS}")
    log.info("=" * 70)

    # Load VariantD baseline
    log.info(f"Loading VariantD baseline: {VARIANT_D_EVENTS_JSON}")
    with open(VARIANT_D_EVENTS_JSON, encoding="utf-8") as f:
        baseline_events = json.load(f)
    log.info(f"  {len(baseline_events)} events loaded.")

    # Load VariantD period report as baseline
    period_report_path = Path(VARIANT_D_EVENTS_JSON).parent / f"period_report_{PERIOD_SLUG}.txt"
    if period_report_path.exists():
        baseline_period = period_report_path.read_text(encoding="utf-8")
        log.info(f"  Baseline period report loaded ({len(baseline_period)} chars).")
    else:
        log.warning(f"  VariantD period report not found at {period_report_path}.")
        log.warning("  Period-level comparison will be skipped.")
        baseline_period = ""

    # Setup
    output_dir = Path(BASE_OUTPUT_DIR) / "Sub3Ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    log.info("LLM client initialised.")

    ablated_events_by_condition  = {}
    ablated_periods_by_condition = {}

    # Run each ablation condition
    for condition in ABLATION_CONDITIONS:
        log.info(f"\n{'─'*60}")
        log.info(f"Condition: {condition}")
        log.info(f"{'─'*60}")

        checkpoint_path = output_dir / f"_checkpoint_{condition}_{PERIOD_SLUG}.json"
        output_events_path = output_dir / f"event_reports_{condition}_{PERIOD_SLUG}.json"

        # Skip if final output already exists
        if output_events_path.exists():
            log.info(f"  Output already exists — loading and skipping generation.")
            with open(output_events_path, encoding="utf-8") as f:
                ablated_events_by_condition[condition] = json.load(f)
        else:
            # Resume from checkpoint if available
            done_events = load_checkpoint(checkpoint_path)
            done_idxs   = {e["event_idx"] for e in done_events}
            remaining   = [e for e in baseline_events if e["event_idx"] not in done_idxs]
            log.info(f"  {len(done_events)} already done, {len(remaining)} remaining.")

            for i, base_er in enumerate(remaining):
                event_idx = base_er["event_idx"]

                if not base_er.get("user_input"):
                    log.warning(f"  Event {event_idx}: no user_input stored — skipping.")
                    done_events.append({
                        **{k: base_er[k] for k in ["event_idx", "chimney", "event_datetime"]},
                        "condition":     condition,
                        "ablated_prompt": "[SKIPPED — no user_input in baseline]",
                        "llm_report":    "[SKIPPED]",
                        "retrieved_ids": base_er.get("retrieved_ids", []),
                    })
                    continue

                # Build ablated prompt
                ablated_prompt = build_ablated_prompt(
                    base_er["user_input"], condition
                )

                # LLM call
                try:
                    llm_report = call_llm(llm_client, ablated_prompt)
                except Exception as e:
                    log.warning(f"  Event {event_idx} failed: {e}")
                    llm_report = f"[GENERATION FAILED: {e}]"

                done_events.append({
                    "event_idx":      event_idx,
                    "chimney":        base_er["chimney"],
                    "event_datetime": base_er["event_datetime"],
                    "condition":      condition,
                    "ablated_prompt": ablated_prompt,
                    "llm_report":     llm_report,
                    "retrieved_ids":  base_er.get("retrieved_ids", []),
                })

                # Checkpoint
                if (i + 1) % CHECKPOINT_EVERY == 0:
                    save_checkpoint(checkpoint_path, done_events)
                    log.info(
                        f"  [{condition}] {len(done_events)}/{len(baseline_events)} done "
                        f"(checkpoint saved)."
                    )

            # Save final output
            with open(output_events_path, "w", encoding="utf-8") as f:
                json.dump(done_events, f, ensure_ascii=False, indent=2, default=str)
            log.info(f"  [{condition}] Event reports saved → {output_events_path}")

            # Clean up checkpoint
            if checkpoint_path.exists():
                checkpoint_path.unlink()

            ablated_events_by_condition[condition] = done_events

        # Generate period report
        period_out_path = output_dir / f"period_report_{condition}_{PERIOD_SLUG}.txt"

        if period_out_path.exists():
            log.info(f"  [{condition}] Period report already exists — loading.")
            ablated_periods_by_condition[condition] = period_out_path.read_text(encoding="utf-8")
        else:
            log.info(f"  [{condition}] Generating period report...")
            try:
                period_prompt = build_period_prompt_ablation(
                    ablated_events_by_condition[condition],
                    condition,
                    DATE_FROM,
                    DATE_TO,
                )
                period_text = call_llm(llm_client, period_prompt)
            except Exception as e:
                log.warning(f"  [{condition}] Period report failed: {e}")
                period_text = f"[PERIOD REPORT GENERATION FAILED: {e}]"

            period_out_path.write_text(
                f"Period    : {DATE_FROM} → {DATE_TO}\n"
                f"Condition : {condition}\n"
                f"Model     : {LLM_MODEL}\n\n"
                f"{period_text}",
                encoding="utf-8",
            )
            log.info(f"  [{condition}] Period report saved → {period_out_path}")
            ablated_periods_by_condition[condition] = period_text

    # Build comparison matrix
    log.info("\nBuilding Sub-RQ 3 comparison matrix...")
    comparison = build_comparison_matrix(
        baseline_events=baseline_events,
        ablated_events_by_condition=ablated_events_by_condition,
        baseline_period=baseline_period,
        ablated_periods_by_condition=ablated_periods_by_condition,
    )

    comparison_path = output_dir / f"sub3_comparison_{PERIOD_SLUG}.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"Comparison matrix saved → {comparison_path}")

    # Print summary
    log.info("\n" + "=" * 70)
    log.info("SUB-RQ 3 ABLATION — SUMMARY")
    log.info("=" * 70)
    for condition, stats in comparison["aggregate"].items():
        log.info(
            f"  {condition:15s} | "
            f"rec_changed={stats['n_rec_changed']}/{stats['n_events_evaluated']} "
            f"({stats['pct_rec_changed']}%) | "
            f"mean_ids_dropped={stats['mean_ids_dropped']:.2f} | "
            f"period_changed={stats['period_rec_changed']}"
        )

    log.info(f"\nPeriod-level recommendations:")
    log.info(f"  Baseline    : {comparison['period_summary']['baseline']['recommendation']}")
    for condition in ABLATION_CONDITIONS:
        cond_data = comparison["period_summary"].get(condition, {})
        log.info(
            f"  {condition:12s}: {cond_data.get('recommendation', '?'):7s} "
            f"(changed={cond_data.get('rec_changed', '?')})"
        )

    log.info(f"\nAll outputs in: {output_dir}")
    log.info("Run complete.")


if __name__ == "__main__":
    main()
