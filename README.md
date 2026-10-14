# Automated Call for Action: RAG-Based Environmental Law Enforcement Screening System

**Master's Thesis — Information Studies (Data Science), University of Amsterdam, 2026**  
**Author:** Matas Jurkevicius  
**Supervisor:** Yen-Chia Hsu (y.c.hsu@uva.nl)  
**Repository:** [github.com/MatasJURK/final_ISDS](https://github.com/MatasJURK/final_ISDS/tree/main)

---

# Update: This is a re-upload of the initial thesis repository. After checking the repository, potentially sensitive data was found to be present, so a call was made to completely remove any access to it by deleting it.

## Overview

This repository contains the full implementation of a two-stage compliance screening pipeline developed as a feasibility study for the **Kooksgasfabriek facilities (KGF1 and KGF2)** at Tata Steel IJmuiden, supervised by Omgevingsdienst Noordzeekanaalgebied (OD NZKG).

**Stage 1 (st1)** aggregates heterogeneous evidence streams — atmospheric dispersion modelling, ambient air quality sensor readings, and citizen complaint reports — into structured natural language event paragraphs.

**Stage 2 (st2)** retrieves relevant environmental permit conditions (voorschriften) using four retrieval strategies of increasing sophistication and generates legally-grounded compliance assessments and period-level investigation recommendations using GPT-family LLMs.

Output quality is evaluated using RAGAS reference-free metrics and an LLM-as-judge rubric. The thesis and full results are available at [INSERT THESIS LINK].

---

## Repository Structure

```
.
├── Data/
│   └── API.txt                        # LiteLLM proxy API key (not included — see Installation)
├── Processed/
│   ├── aggregated_events.parquet      # Stage 1 output: 707 events with evidence fields (included)
│   └── event_paragraphs.parquet       # Stage 1 output: natural language paragraphs (included)
├── Pre_processing.ipynb               # Raw data → parquet conversion
├── st1_aggregation.ipynb              # Evidence aggregation pipeline (Stage 1)
├── st1_evaluation.ipynb               # Stage 1 statistical evaluation
├── st2_retrieval_generation.py        # Stage 2: report generation (all four variants)
├── st2_models.py                      # Stage 2: model comparison (GPT-5-mini, GPT-5-nano)
├── st2_retrieval_ragas.py             # RAGAS evaluation (retrieval variants)
├── st2_models_ragas.py                # RAGAS evaluation (model comparison)
├── st2_retrieval_judge.py             # LLM-as-judge (retrieval variants)
├── st2_models_judge.py                # LLM-as-judge (model comparison)
├── st3_ablation.py                    # SRQ3 modality ablation
├── st3_ablation_judge.py              # LLM-as-judge (ablation period reports)
└── README.md
└── requirements.txt
```

---

## Data Availability

| Dataset | Source | Included |
|---|---|---|
| OD NZKG emission event reports (707 events, 2025) | OD NZKG | No — contact supervisor |
| Citizen complaint reports | OD NZKG | **No — privacy restricted** (see below) |
| RIVM Luchtmeetnet AQ readings (6 monitors, 2025) | [luchtmeetnet.nl](https://www.luchtmeetnet.nl) | Public |
| KNMI hourly meteorological observations (station 240) | [knmi.nl](https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens) | Public |
| CBS PC4 postcode geometry 2024 | [cbs.nl](https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/cbs-gebiedsindelingen) | Public |
| KGF1/KGF2 environmental permit corpus | OD NZKG | Included (`env_permits.txt`) |
| Stage 1 aggregated outputs | This pipeline | Included (`Processed/`) |

**Citizen complaint data:** The raw citizen complaint file contains residential postcode data and is not included due to privacy constraints imposed by OD NZKG. The aggregated outputs in `Processed/event_paragraphs.parquet` contain only anonymised counts and descriptor types derived from this data and are safe to include. Researchers requiring access to the raw complaint data should contact the thesis supervisor (y.c.hsu@uva.nl), who can facilitate communication with OD NZKG.

**Skipping Stage 1:** Because both Stage 1 outputs are included in `Processed/`, Stage 2 can be run directly without the private datasets. See Execution Order below.

---

## Pipeline Architecture

### Stage 1 — Evidence Aggregation (`st1_aggregation.ipynb`)

**Key processing steps:**
- Gaussian puff dispersion modelling (Turner, 1994) for plume travel time and cone geometry
- Pasquill-Gifford stability classification from KNMI solar radiation and cloud cover
- Wind profile power law extrapolation to stack height (Irwin, 1979)
- 30-day rolling z-score normalisation of AQ readings per monitor per chemical
- Temporal and spatial matching of citizen reports to plume cone postcodes

**Outputs:** `Processed/aggregated_events.parquet` (707 events with structured evidence fields) and `Processed/event_paragraphs.parquet` (natural language event summaries and retrieval queries).

### Stage 2 — RAG Generation (`st2_retrieval_generation.py`, `st2_models.py`)

Four retrieval variants are evaluated, building progressively from a naïve dense baseline to a facility-scope-aware hybrid pipeline:

| Variant | Strategy | Embedding model | k |
|---|---|---|---|
| A (baseline) | Dense only | multilingual-e5-base | 3 |
| B | Hybrid BM25 + dense, RRF 70/30 | multilingual-e5-large | 8 |
| C | Variant B + cross-encoder reranking (MiniLM-L-6-v2) | multilingual-e5-large | 8 |
| D | Variant C + chimney-aware post-filter | multilingual-e5-large | 8 |

Three generation models are compared on Variant D retrieval: GPT-5.1, GPT-5-mini, GPT-5-nano.

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `sentence-transformers`, `rank-bm25`, `openai`, `ragas`, `pandas`, `numpy`, `geopandas`, `scikit-learn`.

### API Key

This project uses the **UvA LiteLLM proxy** for access to the GPT-5.1 model family. The pipeline is compatible with any OpenAI-compatible endpoint.

Create `Data/API.txt` and place your API key on the first line:

```
sk-your-api-key-here
```

Update `LLM_BASE_URL` in each script if using a different endpoint:

```python
LLM_BASE_URL = "https://llmproxy.uva.nl/v1"  # UvA proxy (default)
LLM_BASE_URL = "https://api.openai.com/v1"    # OpenAI directly
```

---

## Execution Order

Both Stage 1 output files are included in `Processed/`, so **Stage 2 can be run directly** without needing the private datasets. Steps 1–2 are only required if you want to reproduce Stage 1 from raw data.

```bash
# --- OPTIONAL: reproduce Stage 1 from raw data ---
# Requires: OD NZKG event reports, citizen complaints, RIVM AQ data, KNMI data, CBS PC4
# 1. Convert raw data to parquet
jupyter nbconvert --to notebook --execute Pre_processing.ipynb

# 2. Run Stage 1 aggregation (overwrites included parquets)
jupyter nbconvert --to notebook --execute st1_aggregation.ipynb

# --- Stage 2 onwards: runs from included Processed/ files ---

# 3. Generate reports for all four retrieval variants
python st2_retrieval_generation.py

# 4. Copy GPT-5.1 VariantD outputs to model comparison folder
#    (GPT-5.1 results are taken from the VariantD run — no separate generation needed)
python -c "
import shutil, pathlib
src = pathlib.Path('./Generation_Output/VariantD')
dst = pathlib.Path('./Generation_Output/ModelComparison/gpt-5-1')
dst.mkdir(parents=True, exist_ok=True)
for f in src.glob('*.json'): shutil.copy2(f, dst)
for f in src.glob('*.txt'):  shutil.copy2(f, dst)
"

# 5. Run model comparison (GPT-5-mini and GPT-5-nano on Variant D retrieval)
python st2_models.py

# 6. LLM-as-judge evaluation
python st2_retrieval_judge.py     # retrieval variants
python st2_models_judge.py        # model comparison

# 7. RAGAS evaluation
python st2_retrieval_ragas.py     # retrieval variants
python st2_models_ragas.py        # model comparison

# 8. SRQ3 modality ablation
python st3_ablation.py
python st3_ablation_judge.py

# 9. Stage 1 statistical evaluation
jupyter nbconvert --to notebook --execute st1_evaluation.ipynb
```

Most scripts support `--variant` and `--dry-run` flags. See each script's docstring for details.

> **Note on stale outputs:** Scripts include checkpoint logic that skips already-generated events. If re-running from scratch, delete the relevant `Generation_Output/` subdirectory first to avoid silent cache hits.

---

## Key Results

March 2025 evaluation period, 74 events. Judge model: GPT-5.1. RAGAS scores computed on pre-correction paragraphs; judge scores and mismatch rates from corrected re-run (see thesis Section 4, Disclaimer).

### Retrieval Variants (GPT-5.1 generator)

| Variant | F_legal | F_fact | Rel. | Prec. | Judge/12 | Mm% ↓ |
|---|---|---|---|---|---|---|
| A — Dense only (k=3) | 0.317 | 0.747 | 0.644 | 1.000 | 11.27 | 8.1 |
| B — Hybrid BM25+dense | 0.353 | 0.670 | 0.597 | 0.921 | 11.01 | 14.9 |
| C — Hybrid + reranking | 0.370 | 0.692 | 0.618 | 0.936 | 11.49 | 8.1 |
| **D — Chimney-aware filter** | **0.360** | **0.706** | **0.567** | **0.933** | **11.45** | **2.7** |

Mm% = chimney mismatch rate (fraction of events with at least one cross-facility permit citation; lower is better). Variant A excluded from primary comparison due to k=3 sparsity artefact (context precision 1.000 by definition).

### Model Comparison (Variant D retrieval)

| Model | Judge/12 | LG | EI | Mm% ↓ |
|---|---|---|---|---|
| GPT-5.1 | 11.42 | 2.85 | 2.59 | 0.0 |
| GPT-5-mini | 11.53 | 2.86 | 2.69 | 5.4 |
| GPT-5-nano | 7.08 | 1.92 | 1.97 | 10.8 |

### Modality Ablation (GPT-5.1, Variant D)

| Condition | Events changed | Period recommendation |
|---|---|---|
| Baseline | 0 / 74 | Yes |
| No AQ data | 15 / 74 (20.3%) | No |
| No citizen reports | 15 / 74 (20.3%) | Yes (unchanged) |
| No weather/dispersion | 22 / 74 (29.7%) | Inconclusive |

---

## Known Limitations

- **No ground truth:** No event-level verified violation labels exist. Evaluation uses reference-free metrics and LLM-as-judge only.
- **Meteorological data offset:** KNMI station Schiphol (240) is approximately 30 km from IJmuiden. Local coastal wind conditions may differ from station observations.
- **Sensor coverage gap:** The RIVM ambient network does not measure PAKs, benzeen, or benzo(a)pyreen — the compounds under active OD NZKG enforcement at KGF1/KGF2.
- **Chimney classifier scope:** The post-filter classifies permit conditions using EL/AV emission point codes and Dutch facility name references. Conditions using other naming conventions may evade the filter.
- **Single case study:** The system was designed and evaluated exclusively for KGF1/KGF2. Generalisation to other facilities requires re-deriving chemical fingerprint groups, facility identification logic, and permit chunking strategy.
- **OD NZKG non-participation:** Expert evaluation of generated reports by OD NZKG investigators was planned but could not be completed due to the ongoing permit revocation proceedings against Tata Steel (May 2026).

---

## Citation

```
Jurkevicius, M. (2026). Automated call for action: RAG-based environmental
law enforcement screening system. Master's thesis, University of Amsterdam.
```

---

## License

No license has been formally assigned to this repository. All rights reserved by the author. Researchers wishing to build on this work should contact the author via GitHub.

---

## Acknowledgements

This research was conducted at the University of Amsterdam under the supervision of Yen-Chia Hsu. LLM inference was provided by the UvA AI infrastructure. Public data was sourced from RIVM, KNMI, and CBS.
