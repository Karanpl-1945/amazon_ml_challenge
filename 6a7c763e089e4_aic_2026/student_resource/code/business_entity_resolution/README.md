# Business Entity Resolution — pipeline

Everything runs locally from the provided data only (no external lookups / APIs / pretrained models).
Final model: LightGBM (MIT licence) on hand-built pair features.

## 1. Setup
```bash
python -m pip install -r requirements.txt
```
Expected layout (the default paths assume it):
```
student_resource/
├── dataset/train/*.tsv, dataset/test/*.tsv      # the challenge data
├── utils/validate_submission.py
└── code/business_entity_resolution/            # <- this folder, run commands from here
```
Other locations: set `ER_DATA_DIR` (folder with `train/` and `test/`), `ER_WORK_DIR` (intermediate files),
`ER_OUTPUT_DIR` (submission files).

## 2. Run everything (one command)
```bash
cd student_resource/code/business_entity_resolution
python src/run_pipeline.py --workers 8
```
* Outputs: `student_resource/output/matching_results.tsv` (upload this) and `candidate_pairs.tsv`.
* Log of every step (incl. validation recall and F0.5): `student_resource/work/pipeline_log.txt`.
* If a step fails, fix it and resume: `python src/run_pipeline.py --from-step N` (earlier outputs are reused).
* `--workers`: CPU cores to use. Lower it (e.g. 4) if the machine has little RAM or few cores.

**Resources (full data):** ~8 GB RAM is enough (peak ~2.5–3.5 GB per step), ~6 GB free disk for `work/`.
Rough time on an 8-core laptop: 1–1.5 h total (cleaning ~35 min, blocking ~15 min, features ~20 min, training ~10 min).

### Quick smoke test first (2 minutes, recommended)
Builds a 0.5% copy of the data and runs the whole pipeline on it — confirms the environment works.
```bash
python src/make_sample.py --out ../../sample_data --frac 0.005
# Windows PowerShell:
$env:ER_DATA_DIR="../../sample_data"; $env:ER_WORK_DIR="../../sample_data/work"; $env:ER_OUTPUT_DIR="../../sample_data/output"
# Linux/macOS: export ER_DATA_DIR=../../sample_data ER_WORK_DIR=../../sample_data/work ER_OUTPUT_DIR=../../sample_data/output
python src/run_pipeline.py --workers 6
```
It must end with `PASS — no blocking issues found`. Then open a **new terminal** (so the `ER_*` variables
are gone) and run the full pipeline above.

## 3. Steps
| # | Script | What it does | Output (in `work/`) |
|---|--------|--------------|--------|
| 1 | `mine_dictionaries.py` | Aligns train-fold true pairs (noisy S2/S3 vs clean S1) and mines noisy→canonical maps: native-script words, typos, abbreviations, state forms | `dicts/*.json` |
| 2 | `clean_data.py` | Normalises every record of all 6 files (names, addresses, numbers, postcodes, flags) | `clean/*.parquet` |
| 3–4 | `blocking.py` | Candidate generation per country: rare-feature probing + TF-IDF cosine re-ranking, top 20 per S1. Train run prints validation recall@k | `cand/*.parquet` |
| 5–6 | `features.py` | ~50 pair features: string similarities, number/postcode agreement, competition statistics | `feat/*.parquet` |
| 7 | `train_matcher.py` | LightGBM + threshold tuned for macro F0.5 on the validation fold | `model/lgb.txt`, `model/decision.json` |
| 8 | `predict.py` | Scores test candidates, writes both submission TSVs | `output/*.tsv` |
| 9 | `utils/validate_submission.py` | Official format check | stdout |

Helpers: `run_pipeline.py` (driver), `make_sample.py` (smoke-test data), `cleaning_report.py` (optional
before/after-cleaning similarity report), `config.py` (paths, validation split), `normalize.py`, `io_utils.py`.

**Validation protocol:** 20% of train Source-1 entities (deterministic hash, `config.is_val_entity`) are held out.
Dictionaries and the model are fit on the other 80% only; the reported F0.5 is the leaderboard metric
(per-S1 F0.5 incl. singletons, averaged) on the held-out 20%.

## 4. Design notes
### Cleaning (`normalize.py`)
* NFKC, casefold, `&`→`and`, dotted abbreviations joined (`L.L.C.`→`llc`), explicit punctuation split
  (Python `\w` breaks Indic vowel signs), Latin accents folded, web domains reduced to their label.
* Mined dictionaries (train fold only) per country plus a pooled `_all` map used for unseen countries (France).
* Hand rules: legal-suffix canonical forms (incl. French SARL/SAS/EURL), honorifics, generic noise tails,
  alias markers (DBA / formerly / aka), landmark phrases, French street abbreviations.
* Remaining non-ASCII tokens romanised offline with `anyascii`.

### Blocking (`blocking.py`)
* Features per record: name tokens, 4-char token prefix/suffix (typos), space-less name + prefix/suffix
  (`dypharmaceuticals`), address tokens + split compound numbers (`b3/113`→`113`), house number, postcode.
  Weight = idf², separate name and address TF-IDF norms. Blocking is within the `country` label (open set).
* Each S1 record probes the posting lists of its 5 rarest name and 5 rarest address features (df ≤ 10k),
  the top 3000 touched records get exact name/address cosines, score = cos_name + 1.3·cos_addr, keep top 20.
* numba-parallel, memory-lean (integer CSR arrays, parquet streamed in batches).
* Full-scale validation recall (India): recall@10 0.953, recall@20 0.963.

### Matcher (`features.py`, `train_matcher.py`)
* Name: ratio / token-set / token-sort / partial ratio, Jaro-Winkler + ratio on space-less core, core equality,
  token Jaccard, token counts, legal-suffix agreement, alias similarity.
* Address: ratio / token-set / partial-token-set, number-set Jaccard, component Jaccard, house-number and
  postcode agreement, landmark similarity, lengths, empty/native/caps/junk flags.
* Competition (each S2/S3 record belongs to at most one S1): rank and score gap within the S1's list (overall
  and per source), how many S1s retrieved the candidate, its rank and margin among them.
* No country feature, so the model transfers to France.
* Decision: probability ≥ threshold (optionally only the best S1 per candidate), chosen by macro F0.5 on validation.
