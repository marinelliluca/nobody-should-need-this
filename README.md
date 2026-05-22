# NOBODY SHOULD NEED THIS

> #### The job search has become a humiliation ritual due to (1) previous reckless overhiring (2) financial instability due to the end of the globalized neoliberal world order (3) the greed of executives who think they can replace every single employee with a glorified autocomplete (4) the widespread adoption of such glorified autocomplete in hiring and other automated systems that reject resumes before a human ever sees them. 

**This app helps you find jobs you can realistically apply to**, to give you a fighting chance and save you days of relentless, soul-crushing scrolling over job postings that were never intended for you to read.

This is an end-to-end, local-first job-hunting toolkit. It's divided into three python packages:

1. `scraper/`: **Scrape** job postings from multiple boards into one deduplicated DataFrame
2. `rag/`: **Match & rank** those postings against a CV using a local LLM, with the LLM doing semantic judgment and Python doing the arithmetic
3. `interface/`: **Drive both from a browser** via two small Gradio apps, so you don't have to live in a notebook

The workflow is: scrape a portfolio of searches → save a `jobs_*.parquet` → code your CV into an editable profile → score every posting through a four-pass pipeline → tune the weights and skim the survivors.

For a non-technical, click-by-click walkthrough of both apps, see the [user guide](INTERFACE_README.md).

---

**THIS IS NOT AN AI "AGENT" (no tool calling), IT'S NOT VIBE-CODED, AND PRESUPPOSES THAT YOU ARE USING [OLLAMA](https://ollama.com)**

Check this quick [guide for remote Ollama hosting](thundercompute.md) if you don't have enough VRAM at home.

> (❗) Inference runs through Ollama to keep everything local and to avoid managing model loading and GPU memory by hand.Other backends aren't on the roadmap at the moment, but I'd welcome a pull request if you'd like to add one. Adding another backend essentially means reimplementing the small LLM wrapper in `rag/llm.py` and add a couple of config keys in `rag/config.py` and `config.toml`.

## Repository layout

| Directory | What's in it |
|---|---|
| `scraper/` | Apify-backed scrapers. One file per actor, a `main.py` orchestrator that runs them all and dedupes. |
| `rag/` | Retrieval + LLM matching over the scraped DataFrame. Chroma for vector search, Ollama for inference, LangGraph for orchestration. |
| `interface/` | Two Gradio apps (`scraper_app`, `rag_app`) wrapping the two packages, plus shared helpers in `common.py`. |
| `notebooks/` | `scrape.ipynb` and `rag.ipynb` — the same flows as the apps, for when you want full manual control. |

The two halves communicate through a single artifact: a `jobs_*.parquet` file with a stable, unique `key` column. The scraper produces it; the matcher (and the RAG app) consume it.

## Install

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in the values as described below
```

You also need a running [Ollama](https://ollama.com) server with the configured model pulled (used by the whole `rag` package):

```bash
ollama serve
ollama pull qwen3.6:35b-a3b   # or whatever you set as [llm].model in config.toml
```

### Environment variables

Secrets and per-machine paths live in `.env`; tunable defaults live in `config.toml` (see [Configuration](#configuration)). **Env vars always win over TOML.** Keep `.env` gitignored, but you can commit `.env.example` and `config.toml`.

| Variable | Used by | Purpose |
|---|---|---|
| `APIFY_TOKEN` | scraper | Apify API token. Required to run any scrape. |
| `OLLAMA_HOST` | rag | Ollama HTTP endpoint. Falls back to `[llm].default_host`. |
| `OLLAMA_MODEL` | rag | Override the model from `config.toml`. |
| `HF_TOKEN` | rag | Hugging Face token, for downloading the sentence-transformer embedding model (optional). |
| `CHROMA_DIR` | rag | Persistent Chroma directory. If unset, the index is in-memory and rebuilt every session (I would suggest *against* this). |
| `RAG_CONFIG_PATH` | rag | Path to `config.toml`. Defaults to `./config.toml`, and is therefore optional. |

The Gradio apps surface a warning banner at startup listing whichever of their relevant variables are unset, so a missing token is visible before you click anything.

---

# 1. Scraper

Compact multi-platform scraper. Each actor module is one file exposing a `scrape()` that returns a DataFrame in the unified schema. `main.py` runs the selected actors, tags each row with its actor, concatenates, and dedupes.

Currently supported actors:

- **alljobs** — Apify actor `agentx/all-jobs-scraper`, a single call covering LinkedIn, Indeed, Glassdoor, ZipRecruiter, and a handful of regional boards (such as Stepstone, or Naukri, Bayt, Bdjobs, ...).
- **xing** — Apify actor for XING `shahidirfan/xing-jobs-scraper` (only DACH region).

A few additional actor stubs live under `scraper/sources/underdeveloped/` — they are not wired into `ACTOR_NAMES` and not run by default.

## Actor vs. platform

The code keeps two ideas distinct, and so does the schema:

- An **actor** is one of our scraper modules (`alljobs`, `xing`) and is only recorded transiently as the `actor` column during a run.
- A **platform** is the real job board a listing came from (LinkedIn, Indeed, Glassdoor, Xing, …) and is recorded per row in `source_platform`.

Because the unified `alljobs` actor spans several platforms, the two are not 1:1. After dedupe, `actor`/`source_platform` are dropped and replaced by `found_on`: the sorted list of real platforms that surfaced the same role.

## Usage

### A single actor

```python
from scraper.sources.alljobs import scrape
df = scrape("data scientist", location="Berlin", country="de", limit=50)

from scraper.sources.xing import scrape
df = scrape("data scientist", location="Berlin", limit=50)
```

```bash
python -m scraper.sources.alljobs "data scientist" --location Berlin --country de --limit 50
python -m scraper.sources.xing    "data scientist" --location Berlin --limit 50
```

### All actors at once

`scraper.main` runs every registered actor, concatenates, and dedupes on a normalized `(employer_name, title, city)` key:

```bash
python -m scraper.main "data scientist" --location Berlin --limit 50
python -m scraper.main "ML engineer"    --location Berlin --limit 50 --actors xing
python -m scraper.main "data scientist" --location Berlin --no-save   # just print counts
```

Actor-specific flags pass straight through:

- **alljobs:** `--country` (2-letter code or full name, default `de`), `--date-posted` (`"1"`, `"3"`, `"7"`, `"14"` days, freeform like `"6 months"`, or `""` for any time; default `14`), `--remote-only`, `--distance` (radius in miles), `--job-type` (`fulltime`/`parttime`/`internship`/`contract`/…), `--currency` (ISO code for salary FX normalization).
- **xing:** `--discipline` (professional-field filter), `--max-pages`, `--start-url` (a direct Xing search URL; when set, it overrides keyword/location/discipline).

Output flags: `--out-dir` (default `.`) and `--no-save`.

### Output files

A save run writes, under `--out-dir`:

- `jobs_YYYYMMDD_HHMM.parquet` — the full deduped frame **minus** the heavy `raw` column.
- `jobs_YYYYMMDD_HHMM.csv` — a skim for spreadsheets, minus `raw` **and** `description`.
- `raw_results/YYYYMMDD_HHMM/<NNNN>_<employer>_<title>.json` — one file per row holding that posting's original Apify payload, so nothing scraped is ever lost even though it's stripped from the tabular outputs.

## Unified DataFrame schema

Every actor's `scrape()` returns these columns:

`platform_url`, `job_url`, `title`, `employer_name`, `employer_ratings_count`, `employer_rating`, `country`, `city`, `posted_at`, `employment_type`, `description` (full body, HTML stripped), `source_platform` (real board name), `raw` (original Apify dict).

Fields an actor doesn't expose (e.g. Xing has no employer ratings) are set to `None` so concatenation stays clean. After the orchestrator dedupes, it adds `found_on` (the sorted list of real platforms) and a `key` (a short SHA-1 hash over `found_on` + `platform_url` + `title` + `employer_name`). `key` is what the RAG side relies on as a stable, unique document id.

Shared normalization helpers in `scraper/sources/_utils.py` do the cross-source cleanup: HTML stripping, country-name canonicalization (`Deutschland`/`DE`/`Germany` all collapse to one form, via Babel's CLDR data plus a hand-curated supplement for informal aliases like `UK`/`USA`), location splitting into `(city, country)`, and `posted_at` normalization from epochs, ISO strings, or relative phrases ("3 days ago") to a `YY/MM/DD` string.

## Adding an actor

Create one file per actor (`stepstone.py`, …) exposing a `scrape()` that returns the unified schema, plus a `_normalize()` that maps the actor's raw output. `_normalize` **must** set `source_platform` to the real job-board name — a constant string for single-board actors, or read per-item from the raw output for multi-board ones like alljobs. Then add the module name to `ACTOR_NAMES` in `scraper/sources/__init__.py` and a dispatch branch in `main._scrape_one`.

---

# 2. RAG matcher

A small retrieval-augmented pipeline over a DataFrame of postings. Two entrypoints:

- **`ask(query, df)`** — natural-language Q&A over the postings.
- **`match_cv(profile, cv_text, df)`** — score every posting against a CV through a multi-pass pipeline and return ranked results.

Built on Chroma (vector search), sentence-transformers (embeddings), Ollama (local LLM inference), and LangGraph (orchestration).

## The matching pipeline

`match_cv` runs a LangGraph in four passes, ordered cheapest-to-most-expensive so each pass narrows the field for the next:

```
retrieve → prescreen → recruiter_score → candidate_score → select
```

1. **retrieve** — embed `profile["search_query"]` and pull the top `top_n_retrieve` postings from Chroma.
2. **prescreen** — a cheap binary `pass`/`reject` filter that removes postings the candidate *cannot apply to* at all (a human-language requirement above the CV's level, an explicitly unpaid/Werkstudent/student-only role, or a total vertical mismatch). It is not a fit judgment — it defaults to **pass** and rejects only on a clear, explicit trigger. Anything malformed or errored fails *open* (survives), so a flaky LLM call never silently drops a real job.
3. **recruiter_score (0–100)** — a recruiter persona reads the **raw CV** (not the coded profile) and the posting, and rates five themes 0–10: technical match, seniority match, transferability, trajectory, soft skills. A pure function weight-sums them into a 0–100 score. This is the company's-eye view: would this candidate clear an initial screen? Only the top `recruiter_keep_pct` survive to the next pass.
4. **candidate_score (unbounded)** — the CV-coded profile (role themes, must-haves, disqualifiers) is compared to each surviving posting; the LLM returns which themes matched, which must-haves are met/missing, and which disqualifiers fired. A pure function turns those lists into a score. This is the candidate's-eye view: does this job match what *I* want?
5. **select** — join both verdicts back to the DataFrame, sort by `candidate_score` (with `recruiter_score` as tiebreaker), and return.

The recruiter pass reads the raw CV on purpose: the coded profile is already filtered through the candidate's own framing, and the recruiter is meant to form an independent view from the other side of the table.

If an LLM call fails for a single posting, that row gets an empty verdict, a score of 0, and the exception text in an `error` column — the rest of the batch keeps going. Both scoring passes share one generic threadpool runner (`score_parallelism` concurrent calls); threads, not asyncio, because Ollama's HTTP call releases the GIL and threads behave identically in scripts and Jupyter.

## The CV profile is the leverage point

`code_cv(cv_text)` runs thematic coding and returns a plain `CVProfile` dict — `role_themes`, `must_have_themes`, `nice_to_have_themes`, `disqualifiers`, and a `search_query` (a job-posting-voiced paragraph used for retrieval). Every candidate-pass score is computed against this object, so it's deliberately a dict you can inspect and edit before any posting is scored.

## Usage

### Q&A

```python
import pandas as pd
from rag import ask

df = pd.read_parquet("jobs_20260521.parquet")

answer, sources = ask("Which postings mention remote work?", df)
print(answer)
sources.head()                      # the rows the model actually saw

# filter by city/country, or skip sources:
answer, sources = ask("What are the salary ranges?", df, city="Berlin", country="Germany", k=20)
answer = ask("Most common required skills?", df, return_sources=False)
```

### Matching

```python
from rag import code_cv, match_cv

cv_text = open("cv.md").read()

# Step 1 — thematic coding. Inspect and edit before scoring.
profile = code_cv(cv_text)
profile["must_have_themes"].append("Python codebase")
profile["disqualifiers"].append("on-call rotation")

# Step 2 — score and rank. Both inputs are needed: the recruiter pass reads
# the raw CV, the candidate pass reads the profile.
top = match_cv(profile, cv_text, df, top_n_retrieve=200)
top[["title", "employer_name", "city", "recruiter_score", "candidate_score"]].head(20)
```

The returned frame is the survivors sorted by `candidate_score`, joined with the candidate verdict (`themes_matched`, `must_haves_met`, `must_haves_missing`, `disqualifiers_triggered`) and the `recruiter_score` plus its sub-ratings — so you can see *why* each row scored what it did from both sides.

### Split a run across models (prescreen-only + resume)

The prescreen pass is cheap and order-preserving on disk, which enables a two-machine / two-model workflow: run prescreen with a small fast model, then resume the expensive passes with a big model that reuses the prescreen results from cache.

```python
# Pass 1: cheap model, prescreen only. Writes a dump folder under [match.dump].root.
match_cv(profile, cv_text, df, top_n_retrieve=1500, prescreen_only=True)

# Point [match.dump].previous_run_dir at that folder (or pass previous_run_dir=...),
# switch the model, then run the full pipeline. Matching per-key JSONs are reused
# instead of re-querying the LLM.
top = match_cv(profile, cv_text, df, top_n_retrieve=1500)
```

`match_cv` extra parameters: `top_n_retrieve`, `dump=True` (write per-run artifacts to disk), `prescreen_only=False`, and `previous_run_dir=None` (resume-from-cache source; explicit arg wins over `[match.dump].previous_run_dir`). Cache lookups match on the slugified `key`; errored records are treated as misses so transient failures get retried. A non-existent `previous_run_dir` is ignored with a warning rather than failing the run.

## Dump layout

With `dump=True`, each run lands under `[match.dump].root/<slug>/`:

```
<slug>/
├── config.toml                 # snapshot of the active config
├── profile.json                # the CV-coded profile used by the candidate pass
├── prescreen/<key>.json        # one file per retrieved posting (pass/reject)
├── recruiter/<key>.json        # one file per prescreen survivor (full rubric)
├── candidate/<key>.json        # one file per recruiter survivor (candidate verdict)
├── selected_jobs.parquet       # final joined DataFrame (survivors only)
└── summary.csv                 # title, employer, both scores — quick scan
```

The slug is `<model>__n<top_n_retrieve>__keep<recruiter_keep_pct>__<timestamp>`, so two runs with identical config share a leading slug and only differ by the timestamp suffix. Each pass's full output is dumped even though only a subset reaches the next pass — nothing scored is ever lost, which is exactly what makes resume-from-cache and weight replay possible.

## DataFrame schema (RAG side)

The pipeline expects the columns the scraper produces. Missing columns are tolerated (they read as `None` in prompts, `NaN` in metadata) but retrieval and scoring quality degrade.

| Column | Used for |
|---|---|
| `key` | **Chroma document id (must be stable and unique).** Rows are addressed by `key` throughout. |
| `title` | embedding text, prompt context |
| `employer_name` | embedding text, prompt context, metadata filter |
| `city` | embedding text, prompt context, metadata filter |
| `country` | prompt context, metadata filter |
| `description` | embedding text, prompt context (truncated to `description_truncate`) |
| `posted_at` | prompt context |
| `platform` | metadata filter |

Indexing is keyed on `key` (a non-colliding hash), not the DataFrame's positional index, so reordering or re-indexing the frame doesn't invalidate a persisted collection.

---

# 3. Gradio apps

Two independent browser UIs wrap the packages above. They share helpers in `interface/common.py` (log streaming, output writing, dynamic list widgets) but each app's flow lives in its own module.

```bash
python -m interface.scraper_app     # scrape a portfolio of searches
python -m interface.rag_app         # code a CV, match, tune, export

# hot-reload during development:
gradio interface/rag_app.py
```

Importing the `interface` package disables Gradio analytics for every app.

**`scraper_app`** drives a *portfolio* of queries: the form preloads a default list of searches, and you can edit, add, or remove rows. Each query runs independently with its own try/except (one failure doesn't kill the rest), results are concatenated and deduped on `(employer_name, title, city)`, and you get parquet + csv downloads plus a zipped bundle of the raw per-posting JSONs. Requires `APIFY_TOKEN`.

**`rag_app`** drives the matching loop: load a jobs parquet (with a key-uniqueness check), set up a CV profile (auto-code from a CV file or upload an existing profile JSON, then edit the four theme lists in place), run `match_cv`, and then *interactively* re-weight the recruiter and candidate scorers and move a threshold to filter the preview — all without re-prompting, since the LLM extractions are fixed and only the arithmetic re-runs. Export the scored top-N when you're happy. Surfaces `CHROMA_DIR`, `RAG_CONFIG_PATH`, and `HF_TOKEN` as a startup banner if unset.

---

# Configuration

Everything tunable lives in `config.toml`. The loader (`rag/config.py`) reads `.env` and `config.toml` once at import; modules read from the `CONFIG` dict or the helpers `ollama_host()`, `ollama_model()`, `chroma_dir()`. In a long-running notebook you can edit the TOML and call `rag.config.reload()` to refresh in place.

| Section | Key | Notes |
|---|---|---|
| `[llm]` | `model` | Any model your Ollama server has pulled. |
| `[llm]` | `default_host` | Used only if `OLLAMA_HOST` is unset. |
| `[embed]` | `model` | Sentence-transformer model. **Changing this invalidates any persisted Chroma collection** — embeddings aren't comparable across models. Delete `CHROMA_DIR` and re-embed. |
| `[embed]` | `device` | `"cpu"` or `"gpu"` (needs CUDA + enough VRAM). |
| `[index]` | `collection_name` | Chroma collection name. |
| `[index]` | `hnsw_space` | `"cosine"`, paired with `normalize_embeddings=True` in `embed.py`. |
| `[index]` | `filterable_cols` | Columns copied into Chroma metadata for `where`-filtering. `key` is always stored regardless, since scoped search filters on it. |
| `[ask]` | `top_k` | Default k for `ask()`. |
| `[ask]` | `description_truncate` | Per-posting description cap (chars) before the prompt. |
| `[match]` | `top_n_retrieve` | Postings retrieved before scoring. The prescreen and recruiter passes each run once per posting here, so this is the main cost knob. |
| `[match]` | `recruiter_keep_pct` | % of the recruiter pass that survives to the candidate pass. `100` disables the cut. |
| `[match]` | `score_parallelism` | Concurrent extraction calls (both passes). Raise if Ollama has headroom; lower on timeouts. |
| `[match.candidate_scoring]` | `must_haves_met` / `must_haves_missing` / `themes_matched` / `disqualifiers_triggered` | Points per item in each verdict list. Keys must match `_candidate_score()` exactly. |
| `[match.recruiter_scoring]` | `technical_match` / `seniority_match` / `transferability` / `trajectory` / `soft_skills` | Per-theme weights. The LLM rates each 0–10; the score is `sum(weight × rating) / (10 × sum(weights)) × 100`, so each weight sets that theme's share of the 0–100 budget. |
| `[match.dump]` | `root` | Where per-run artifacts go. `""` disables dumping. |
| `[match.dump]` | `previous_run_dir` | A prior dump folder to reuse per-record JSONs from (resume-from-cache). Empty for a fresh run. |
| `[prompts]` | `ask_system`, `code_cv_system`, `prescreen_system`, `match_recruiter_system`, `match_extract_system` | System prompts. The JSON contracts are load-bearing: keys must stay aligned with `RECRUITER_THEMES`, `_candidate_score()`, and the prescreen `decision` field in `match_graph.py`. Edit with care, and keep the "output only JSON" instruction since some smaller models drop the contract under prompt edits. |

## Tuning notes

- **First run is slow.** Embedding a fresh DataFrame takes time. Set `CHROMA_DIR` to skip re-embedding next session — `build_index` skips rows whose `key` already exists.
- **Cost lives in the per-posting passes.** Prescreen and recruiter each run `top_n_retrieve` times; candidate runs `top_n_retrieve × recruiter_keep_pct / 100` times. With `n=200, keep=30` the candidate pass is ~60 calls, not 200.
- **For quick iteration**, drop `top_n_retrieve`. **For high recall**, raise it (and consider raising `recruiter_keep_pct`).
- **Re-weighting is free.** Because the LLM only fills in integers/lists, you can replay `[match.recruiter_scoring]` and `[match.candidate_scoring]` over existing dumps — or live in the `rag_app` — without re-prompting.
- **Split the work across models** with `prescreen_only=True` + `previous_run_dir` when you want a cheap model to do the bulk filtering and a strong one to do the scoring.
- **Malformed JSON?** Check that the system prompts in `config.toml` still end with their JSON-only instruction.
- **Changing `[embed].model` is breaking for persisted indexes.** Delete `CHROMA_DIR` and re-embed.