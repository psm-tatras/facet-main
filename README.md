# facet-main

Implementation of the FACET citation-ranking work from the FACE corpus, the pipeline:

1. extracts M's **narrative** (TASK / LIMITATION / CONTRIBUTION / METHOD / RESULT) from its abstract;
2. finds the citing sentences in M for each in-corpus reference **R**;
3. extracts each R's narrative;
4. classifies every citing sentence's **relation** of R to M
   (FUNCTION / M_ASPECT / POLARITY / EVIDENCE / R_MATCH).

Both models are LoRA adapters attached to a single 4-bit
`Qwen/Qwen2.5-7B-Instruct` base model.


Downstream steps then score each M→R pair and compare FACET against
retrieval baselines:

```
facet_main               -> data/facet_predictions/<m_id>.json      narratives + relations
prediction.qwen_embedding  (adds aspect_similarity, in place)       M~R similarity per aspect
utils.facet_scoring      -> data/facet_scores/<m_id>.json           FACET relevance score per R
baseline_main            -> data/baselines/<m_id>_bl.json           FACET vs. BM25 / SPECTER2 / SciNCL / SciBERT / Qwen3
```

## FACE Dataset

Download FACE dataset (FACE-Abstract, FACE-Context and FACE-Full) from
[Google Drive](https://drive.google.com/drive/folders/1OYWVjBXG24lpQ8Aa_nwy2onlWxtmb5yz?usp=drive_link).

A main paper (M) cites a set of reference papers (R). `FACE_index.json`, at
the top level, is the index that ties everything else together:

```json
{
  "<m_id>": ["<r_id>", "<r_id>", ...],
  ...
}
```

Look up an M id to get the list of R ids it cites. Every id elsewhere in
this dataset (filenames, JSONL records) refers back to this same id space.

There are three variants of the same M/R pairs, differing only in how much
text is included per paper:

### FACE-Full

Full raw paper records (title, abstract, sections, bibliography, metadata)
for **both M and R papers together**, no distinction between the two roles.

- `paper_ids.txt` — one id per line, M ids first, then R ids (an id already
  listed as an M is not repeated in the R section).
- `papers_batch_NN.zip` — several zips, each containing one `<id>.json` file
  per paper (the raw record). Old-style arXiv ids that contain a `/`
  (e.g. `hep-th/0612132`) are stored with the `/` replaced by `_` in the
  filename — the id inside `FACE_index.json` always keeps the original `/`.
  A given paper's record lives in exactly one of these zips; check each in
  turn (or build a lookup from `paper_ids.txt`'s order) to find it.

### FACE-Context

Full text for M, lighter weight for R — the typical "M in full, R by
abstract" setup for context-aware citation ranking.

- `M_full_text.zip` — one `<id>.json` full record per M paper (same schema
  as FACE-Full).
- `r_title_abstract.jsonl.gz` — one JSON object per line per R paper:
  `{"id": ..., "title": ..., "abstract": ...}`.

### FACE-Abstract

Title + abstract only, for both roles — the lightest variant.

- `m_title_abstract.jsonl.gz` — `{"id", "title", "abstract"}` per M paper.
- `r_title_abstract.jsonl.gz` — `{"id", "title", "abstract"}` per R paper
  (identical content to the one in FACE-Context).

### Typical access pattern

1. Load `FACE_index.json` to get `{m_id: [r_id, ...]}`.
2. Pick an `m_id`, read its M record from whichever variant you need
   (`FACE-Full`'s zips, `FACE-Context/M_full_text.zip`, or
   `FACE-Abstract/m_title_abstract.jsonl.gz`).
3. For each `r_id` under that `m_id`, read its R record the same way from
   the matching variant.

Across all three folders, R ids that are not ground-truth-missing cover the
full `FACE_index.json` reference set — a lookup should never come up empty
unless the id was genuinely unavailable from the source corpus.


## Requirements

- Python 3.10+ and an NVIDIA GPU (a 12 GB card works with the defaults; see
  [Memory tuning](#memory-tuning)).
- Read access to the S3 bucket holding the corpus papers and the adapters.
- A W&B account with access to the project the adapters are registered in.
- The paper index JSONL (see [Paper index](#paper-index)).

## Setup

```bash
# 1. Install a CUDA build of PyTorch that matches your driver, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Install the rest
pip install -r requirements.txt

# 3. Log in to W&B (adapters are resolved through W&B artifacts)
wandb login
```

### Configuration (`.env`)

Create a `.env` in the project root. `facet_main.py` loads it automatically.

```dotenv
# Corpus papers on S3: papers are read from s3://$AWS_BUCKET_NAME/$AWS_DIRECTORY/<paper_id>.json
AWS_BUCKET_NAME=...
AWS_DIRECTORY=...
AWS_REGION=us-east-1
# Optional: omit these to use an IAM role / the default AWS credential chain
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# W&B project holding the adapter artifacts
WANDB_PROJECT=...

# Adapter artifact BASE names (no "-train-best"/"-eval-best" suffix)
WANDB_NARRATIVE_ADAPTER_ARTIFACT_NAME=qwen-narrative-adapter
WANDB_CONTEXT_CLASSIFIER_ADAPTER_ARTIFACT_NAME=qwen-context-classifier-adapter

# Where the adapters are downloaded to (defaults: model/narrative, model/context_classifier)
NARRATIVE_MODEL_LOCAL_DIR=model/narrative
CONTEXT_CLASSIFIER_MODEL_LOCAL_DIR=model/context_classifier

# Optional -- embedding models for aspect similarity and baselines. Each is
# downloaded from the HF Hub on first use and cached in its local dir.
QWEN_EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-4B        # or Qwen/Qwen3-Embedding-0.6B
QWEN_EMBEDDING_MODEL_LOCAL_DIR=model/embedding/qwen-4B
SPECTER2_LOCAL_DIR=model/embedding/specter2
SCINCL_LOCAL_DIR=model/embedding/scincl
SCIBERT_LOCAL_DIR=model/embedding/scibert
```

At startup, each adapter is looked up in W&B. The artifact's metadata holds
its S3 location, and the adapter files are downloaded from there into the
local directory. The context classifier's inference prompt is also read from
its artifact's metadata, so it always matches the loaded adapter.

### Paper index

The pipeline needs a JSONL index with one record per paper. Each record
must include `paper_id`, `title` and `in_corpus_references` (a list of paper
ids). Its path is set by `PAPER_INDEX_FILE` in
[utils/seku_crd_corpus_utils.py](utils/seku_crd_corpus_utils.py). Point that
constant at your local copy before running.

## Running

Run from the project root.

**One paper** (prints the result as JSON):

```bash
python -m facet_main 2308.01689
```

**One paper, saved to disk** (`data/facet_predictions/2308.01689.json`):

```bash
python -m facet_main 2308.01689 --save
```

**Many papers** from an ids file (one id per line; an optional `:domain`
suffix, e.g. `2305.09630:Physics`, is ignored):

```bash
python -m facet_main --ids-file data/M_test_set.txt --save
python -m facet_main --ids-file data/M_test_set.txt --nb-ids 10 --save   # first 10 only
```

In bulk mode, papers are processed one at a time. Each result is written as
soon as it finishes, and its id is appended to
`data/M_test_set_processed.txt`. A paper that fails is logged and skipped,
and the run continues. Ids containing `/` (old-style arXiv ids such as
`hep-th/0404065`) are skipped.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--max-references` | `32` | Max in-corpus references processed per paper (`-1` = all) |
| `--max-index-files` | `4096` | Number of paper-index lines to load (`All` = whole index). The main paper must be within the loaded lines |
| `--context-checkpoint-kind` | `train-best` | Context-classifier artifact to load: `final`, `train-best` or `eval-best` |
| `--batch-size` | `8` | Batch size for narrative generation |
| `--relation-batch-size` | `8` | Batch size for relation classification (one item per citing sentence) |
| `--narrative-max-new-tokens` | `2048` | Generation cap for narratives |
| `--save` | off | Write JSON files instead of printing |
| `--output-dir` | `data/facet_predictions` | Where `--save` writes |

### Memory tuning

If you hit CUDA OOM:

- Lower `--narrative-max-new-tokens` to around `400`–`600`. This saves the
  most memory, because real narrative outputs are about 270 tokens.
- Lower `--batch-size` if the OOM happens during narrative generation, or
  `--relation-batch-size` if it happens during relation classification.

## Output

Each result has the following shape:

```json
{
  "m_id": "2308.01689",
  "m_title": "...",
  "m_abstract": "...",
  "m_aspects": {"TASK": "...", "LIMITATION": "...", "CONTRIBUTION": "...", "METHOD": "...", "RESULT": "..."},
  "references": {
    "<r_id>": [{
      "<r_id>": {
        "title": "...",
        "abstract": "...",
        "aspect": {"TASK": "...", "...": "..."},
        "m_context": [{"section": "...", "sentence_index": 0, "before": [], "center": "...", "after": []}],
        "facet_relevance": [{"FUNCTION": "...", "M_ASPECT": "...", "POLARITY": "...", "EVIDENCE": "...", "R_MATCH": "..."}]
      }
    }]
  }
}
```

`facet_relevance[i]` is the relation predicted for citing sentence
`m_context[i]`.

## Scoring and baselines

Run these after `facet_main --save`, in order. Each step accepts a single
JSON file or a directory of them.

**1. Aspect similarity:** embeds M's and each R's aspects with
Qwen3-Embedding and adds `aspect_similarity` to every reference. The input
files are updated in place.

```bash
python -m prediction.qwen_embedding --input_dir data/facet_predictions
python -m prediction.qwen_embedding --input_file data/facet_predictions/2308.01689.json --overwrite
```

Already-computed entries are skipped unless you pass `--overwrite`. Other
options: `--model`, `--batch_size`, `--max_seq_length`, `-n`.

**2. FACET scoring:** combines the aspect-similarity percentiles with the
citation relations into `facet_score` per reference, plus a top-level
`facet_ranking`. The formula and its constants (`BETA`, `LAMBDA`,
per-FUNCTION weights) are documented at the top of
[utils/facet_scoring.py](utils/facet_scoring.py).

```bash
python -m utils.facet_scoring --input data/facet_predictions --output_dir data/facet_scores
```

Without `--output_dir`, the input files are scored in place. Files already
scored are skipped unless you pass `--overwrite`.

**3. Baselines:** writes `<m_id>_bl.json` holding FACET's vectors next to
each baseline's per-aspect scores for every reference.

```bash
python -m baseline_main --input data/facet_scores --output_dir data/baselines
python -m baseline_main --input data/facet_scores --output_dir data/baselines --methods bm25 bm25_abstract
```

| Method | Family |
|---|---|
| `bm25`, `specter2`, `scincl`, `scibert`, `qwen3_aspect` | Aspect-level: compares FACET's extracted aspects |
| `bm25_abstract`, `specter2_abstract`, `scincl_abstract`, `scibert_abstract`, `qwen3_abstract` | Paper-level: compares title + abstract only |

Runs are incremental: an existing `_bl.json` is extended with only the
missing methods, and models for methods you don't need are never loaded.
BM25 runs on CPU; the others use the GPU. Tuning options:
`--chunk-size`, `--batch-size`, `--qwen-batch-size`. Use
`--bl_dir` to read existing results from a different directory than
`--output_dir`.

## Using it as a library

Load the models and index once, then reuse them across calls:

```python
import asyncio
from facet_main import facet_analyse
from prediction.predict_main import CitationPipelinePredictor
from utils.citation_context import CitationContextExtractor
from utils.seku_crd_corpus_utils import load_paper_index

predictor = CitationPipelinePredictor(
    narrative_local_dir="model/narrative",
    context_classifier_local_dir="model/context_classifier",
    narrative_checkpoint_kind="train-best", narrative_version="v2",
    context_checkpoint_kind="train-best",
)
paper_index = load_paper_index(max_files=-1)
extractor = CitationContextExtractor()

result = asyncio.run(facet_analyse("2308.01689", paper_index, predictor, extractor))
```

## Project layout

```
facet_main.py                # Pipeline entry point (facet_analyse / facet_analyse_many) + CLI
baseline_main.py             # FACET vs. retrieval baselines
prediction/
  predict_*.py               # NarrativeModel, ContextClassifierModel, CitationPipelinePredictor
  qwen_embedding.py          # Aspect similarity (Qwen3-Embedding)
utils/
  facet_scoring.py           # Final FACET M->R relevance score
  baseline_methods.py        # BM25 + cosine helpers for baselines
  embedding_utils.py         # SPECTER2 / SciNCL / SciBERT / Qwen3 encoders
  generate_report.py         # Markdown report per main paper (refreshed by facet_scoring if present)
  ...                        # Corpus/S3 access, citation-context extraction, adapter loading, prompts
```
