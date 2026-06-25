# Automated IT Troubleshooting and Resolution Prediction

Implementation for the dissertation *"Automated IT Troubleshooting and Resolution Prediction Using NLP, Deep Learning, and Large Language Models"* (MSc Data Science, AI and Digital Business, GISMA University of Applied Sciences, 2026).

This repository contains the full implementation referenced in the dissertation's appendices: every preprocessing function, all 58 model configurations across eight model families, the Retrieval-Augmented Generation pipelines (including the local-Mistral variant), and the Apache Spark distributed port.

## Repository contents

| File | Description |
|---|---|
| `thesis7.ipynb` | Main pipeline — data loading, preprocessing, conventional ML, deep learning, fine-tuned transformers, local LLMs, retrieval/seq2seq generation, proprietary API models (Gemini/GPT-4o-mini/Claude Haiku), and the RAG pipeline (proprietary + local-Mistral generator). 17 cells. |
| `thesis_7_pyspark_version.ipynb` | Distributed reimplementation in Apache Spark — MLlib classification pipeline and a 4-generator distributed RAG comparison (Spark-native T5, Claude, GPT-4o-mini, Gemini). 13 cells. |
| `figures/` | Figures 1–10 as referenced in the dissertation (category distribution, text length, word clouds, bigrams, heatmaps, vocabulary overlap, severity/priority, score analysis, preprocessing effect, confusion matrix). |
| `data/` | The two source datasets (see "Data" section below). |

## Notebook structure

### `thesis7.ipynb`
1. Load and clean data (ServerFault + synthetic corpus, category mapping, HTML stripping)
2. Imports, config, API key loading, result-recording helpers
3. Exploratory data analysis (Figures 1–9 in the dissertation)
4. Preprocessing (lexical normalisation, train/test split, class balancing)
5. Conventional & boosting models (14 classifiers + ensembles, Table 4.1, Figure 10)
6. Deep learning models (9 architectures × 3 embeddings, Table 4.2)
7. Fine-tuned transformers (BERT, DistilBERT, RoBERTa, ALBERT — Table 4.3)
8. Local LLMs — Mistral 7B, Falcon 7B, Llama3 8B, 4-bit quantised (Table 4.4)
9. Generation & retrieval models — TF-IDF/BM25/SBERT/Word2Vec retrieval, T5/BART fine-tuning (Table 4.5)
10. Gemini 2.5 Flash API (Table 4.6)
11. GPT-4o-mini API (Table 4.6)
12. Claude Haiku API (Table 4.6)
13. RAG pipeline — 3 Gemini-based variants (Table 4.7)
14. RAG pipeline — local Mistral 7B generator variant (Table 4.7, RAG-Mistral)
15. Final results dashboard (Appendices E and F)

### `thesis_7_pyspark_version.ipynb`
Runs Spark NLP under a separate Python 3.10 interpreter (installed alongside, not replacing, the notebook's own runtime) since Spark NLP requires Python 3.10 while Colab defaults to a newer version.

1. Install Python 3.10 + PySpark/Spark NLP for that interpreter
2. Dependency-version fixes (pandas/numpy pinned for PySpark 3.3.1 compatibility)
3. Write and run the classification pipeline as a subprocess script → Table 4.8
4. Install API client libraries; write and run the 4-generator RAG pipeline as a subprocess script → Table 4.9

## Data

Both source datasets are included directly in this repository (see `data/`).

### `data/QueryResults.csv.gz` (ServerFault — training set)
Sourced from the **[Stack Exchange Data Explorer](https://data.stackexchange.com/serverfault/query/new)** for the ServerFault site (CC BY-SA 4.0 licensed). 50,000 questions with `Title`, `Body`, `Tags`, `Score`, and the joined `AcceptedAnswer` text.

This file is **gzip-compressed** (40MB vs. ~115MB uncompressed) to stay under GitHub's 100MB per-file limit. No manual step needed — `thesis7.ipynb` decompresses it automatically on first run if `data/QueryResults.csv` doesn't already exist locally. If you need it uncompressed yourself: `gunzip -k data/QueryResults.csv.gz`.

### `data/Expanded_Synthetic_Dataset_50k.csv` (synthetic test set)
Synthetically generated using a generative AI model (LLM-prompted), producing 50,100 IT incidents across the six shared categories with fields for `Category`, `Error_Log_Trace`, `Symptom`, `Root_Cause`, `Resolution_Steps`, `Severity`, and `Priority`. Built independently from the ServerFault corpus to support the cross-domain evaluation design described in the dissertation (Section 1.6).


### Package versions
- **Main notebook**: `transformers`, `accelerate`, `bitsandbytes`, `rouge-score`, `bert-score`, `sentence-transformers`, `gensim`, `rank-bm25`, `wordcloud`, `seaborn`, `scipy`, `xgboost`, `lightgbm`, `catboost`, `google-generativeai`, `openai`, `anthropic`
- **PySpark notebook**: Java 11, Python 3.10, `pyspark==3.3.1`, `spark-nlp==4.4.0`, `py4j==0.10.9.5`, `pandas==1.5.3`, `numpy==1.23.5` (pinned specifically for PySpark 3.3.1 compatibility — see the fix cells in the notebook for why)

### API keys
**No API keys are stored in this repository.** Both notebooks read keys from environment variables at runtime and will print which ones are missing rather than failing silently.

You'll need keys for: Gemini, OpenAI, Anthropic, and a HuggingFace token (for gated/rate-limited model downloads).

**In Google Colab:**
1. Open the Secrets manager (key icon in the left sidebar)
2. Add `GEMINI_KEY`, `OPENAI_KEY`, `CLAUDE_KEY`, `HF_TOKEN` (main notebook) and/or `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` (PySpark notebook — note the different naming)
3. Uncomment the `userdata.get(...)` lines in the relevant setup cell

**Locally:**
```bash
export GEMINI_KEY="..."
export OPENAI_KEY="..."
export CLAUDE_KEY="..."
export HF_TOKEN="..."
```
or use a `.env` file with `python-dotenv` (already excluded via `.gitignore`).

### Data
The two source datasets (`QueryResults.csv` — ServerFault export, and `Expanded_Synthetic_Dataset_50k.csv` — synthetic incident corpus) are not included in this repository due to size. Place them in the working directory before running `thesis7.ipynb`.

## Reproducibility
A fixed random seed (42) is applied across Python's `random`, NumPy, PyTorch, and TensorFlow at the start of every experiment. Proprietary API calls use a fixed low temperature (0.1) to keep output close to deterministic across runs.

## Citation
If referencing this work, please cite the dissertation directly (see the institution's repository for formal citation details).
