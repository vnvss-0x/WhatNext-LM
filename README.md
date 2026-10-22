# WhatNext-LM: N-Gram Language Modeling & Next-Word Prediction Engine

A statistical language modeling and interactive smart keyboard system implementing smoothed n-gram architectures trained on the TinyStories narrative corpus. The project evaluates and compares Maximum Likelihood Estimation, Add-k (Laplace) smoothing, Jelinek-Mercer linear interpolation, and Interpolated Kneser-Ney smoothing on held-out test data.

Live Web Application: https://whatnext-lm.streamlit.app/

---

## Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Dataset and Tokenization](#dataset-and-tokenization)
- [Interactive Web Interface Features](#interactive-web-interface-features)
- [Repository Structure](#repository-structure)
- [Installation and Local Execution](#installation-and-local-execution)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
  - [Running the Streamlit Application](#running-the-streamlit-application)
  - [Running the Evaluation Suite](#running-the-evaluation-suite)
  - [Regenerating Corpus Data](#regenerating-corpus-data)

---

## Overview

Predictive typing in smart keyboards requires low-latency, memory-efficient language modeling capable of delivering robust probability estimates over next-word candidates. While unsmoothed Maximum Likelihood Estimation (MLE) assigns zero probability to unobserved sequences, smoothing algorithms distribute probability mass to unseen transitions.

WhatNext-LM implements and benchmarks three smoothing strategies against unsmoothed n-gram baselines:
1. Trigram Hard Back-off with Add-k smoothing
2. Linear Interpolation across Trigram, Bigram, and Unigram distributions
3. Interpolated Kneser-Ney smoothing utilizing absolute discounting and lower-order continuation counts

The models are deployed in a Streamlit application featuring predictive suggestion chips, side-by-side model outputs, and held-out performance visualizations.

---

## Live Demo

The application is deployed and publicly accessible at:
https://whatnext-lm.streamlit.app/

---

## Dataset and Tokenization

- **Corpus Source**: `roneneldan/TinyStories` from Hugging Face Datasets.
- **Corpus Volume**: 2,000 synthetic children's stories.
- **Sentence Count**: 31,754 sentences.
- **Total Token Count**: 396,664 tokens.
- **Vocabulary Size**: 5,858 unique word types.
- **Tokenization**:
  - Punctuation separation: `([.!?,;:'"()\\-])` mapped to distinct tokens.
  - Case normalization: all tokens lowercased.
  - Boundary markers: `<s>` (BOS) and `</s>` (EOS) appended per sentence.
- **Partitioning Strategy**: Partitioned strictly by document index (last 20% reserved for testing) to prevent cross-sentence leakage between training and evaluation splits.

---

## Interactive Web Interface Features

The Streamlit web interface is structured into four main operational tabs:

1. **Smart Keyboard**:
   - **Clickable Suggestion Chips**: Top predicted candidate tokens are rendered as clickable buttons with confidence percentages. The highest-ranked token is highlighted as the primary candidate.
   - **Click-to-Type Flow**: Clicking any suggestion chip appends the token, formats punctuation spacing automatically, and re-predicts for the next position.
   - **Action Toolbar**: Includes Undo Last Word, Autofill (+1 token), Auto-Complete (+5 tokens), Random Story Starter, and Clear.
   - **Context Inspector**: Real-time display of active trigram and bigram contexts, corpus lookup hit status, and back-off tracking.
   - **Distribution Visualization**: Altair bar chart displaying candidate probability distribution.
2. **Side-by-Side Model Comparison**:
   - Compares Kneser-Ney, Trigram Back-off, and Linear Interpolation predictions on identical user input.
   - Includes one-click test cases for known failure modes and unseen trigram contexts.
3. **Evaluation and Leaderboard**:
   - Dynamic leaderboard summarizing held-out perplexity, zero-probability token share, and Top-3 accuracy.
   - Interactive hyperparameter sensitivity plots ($k$-sweep and $d$-sweep).
4. **NLP Guide and Theory**:
   - Complete reference documentation with mathematical formulas explaining n-gram mechanics, smoothing derivations, and continuation count proofs.

---

## Repository Structure

```
.
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── src/
│   ├── pipeline.py            # NGramModel, InterpolatedModel, KneserNeyModel implementations
│   └── evaluate.py            # Evaluation suite (perplexity, accuracy, sweeps)
├── data/
│   ├── corpus.csv             # Processed TinyStories tokenized sentences
│   ├── fetch_tinystories.py   # Script to stream and process dataset from Hugging Face
│   └── make_data.py           # Corpus generation entrypoint
└── results/
    ├── metrics.json           # Serialized held-out evaluation results and parameter sweeps
    ├── failures.txt           # Error analysis of mispredicted contexts
    └── ngram_results.png      # Comparison plots (perplexity vs k, accuracy, etc.)
```

---

## Installation and Local Execution

### Prerequisites

- Python 3.10, 3.11, or 3.12
- Git

### Setup

Clone the repository and install required packages:

```bash
git clone https://github.com/vnvss-0x/WhatNext-LM.git
cd WhatNext-LM

# Create and activate virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Running the Streamlit Application

Start the local development server:

```bash
streamlit run app.py
```

The application will launch at `http://localhost:8501`.

### Running the Evaluation Suite

To recompute held-out perplexity, accuracy, and hyperparameter sweeps:

```bash
python -m src.evaluate
```

This updates `results/metrics.json`, `results/failures.txt`, and `results/ngram_results.png`.

### Regenerating Corpus Data

To re-stream and preprocess stories from Hugging Face:

```bash
# Default: 2,000 stories
python data/fetch_tinystories.py

# Custom story volume:
python data/fetch_tinystories.py --max-stories 5000
```

---

