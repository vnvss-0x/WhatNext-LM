"""Stream roneneldan/TinyStories from HuggingFace and write data/corpus.csv.

Each HuggingFace story becomes one document (doc_id = story index).
Sentences are split on punctuation boundaries and word-tokenised (lower-case,
punctuation separated).  The tags column is left empty — TinyStories carries
no gold POS annotations.

Run from the project root
--------------------------
    python data/fetch_tinystories.py                    # 2 000 stories
    python data/fetch_tinystories.py --max-stories 500  # quick smoke-test
    python data/fetch_tinystories.py --max-stories 5000 # larger corpus

Output
------
    data/corpus.csv   columns: doc_id, sentence, tags
    (same schema as make_data.py — downstream code needs no changes)
"""
import argparse
import csv
import os
import re
import sys
from pathlib import Path


# ------------------------------------------------------------------ text utils

# Separate common punctuation so they become individual tokens
_PUNCT_RE  = re.compile(r"([.!?,;:'\"()\-])")
_SPACE_RE  = re.compile(r"\s+")


def tokenize(text: str) -> list:
    """Lowercase and space-separate punctuation marks."""
    text = _PUNCT_RE.sub(r" \1 ", text.lower())
    return _SPACE_RE.sub(" ", text).strip().split()


# Split on whitespace that follows a sentence-ending punctuation mark.
# Keeps the punctuation attached to the preceding sentence.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list:
    """Rough sentence boundary detection for narrative text."""
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(
        description="Fetch TinyStories from HuggingFace and write data/corpus.csv"
    )
    ap.add_argument(
        "--max-stories", type=int, default=2000,
        help="Number of stories to stream (default: 2000)",
    )
    ap.add_argument(
        "--min-sent-tokens", type=int, default=3,
        help="Drop sentences shorter than this many tokens (default: 3)",
    )
    ap.add_argument(
        "--hf-split", default="train",
        help="HuggingFace dataset split to use (default: train)",
    )
    args = ap.parse_args()
    if args.max_stories < 1:
        ap.error("--max-stories must be greater than zero")
    if args.min_sent_tokens < 1:
        ap.error("--min-sent-tokens must be greater than zero")

    # ---- import guard
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "\n[ERROR] 'datasets' package not found.\n"
            "Install it with:  pip install datasets\n"
        )

    print(f"Streaming roneneldan/TinyStories  (split={args.hf_split}) ...")
    ds = load_dataset(
        "roneneldan/TinyStories",
        split=args.hf_split,
        streaming=True,
    )

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    out_path = data_dir / "corpus.csv"
    temp_path = data_dir / "corpus.csv.tmp"

    n_stories = n_sents = n_tokens = 0
    completed = False

    try:
        with open(temp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["doc_id", "sentence", "tags"])

            for example in ds:
                if n_stories >= args.max_stories:
                    break

                text = (example.get("text") or "").strip()
                if not text:
                    continue

                # --- sentence splitting + tokenisation
                story_rows = []
                for raw_sent in split_sentences(text):
                    tokens = tokenize(raw_sent)
                    if len(tokens) >= args.min_sent_tokens:
                        story_rows.append(tokens)

                if not story_rows:
                    continue          # skip stories that yield no usable sentences

                # --- write one CSV row per sentence
                for tokens in story_rows:
                    writer.writerow([n_stories, " ".join(tokens), ""])

                n_stories += 1
                n_sents   += len(story_rows)
                n_tokens  += sum(len(r) for r in story_rows)

                if n_stories % 200 == 0:
                    print(
                        f"  {n_stories:>5}/{args.max_stories} stories | "
                        f"{n_sents:>6} sentences | {n_tokens:>8,} tokens"
                    )
        if n_stories == 0:
            raise RuntimeError("TinyStories stream contained no usable stories")
        os.replace(temp_path, out_path)
        completed = True
    finally:
        if not completed and temp_path.exists():
            temp_path.unlink()

    print(
        f"\nDone -- {n_stories} stories / {n_sents} sentences / {n_tokens:,} tokens"
    )
    print(f"Saved  ->  {out_path}")


if __name__ == "__main__":
    main()
