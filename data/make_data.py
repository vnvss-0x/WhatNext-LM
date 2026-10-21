"""Compatibility entry point for building the TinyStories training corpus.

Run from the project root with ``python data/make_data.py``.  The implementation
lives in ``fetch_tinystories.py`` so the source is unambiguously the streamed
roneneldan/TinyStories dataset rather than the former synthetic grammar.
"""

from fetch_tinystories import main


if __name__ == "__main__":
    main()
