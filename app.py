"""Lab 02 - Smart Keyboard V1 (Streamlit UI). Run:  streamlit run app.py"""
import streamlit as st

from src.pipeline import NGramModel, EOS, load_documents, backoff_top_k

st.set_page_config(page_title="Smart Keyboard V1")
st.title("Smart Keyboard V1")
st.caption("Next-word suggestions from a trigram model with add-k smoothing and back-off.")


@st.cache_resource
def get_models():
    docs = load_documents()
    sentences = [s for d in docs.values() for s in d]
    return [NGramModel(sentences, n, 0.1) for n in (3, 2, 1)], sentences


(tri, bi, uni), sentences = get_models()

text = st.text_input("Type the beginning of a sentence:", "the happy girls")
words = text.lower().replace(".", " . ").split()
unknown = [w for w in words if w not in uni.vocab]

if unknown:
    st.warning(f"Not in the training vocabulary (treated as <unk>): {', '.join(unknown)}")
if not words:
    st.info("Type at least one word - showing how sentences usually start.")

suggestions, order = backoff_top_k(tri, bi, uni, words, 3)
st.subheader(f"Top-3 suggestions  ·  used the {order} model")
cols = st.columns(3)
for col, (w, p) in zip(cols, suggestions):
    label = "(end of sentence)" if w == EOS else w
    col.metric(label, f"{p:.1%}")
st.caption("Probabilities are model estimates on a tiny synthetic corpus - not real-world typing statistics.")
