"""Lab 02 - Smart Keyboard V1 (Streamlit UI). Run:  streamlit run app.py"""
import streamlit as st

from src.pipeline import NGramModel, KneserNeyModel, InterpolatedModel, EOS, load_documents, backoff_top_k

st.set_page_config(page_title="Smart Keyboard V1", layout="wide")
st.title("⌨️ Smart Keyboard V1")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Model Settings")
    model_choice = st.radio(
        "Language model",
        ["Kneser-Ney trigram (recommended)", "Trigram back-off (add-k)", "Interpolated 3/2/1-gram"],
    )
    top_n = st.slider("Number of suggestions", min_value=1, max_value=5, value=3)
    st.divider()
    st.caption(
        "**Add-k**: simple Laplace-style smoothing.  \n"
        "**Kneser-Ney**: uses *continuation counts* — how many unique "
        "contexts a word appears in — which better estimates rare events.  \n"
        "**Interpolated**: weighted sum of tri/bi/unigram add-k models."
    )


# ------------------------------------------------------------------ model cache
@st.cache_resource
def get_all_models():
    docs = load_documents()
    sentences = [s for d in docs.values() for s in d]
    tri   = NGramModel(sentences, 3, 0.1)
    bi    = NGramModel(sentences, 2, 0.1)
    uni   = NGramModel(sentences, 1, 0.1)
    kn    = KneserNeyModel(sentences, n=3)
    interp = InterpolatedModel(sentences, k=0.1)
    return tri, bi, uni, kn, interp


with st.spinner("Building language models from TinyStories corpus …"):
    tri, bi, uni, kn_model, interp_model = get_all_models()

# ------------------------------------------------------------------ input
text  = st.text_input("Type the beginning of a sentence:", "once upon a time")
words = text.lower().replace(".", " . ").split()
unknown = [w for w in words if w not in uni.vocab]

if unknown:
    st.warning(f"Not in the training vocabulary (treated as \u003cunk\u003e): {', '.join(unknown)}")
if not words:
    st.info("Type at least one word — showing how sentences usually start.")

# ------------------------------------------------------------------ predict
if model_choice == "Trigram back-off (add-k)":
    suggestions, order = backoff_top_k(tri, bi, uni, words, top_n)
    caption = f"Trigram hard back-off · used the **{order}** model · add-k smoothing (k = 0.1)"
elif model_choice == "Kneser-Ney trigram (recommended)":
    suggestions = kn_model.top_k(words, top_n)
    caption = (
        f"Interpolated Kneser-Ney · trigram · "
        f"auto-estimated discount **d = {kn_model.d:.3f}**"
    )
else:
    suggestions = interp_model.top_k(words, top_n)
    caption = "Interpolated 3/2/1-gram · λ = (0.6, 0.3, 0.1) · add-k (k = 0.1)"

# ------------------------------------------------------------------ display
st.subheader(f"Top-{top_n} suggestions")
st.caption(caption)
cols = st.columns(top_n)
for col, (w, p) in zip(cols, suggestions):
    label = "(end of sentence)" if w == EOS else w
    col.metric(label, f"{p:.1%}")

st.divider()
st.caption(
    "Probabilities are model estimates on the streamed TinyStories corpus — not real-world typing statistics. "
    "Kneser-Ney typically achieves lower perplexity than add-k smoothing on held-out data."
)
