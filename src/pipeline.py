"""Lab 02 - Smart Keyboard V1: n-gram language models (no UI code here).

Models implemented
------------------
NGramModel         – add-k (Laplace-style) smoothing
                     P(w|ctx) = (C(ctx,w) + k) / (C(ctx) + k*V)
InterpolatedModel  – linear interpolation of tri/bi/unigram add-k models
KneserNeyModel     – interpolated Kneser-Ney smoothing (Chen & Goodman 1999)
                     Uses continuation counts at lower orders and an
                     auto-estimated absolute discount d = n1/(n1+2*n2).
"""
import csv
import math
from collections import Counter, defaultdict

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


# ----------------------------------------------------------------- data
def load_documents(path="data/corpus.csv"):
    """Return {doc_id: [sentence_tokens, ...]} (tokens are already lower-case, space separated)."""
    docs = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            docs[int(row["doc_id"])].append(row["sentence"].split())
    return dict(docs)


def split_by_document(docs, test_fraction=0.2):
    """Fixed, reproducible split BY DOCUMENT (the last 20 % of documents are the test set)."""
    ids = sorted(docs)
    cut = int(len(ids) * (1 - test_fraction))
    train = [s for i in ids[:cut] for s in docs[i]]
    test = [s for i in ids[cut:] for s in docs[i]]
    return train, test


# ----------------------------------------------------------------- model
class NGramModel:
    def __init__(self, sentences, n=3, k=1.0):
        self.n, self.k = n, k
        self.vocab = {w for s in sentences for w in s} | {EOS, UNK}   # BOS is never predicted
        self.V = len(self.vocab)
        self.ctx_counts = defaultdict(Counter)       # context tuple -> Counter(next word)
        for s in sentences:
            padded = [BOS] * (n - 1) + s + [EOS]
            for i in range(n - 1, len(padded)):
                self.ctx_counts[tuple(padded[i - n + 1:i])][padded[i]] += 1
        self.ctx_totals = {c: sum(cnt.values()) for c, cnt in self.ctx_counts.items()}

    # -- helpers
    def clean(self, words):
        """Map out-of-vocabulary words to <unk> (so the model never crashes on new words)."""
        return [w if w in self.vocab else UNK for w in words]

    def context(self, words):
        words = [BOS] * (self.n - 1) + self.clean(words)
        return tuple(words[-(self.n - 1):]) if self.n > 1 else ()

    def prob(self, ctx, word):
        counts = self.ctx_counts.get(ctx)
        total = self.ctx_totals.get(ctx, 0)
        num = (counts[word] if counts else 0) + self.k
        den = total + self.k * self.V
        return num / den if den > 0 else 0.0         # k = 0 and unseen context -> probability 0

    def next_word_distribution(self, words):
        ctx = self.context(words)
        return {w: self.prob(ctx, w) for w in self.vocab if w != UNK}

    def top_k(self, words, k=3):
        dist = self.next_word_distribution(words)
        return sorted(dist.items(), key=lambda x: (-x[1], x[0]))[:k]

    def seen_context(self, words):
        return self.context(words) in self.ctx_counts

    # -- evaluation
    def perplexity(self, sentences):
        """exp( - mean log P(token) ) over every token of every sentence (EOS included).
        Returns (perplexity, share_of_zero_probability_tokens)."""
        log_sum, n_tok, zeros = 0.0, 0, 0
        for s in sentences:
            words = self.clean(s)
            padded = [BOS] * (self.n - 1) + words + [EOS]
            for i in range(self.n - 1, len(padded)):
                p = self.prob(tuple(padded[i - self.n + 1:i]), padded[i])
                n_tok += 1
                if p <= 0:
                    zeros += 1
                else:
                    log_sum += math.log(p)
        if zeros:
            return float("inf"), zeros / n_tok
        return math.exp(-log_sum / n_tok), 0.0


class InterpolatedModel:
    """p = l3 * p_trigram + l2 * p_bigram + l1 * p_unigram  (a smoother alternative to hard back-off)."""

    def __init__(self, sentences, k=0.1, lambdas=(0.6, 0.3, 0.1)):
        self.models = [NGramModel(sentences, n, k) for n in (3, 2, 1)]
        self.lambdas = lambdas
        self.vocab, self.V, self.n = self.models[0].vocab, self.models[0].V, 3

    def clean(self, words):
        return self.models[0].clean(words)

    def prob_words(self, history, word):
        history = [BOS, BOS] + history
        p = 0.0
        for lam, m in zip(self.lambdas, self.models):
            ctx = tuple(history[len(history) - (m.n - 1):]) if m.n > 1 else ()
            p += lam * m.prob(ctx, word)
        return p

    def top_k(self, words, k=3):
        hist = self.clean(words)
        dist = {w: self.prob_words(hist, w) for w in self.vocab if w != UNK}
        return sorted(dist.items(), key=lambda x: (-x[1], x[0]))[:k]

    def perplexity(self, sentences):
        log_sum, n_tok = 0.0, 0
        for s in sentences:
            words = self.clean(s)
            for i, w in enumerate(words + [EOS]):
                log_sum += math.log(self.prob_words(words[:i], w))
                n_tok += 1
        return math.exp(-log_sum / n_tok), 0.0


# ----------------------------------------------------------------- Kneser-Ney
class KneserNeyModel:
    """Interpolated Kneser-Ney language model (Chen & Goodman 1999).

    For the top-order n  (raw counts):
        P_KN(w | ctx) = max(C(ctx,w) - d, 0) / C(ctx)  +  λ(ctx) · P_KN(w | ctx[1:])

    For intermediate orders  1 < m < n  (continuation counts):
        P_KN(w | ctx) = max(C_cont(ctx,w) - d, 0) / ΣC_cont(ctx,·)  +  λ(ctx) · P_KN(w | ctx[1:])

    For the unigram level  (m = 1):
        P_KN(w) = C_cont(w) / Σ_w' C_cont(w')
        where C_cont(w) = |{v : bigram (v, w) seen in training}|

    The discount d is estimated automatically:
        d = n1 / (n1 + 2·n2),  n1 = # top-order n-grams with count 1,
                                n2 = # top-order n-grams with count 2.
    Supply d explicitly to override (e.g. KneserNeyModel(sents, n=3, d=0.75)).
    """

    def __init__(self, sentences, n=3, d=None):
        if n < 2:
            raise ValueError("KneserNeyModel requires n >= 2 (use NGramModel for unigrams).")
        self.n = n
        self.vocab = {w for s in sentences for w in s} | {EOS, UNK}
        self.V = len(self.vocab)

        # ---- Step 1: raw n-gram counts for every order 1 … n
        # raw[order] = { ctx_tuple (len order-1) : Counter(next_word) }
        raw = {}
        for order in range(1, n + 1):
            raw[order] = defaultdict(Counter)
            for s in sentences:
                padded = [BOS] * (order - 1) + s + [EOS]
                for i in range(order - 1, len(padded)):
                    ctx = tuple(padded[i - order + 1: i])
                    raw[order][ctx][padded[i]] += 1

        # ---- Step 2: top-order uses actual counts
        self.top_counts = raw[n]          # {ctx (len n-1) : Counter}
        self.top_totals = {ctx: sum(c.values()) for ctx, c in self.top_counts.items()}
        self.top_unique  = {ctx: len(c)           for ctx, c in self.top_counts.items()}

        # ---- Step 3: continuation counts for orders 1 … n-1
        #
        # cont[m][ctx_m][w]  =  |{ v : (v,) + ctx_m + (w,) seen as an (m+1)-gram }|
        # where ctx_m has length m-1.
        #
        # Built from raw[m+1]: each entry raw[m+1][ctx_m1][w] has ctx_m1 of length m.
        #   left    = ctx_m1[0]    – the incoming word we are counting unique values of
        #   right_ctx = ctx_m1[1:] – the context for this order (length m-1)
        self.cont        = {}   # m -> defaultdict(Counter)
        self.cont_totals = {}   # m -> {ctx : int}
        self.cont_unique = {}   # m -> {ctx : int}  (used to compute λ)

        for m in range(1, n):
            self.cont[m] = defaultdict(Counter)
            for ctx_m1, word_cnt in raw[m + 1].items():
                right_ctx = ctx_m1[1:]          # strip the leftmost word
                for w in word_cnt:              # only unique left contexts count
                    self.cont[m][right_ctx][w] += 1
            self.cont_totals[m] = {ctx: sum(c.values()) for ctx, c in self.cont[m].items()}
            self.cont_unique[m]  = {ctx: len(c)           for ctx, c in self.cont[m].items()}

        # ---- Step 4: estimate discount d from top-order singleton/doubleton counts
        if d is None:
            all_top = [c for cc in self.top_counts.values() for c in cc.values()]
            n1 = sum(1 for c in all_top if c == 1)
            n2 = sum(1 for c in all_top if c == 2)
            self.d = n1 / (n1 + 2 * n2) if n2 > 0 else 0.75
        else:
            self.d = float(d)

    # ---------------------------------------------------------------- helpers
    def clean(self, words):
        """Map OOV words to <unk>."""
        return [w if w in self.vocab else UNK for w in words]

    def context(self, words):
        words = [BOS] * (self.n - 1) + self.clean(words)
        return tuple(words[-(self.n - 1):])

    # ---------------------------------------------------------------- core
    def _prob(self, order, ctx, word):
        """Recursively compute interpolated KN probability for 'word' given 'ctx'."""
        if order == 1:
            # Unigram level: pure continuation probability
            total = self.cont_totals.get(1, {}).get((), 0)
            count = self.cont.get(1, {}).get((), Counter()).get(word, 0)
            return (count / total) if total > 0 else (1.0 / self.V)

        d = self.d
        if order == self.n:
            # Top order: raw counts
            count  = self.top_counts.get(ctx, Counter()).get(word, 0)
            total  = self.top_totals.get(ctx, 0)
            unique = self.top_unique.get(ctx, 0)
        else:
            # Intermediate order: continuation counts
            count  = self.cont[order].get(ctx, Counter()).get(word, 0)
            total  = self.cont_totals[order].get(ctx, 0)
            unique = self.cont_unique[order].get(ctx, 0)

        if total == 0:
            # Context never seen: delegate entirely to lower order
            return self._prob(order - 1, ctx[1:], word)

        lam   = (d * unique) / total          # probability mass for back-off
        lower = self._prob(order - 1, ctx[1:], word)
        return max(count - d, 0.0) / total + lam * lower

    def prob(self, ctx, word):
        """P_KN(word | ctx)  –  ctx is the last n-1 words as a tuple."""
        return self._prob(self.n, ctx, word)

    # ---------------------------------------------------------------- prediction
    def next_word_distribution(self, words):
        ctx = self.context(words)
        return {w: self.prob(ctx, w) for w in self.vocab if w != UNK}

    def top_k(self, words, k=3):
        dist = self.next_word_distribution(words)
        return sorted(dist.items(), key=lambda x: (-x[1], x[0]))[:k]

    def seen_context(self, words):
        return self.context(words) in self.top_counts

    # ---------------------------------------------------------------- evaluation
    def perplexity(self, sentences):
        """exp(-mean log P) over all tokens (EOS included).
        Returns (perplexity, share_of_zero_probability_tokens).
        KN with n >= 2 should never produce zero probabilities."""
        log_sum, n_tok, zeros = 0.0, 0, 0
        for s in sentences:
            words  = self.clean(s)
            padded = [BOS] * (self.n - 1) + words + [EOS]
            for i in range(self.n - 1, len(padded)):
                ctx = tuple(padded[i - self.n + 1: i])
                p   = self.prob(ctx, padded[i])
                n_tok += 1
                if p <= 0:
                    zeros += 1
                else:
                    log_sum += math.log(p)
        if zeros:
            return float("inf"), zeros / n_tok
        return math.exp(-log_sum / n_tok), 0.0


def backoff_top_k(tri, bi, uni, words, k=3):
    """Hard back-off for suggestions: use the highest order whose context was actually seen."""
    for model in (tri, bi, uni):
        if model.seen_context(words) or model.n == 1:
            return model.top_k(words, k), f"{model.n}-gram"
    return [], "none"


def top_k_accuracy(model, sentences, k=3):
    """Share of positions where the true next word is among the k suggestions."""
    hits = total = 0
    failures = []
    for s in sentences:
        for i, true_w in enumerate(s + [EOS]):
            guess = [w for w, _ in model.top_k(s[:i], k)]
            truth = true_w if true_w in model.vocab else UNK
            total += 1
            if truth in guess:
                hits += 1
            elif len(failures) < 200:
                failures.append((" ".join(s[:i]) or "<start>", true_w, guess))
    return hits / total, failures


if __name__ == "__main__":
    docs = load_documents()
    train, test = split_by_document(docs)

    m_addk = NGramModel(train, n=3, k=0.1)
    m_kn   = KneserNeyModel(train, n=3)

    print("=== after 'the girls' ===")
    print(f"  add-k  (k=0.1) : {m_addk.top_k(['the', 'girls'])}")
    print(f"  Kneser-Ney (d={m_kn.d:.3f}) : {m_kn.top_k(['the', 'girls'])}")

    print("\n=== test perplexity ===")
    print(f"  add-k  : {m_addk.perplexity(test)[0]:.2f}")
    print(f"  KN     : {m_kn.perplexity(test)[0]:.2f}")
