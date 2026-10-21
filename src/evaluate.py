"""Lab 02 - evaluation: held-out perplexity and top-3 accuracy.   Run:  python -m src.evaluate"""
import json
import os

from src.pipeline import (NGramModel, InterpolatedModel, KneserNeyModel,
                          load_documents, split_by_document,
                          top_k_accuracy, backoff_top_k, EOS, UNK)


def backoff_accuracy(tri, bi, uni, sentences, k=3):
    hits = total = 0
    for s in sentences:
        for i, w in enumerate(s + [EOS]):
            guess = [x for x, _ in backoff_top_k(tri, bi, uni, s[:i], k)[0]]
            total += 1
            hits += (w if w in tri.vocab else UNK) in guess
    return hits / total


def main():
    docs = load_documents()
    train, test = split_by_document(docs)
    n_tr = sum(len(s) for s in train); n_te = sum(len(s) for s in test)
    print(f"Split BY DOCUMENT (fixed): train = {len(train)} sentences / {n_tr} tokens, "
          f"test = {len(test)} sentences / {n_te} tokens")

    uni, bi, tri = NGramModel(train, 1, 1.0), NGramModel(train, 2, 1.0), NGramModel(train, 3, 1.0)
    tri01, tri0 = NGramModel(train, 3, 0.1), NGramModel(train, 3, 0.0)
    interp = InterpolatedModel(train, k=0.1)
    kn_bi  = KneserNeyModel(train, n=2)
    kn_tri = KneserNeyModel(train, n=3)
    print(f"  Kneser-Ney discount (auto-estimated): d={kn_tri.d:.4f}")

    rows = []
    def add(name, model):
        ppl, zero = model.perplexity(test)
        acc, _ = top_k_accuracy(model, test, 3)
        rows.append({"model": name, "perplexity": ppl, "zero_prob_share": zero, "top3": acc})

    add("unigram (k=1)", uni)
    add("bigram (k=1)", bi)
    add("trigram (k=1)", tri)
    add("trigram (k=0.1)", tri01)
    add("trigram unsmoothed (k=0)", tri0)
    add("interpolated 3/2/1-gram (k=0.1)", interp)
    add(f"Kneser-Ney bigram  (d={kn_bi.d:.3f})",  kn_bi)
    add(f"Kneser-Ney trigram (d={kn_tri.d:.3f})", kn_tri)

    print("\nmodel                               perplexity   zero-prob tokens   top-3 accuracy")
    for r in rows:
        ppl = "inf" if r["perplexity"] == float("inf") else f"{r['perplexity']:.2f}"
        print(f"{r['model']:<40}{ppl:>8}{r['zero_prob_share']:>16.1%}{r['top3']:>17.1%}")

    bo = backoff_accuracy(NGramModel(train, 3, 0.1), NGramModel(train, 2, 0.1), NGramModel(train, 1, 0.1), test)
    print(f"\nHard back-off (3->2->1-gram, k=0.1) top-3 accuracy: {bo:.1%}")

    # --- effect of the smoothing constant k  (add-k models)
    ks = [0.001, 0.01, 0.1, 0.5, 1, 2, 5]
    sweep = {n: [NGramModel(train, n, k).perplexity(test)[0] for k in ks] for n in (2, 3)}
    print("\nPerplexity vs k (add-k smoothing):")
    for n in (2, 3):
        print(f"  {n}-gram: " + ", ".join(f"k={k}: {p:.2f}" for k, p in zip(ks, sweep[n])))

    # --- effect of discount d  (Kneser-Ney)
    ds = [0.1, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
    kn_sweep = {n: [KneserNeyModel(train, n, d=dv).perplexity(test)[0] for dv in ds] for n in (2, 3)}
    print("\nPerplexity vs d (Kneser-Ney discount):")
    for n in (2, 3):
        print(f"  KN {n}-gram: " + ", ".join(f"d={dv}: {p:.2f}" for dv, p in zip(ds, kn_sweep[n])))

    # --- failure analysis
    _, misses = top_k_accuracy(NGramModel(train, 3, 0.1), test, 3)
    unseen_ctx = []
    t = NGramModel(train, 3, 0.1)
    for s in test:
        for i in range(len(s) + 1):
            if not t.seen_context(s[:i]):
                unseen_ctx.append(" ".join(s[max(0, i - 2):i]))
    os.makedirs("results", exist_ok=True)
    lines = ["Trigram (k=0.1) top-3 misses (context -> true word | suggestions):"]
    lines += [f"  '{c}' -> {w!r} | {g}" for c, w, g in misses[:6]]
    lines += [f"\nTest positions whose trigram context never appears in training: {len(unseen_ctx)}",
              f"  examples: {sorted(set(unseen_ctx))[:5]}"]
    with open("results/failures.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines[:5]))
    print(lines[-2]); print(lines[-1])

    with open("results/metrics.json", "w") as f:
        json.dump({"rows": rows, "backoff_top3": bo,
                   "k_sweep":  {"k":  ks,  **{str(n): sweep[n]    for n in sweep}},
                   "kn_sweep": {"d":  ds,  **{str(n): kn_sweep[n] for n in kn_sweep}},
                   "train_tokens": n_tr, "test_tokens": n_te,
                   "unseen_trigram_contexts": len(unseen_ctx)},
                  f, indent=2, default=str)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))

    # Panel 0: add-k perplexity vs k
    for n, c in ((2, "#8a8a8a"), (3, "#333333")):
        axes[0].plot(ks, sweep[n], marker="o", color=c, label=f"{n}-gram add-k")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("smoothing constant k (log scale)")
    axes[0].set_ylabel("test perplexity (lower = better)")
    axes[0].legend(); axes[0].set_title("Add-k: effect of k")

    # Panel 1: KN perplexity vs d
    for n, c in ((2, "#5b8dd9"), (3, "#1a3f7a")):
        axes[1].plot(ds, kn_sweep[n], marker="s", color=c, label=f"KN {n}-gram")
    axes[1].axvline(kn_tri.d, color="#1a3f7a", linestyle="--", linewidth=0.8,
                    label=f"auto d={kn_tri.d:.3f}")
    axes[1].set_xlabel("discount d")
    axes[1].set_ylabel("test perplexity (lower = better)")
    axes[1].legend(); axes[1].set_title("Kneser-Ney: effect of d")

    # Panel 2: top-3 accuracy bar chart (all finite-perplexity models)
    finite = [r for r in rows if r["perplexity"] != float("inf")]
    names  = [r["model"].replace(" (", "\n(") for r in finite]
    vals   = [r["top3"] for r in finite]
    colors = ["#1a3f7a" if "Kneser" in r["model"] else "#333333" for r in finite]
    axes[2].barh(names, vals, color=colors)
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("top-3 accuracy (held-out documents)")
    axes[2].tick_params(axis="y", labelsize=6.5)
    axes[2].set_title("Next-word suggestion quality")

    fig.tight_layout()
    fig.savefig("results/ngram_results.png", dpi=160)
    print("\nSaved results/metrics.json, results/failures.txt, results/ngram_results.png")


if __name__ == "__main__":
    main()
