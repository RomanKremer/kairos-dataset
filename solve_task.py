import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kairos import features
from kairos.data import DATA_DIR, load_followers, load_labels, load_posts
from kairos.label import author_level
from kairos.metrics import score

SEED = 42
RESULTS = Path(__file__).resolve().parent / "results"

NUMERIC = features.CONTENT_NUM + features.HISTORY_NUM + features.ARTEFACT_NUM
TXT = ["txt_below", "txt_typical", "txt_above"]
HISTORY = ["last1_rel", "last3_rel", "last5_rel", "hist_std", "hist_max_rel", "hist_min_rel",
           "trend5", "posts_30d", "len_vs_author", "same_type_rel", "same_type_n"]
TEXT_COLUMNS = ["text", "commentary", "content", "postText", "post_text", "body"]
PLACEHOLDER = re.compile(r"⟨([A-Za-z]+)(?:_\d+)?⟩")


def find_text_column(df):
    for c in TEXT_COLUMNS:
        if c in df.columns:
            return c
    raise SystemExit("no text column, columns: " + ", ".join(df.columns))


def clean(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        value = ""
    return PLACEHOLDER.sub(r" ph\1 ", str(value)).lower()


def text_model(fast):
    parts = [("word", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200000, sublinear_tf=True, dtype=np.float32))]
    if not fast:
        parts.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                              min_df=10, max_features=200000,
                                              sublinear_tf=True, dtype=np.float32)))
    return Pipeline([("tfidf", FeatureUnion(parts)), ("lr", LogisticRegression(C=0.5, max_iter=300, class_weight="balanced"))])


def text_probs(train, test, fast):
    x, y = train["clean_text"], train["label"].to_numpy()
    oof = np.zeros((len(train), 3))
    folds = GroupKFold(n_splits=5).split(x, y, groups=train["author"])
    for k, (fit, hold) in enumerate(folds):
        model = text_model(fast).fit(x.iloc[fit], y[fit])
        oof[hold] = model.predict_proba(x.iloc[hold])
        print("text fold", k + 1, "of 5", flush=True)
    full = text_model(fast).fit(x, y)
    return oof, full.predict_proba(test["clean_text"])


def add_history(df):
    df = df.copy()
    g = df.groupby("author")["log_reactions"]
    level = df["level"]

    df["last1_rel"] = g.shift(1) - level
    df["last3_rel"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()) - level
    df["last5_rel"] = g.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean()) - level
    df["hist_std"] = g.transform(lambda s: s.shift(1).expanding(min_periods=2).std())
    df["hist_max_rel"] = g.transform(lambda s: s.shift(1).expanding().max()) - level
    df["hist_min_rel"] = g.transform(lambda s: s.shift(1).expanding().min()) - level

    recent = g.transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    older = g.transform(lambda s: s.shift(6).rolling(5, min_periods=3).mean())
    df["trend5"] = recent - older

    secs = (df["posted_at"] - df["posted_at"].min()).dt.total_seconds()
    if not secs.groupby(df["author"]).apply(lambda s: s.is_monotonic_increasing).all():
        raise SystemExit("posts are not sorted by time")

    def count_30d(s):
        v = s.to_numpy()
        n = np.searchsorted(v, v, side="left") - np.searchsorted(v, v - 30 * 86400, side="left")
        return pd.Series(n, index=s.index)

    df["posts_30d"] = secs.groupby(df["author"]).transform(count_30d)

    mean_len = df.groupby("author")["n_chars"].transform(lambda s: s.shift(1).expanding().mean())
    df["len_vs_author"] = np.log1p(df["n_chars"]) - np.log1p(mean_len)

    key = [df["author"], df["content_type"].fillna("none")]
    same = df.groupby(key)["log_reactions"].transform(lambda s: s.shift(1).expanding().mean())
    df["same_type_rel"] = same - level
    df["same_type_n"] = df.groupby(key).cumcount()
    return df


def booster(numeric):
    prep = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), features.CONTENT_CAT),
    ])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=SEED)
    return Pipeline([("prep", prep), ("clf", clf)])


def macro_f1(y, pred):
    cm = np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    return float(np.mean(2 * prec * rec / np.maximum(prec + rec, 1e-12)))


def oof_proba(train, numeric):
    p = np.zeros((len(train), 3))
    y = train["label"]
    folds = GroupKFold(n_splits=5).split(train, y, groups=train["author"])
    for k, (fit, hold) in enumerate(folds):
        model = booster(numeric).fit(train.iloc[fit], y.iloc[fit])
        p[hold] = model.predict_proba(train.iloc[hold])
        print("booster fold", k + 1, "of 5", flush=True)
    return p


def tune_weights(y, proba):
    best = (macro_f1(y, proba.argmax(1)), 1.0, 1.0)
    grid = np.arange(0.2, 1.601, 0.05)
    for wb in grid:
        for wa in grid:
            f = macro_f1(y, (proba * np.array([wb, 1.0, wa])).argmax(1))
            if f > best[0] + 1e-9:
                best = (f, float(wb), float(wa))
    return best


def bootstrap(y, pred_a, pred_b, authors, n):
    groups = list(pd.DataFrame({"author": authors}).groupby("author").indices.values())
    rng = np.random.default_rng(SEED)
    rows = {"macro_f1": [], "qwk": []}
    for i in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        ra, rb = score(y[idx], pred_a[idx]), score(y[idx], pred_b[idx])
        for k in rows:
            rows[k].append((ra[k], rb[k]))
        if (i + 1) % 100 == 0:
            print("bootstrap", i + 1, "of", n, flush=True)
    fa, fb = score(y, pred_a), score(y, pred_b)
    print("\n%-9s %26s %26s %26s" % ("", "baseline", "final model", "difference"))
    out = {}
    for k, vals in rows.items():
        v = np.array(vals)
        la, ha = np.percentile(v[:, 0], [2.5, 97.5])
        lb, hb = np.percentile(v[:, 1], [2.5, 97.5])
        d = v[:, 1] - v[:, 0]
        ld, hd = np.percentile(d, [2.5, 97.5])
        print("%-9s %.4f [%.4f, %.4f]   %.4f [%.4f, %.4f]   %+.4f [%+.4f, %+.4f]" % (k, fa[k], la, ha, fb[k], lb, hb, fb[k] - fa[k], ld, hd))
        out[k] = {"baseline": [fa[k], la, ha], "final": [fb[k], lb, hb], "difference": [fb[k] - fa[k], ld, hd]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--fast", action="store_true", help="word n-grams only")
    ap.add_argument("--bootstrap", type=int, default=1000, help="0 = skip")
    args = ap.parse_args()
    t0 = time.time()

    labels = load_labels(args.data)
    levelled = author_level(load_posts(args.data))
    feats = add_history(features.build(levelled, load_followers(args.data)))
    table = labels.merge(feats, on=["author", "post"], how="left", validate="one_to_one")
    col = find_text_column(table)
    table["clean_text"] = table[col].map(clean)

    train = table[table["split"] == "train"].reset_index(drop=True)
    test = table[table["split"] == "test"].reset_index(drop=True)
    print("train", len(train), "posts, test", len(test), "posts", flush=True)

    oof, test_p = text_probs(train, test, args.fast)
    train[TXT], test[TXT] = oof, test_p
    results = {"text_only": score(test["label"], test_p.argmax(1))}

    variants = {
        "baseline": NUMERIC,
        "baseline+text": NUMERIC + TXT,
        "baseline+history": NUMERIC + HISTORY,
        "baseline+text+history": NUMERIC + TXT + HISTORY,
    }
    preds = {}
    for name, numeric in variants.items():
        print("fitting", name, flush=True)
        proba = booster(numeric).fit(train, train["label"]).predict_proba(test)
        preds[name] = proba.argmax(1)
        results[name] = score(test["label"], preds[name])
    final_proba = proba

    y_train = train["label"].to_numpy().astype(int)
    oof_p = oof_proba(train, variants["baseline+text+history"])
    f_plain = macro_f1(y_train, oof_p.argmax(1))
    f_best, w_below, w_above = tune_weights(y_train, oof_p)
    print("train out-of-fold macro-F1 %.4f -> %.4f, weights below=%.2f above=%.2f" % (f_plain, f_best, w_below, w_above))

    final = (final_proba * np.array([w_below, 1.0, w_above])).argmax(1)
    results["final (+class weights)"] = score(test["label"], final)

    print("\n%-26s %9s %7s %9s" % ("", "macro-F1", "QWK", "accuracy"))
    for name, m in results.items():
        print("%-26s %9.4f %7.4f %9.4f" % (name, m["macro_f1"], m["qwk"], m["accuracy"]))

    intervals = None
    if args.bootstrap > 0:
        intervals = bootstrap(test["label"].to_numpy().astype(int), preds["baseline"], final, test["author"], args.bootstrap)

    (RESULTS / "predictions").mkdir(parents=True, exist_ok=True)
    test[["author", "post"]].assign(label=final).to_csv(RESULTS / "predictions" / "solve_task.csv", index=False)
    info = {"seed": SEED,
            "fast": args.fast,
            "class_weights": {"below": w_below, "typical": 1.0, "above": w_above},
            "results": results,
            "bootstrap_95": intervals}
    (RESULTS / "solve_task.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("\ndone in %d s" % (time.time() - t0))


if __name__ == "__main__":
    main()
