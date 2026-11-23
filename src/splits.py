"""Patient-level 10%/90% splitting on count-aggregated pseudo-IPD.

Each aggregated row = (covariate pattern, outcome y, count n). Sampling patients
uniformly at random is equivalent to multivariate-hypergeometric partitioning of
each row's count, so we never need to materialise millions of rows.

RESEARCH PROTOTYPE - synthetic_only=false (published-count reconstruction),
research_prototype=true. NOT for clinical / diagnostic / regulatory use.
"""
import numpy as np
import pandas as pd


def holdout_10_90(df, count_col="n", train_frac=0.10, seed=0):
    """Train on 10% of patients, validate on remaining 90%."""
    rng = np.random.default_rng(seed)
    tr = rng.binomial(df[count_col].to_numpy().astype(np.int64), train_frac)
    train = df.assign(**{count_col: tr})
    test = df.assign(**{count_col: df[count_col].to_numpy() - tr})
    return train[train[count_col] > 0], test[test[count_col] > 0]


def kfold_10pct(df, k=10, count_col="n", seed=0):
    """k disjoint folds, each a different random ~10% of patients (k=10 covers all).

    Yields (fold, train=that 10%, test=the other 90%)."""
    rng = np.random.default_rng(seed)
    counts = df[count_col].to_numpy().astype(np.int64)
    # each patient independently assigned to one of k folds -> disjoint random ~10% samples
    parts = np.stack([rng.multinomial(c, np.full(k, 1.0 / k)) for c in counts])
    for f in range(k):
        tr = df.assign(**{count_col: parts[:, f]})
        te = df.assign(**{count_col: counts - parts[:, f]})
        yield f, tr[tr[count_col] > 0], te[te[count_col] > 0]
