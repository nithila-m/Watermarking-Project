"""Is adaptive really better than uniform, or is it noise?

Both tests are PAIRED: every config sees the same images, payloads and attacks,
so we compare config A and B case by case instead of comparing two averages.

  paired_bootstrap : 95% confidence interval for (success rate of A - success rate of B),
                     resampling whole IMAGES (cases from one image are not independent).
  mcnemar          : exact test on the cases where exactly one of A, B succeeded.

If the interval includes 0, report the methods as "not significantly different".
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest

CASE = ["image", "attack", "severity"]


def _paired(df, a, b, metric):
    sub = df[df.config.isin([a, b])]
    wide = sub.pivot_table(index=CASE, columns="config", values=metric, aggfunc="first").dropna()
    if a not in wide or b not in wide:
        raise ValueError(f"need both configs {a!r} and {b!r} in the results")
    return wide[a].astype(float), wide[b].astype(float)


def paired_bootstrap(df, a, b, metric="success", n_boot=2000, seed=0):
    """Return dict(diff, lo, hi, n_images, n_cases) for mean(metric | A) - mean(metric | B)."""
    xa, xb = _paired(df, a, b, metric)
    per_image = (xa - xb).groupby(level="image").agg(["sum", "count"])
    sums, counts = per_image["sum"].values, per_image["count"].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return {"diff": float(sums.sum() / counts.sum()),
            "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
            "n_images": int(len(sums)), "n_cases": int(counts.sum())}


def mcnemar(df, a, b, metric="success"):
    """Exact McNemar test. Returns dict(a_only, b_only, p_value)."""
    xa, xb = _paired(df, a, b, metric)
    xa, xb = xa.astype(bool), xb.astype(bool)
    a_only, b_only = int((xa & ~xb).sum()), int((~xa & xb).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(a_only, n, 0.5).pvalue)
    return {"a_only": a_only, "b_only": b_only, "p_value": p}


def compare(df, pairs, metric="success", by=None):
    """Table of paired comparisons. pairs: [(A, B), ...]. by: optional column to split on."""
    rows = []
    groups = [("all", df)] if by is None else list(df.groupby(by))
    for key, part in groups:
        for a, b in pairs:
            try:
                r = {**paired_bootstrap(part, a, b, metric), **mcnemar(part, a, b, metric)}
            except ValueError:
                continue
            r.update({"A": a, "B": b, "group": key,
                      "significant": bool(r["lo"] > 0 or r["hi"] < 0)})
            rows.append(r)
    cols = ["group", "A", "B", "diff", "lo", "hi", "a_only", "b_only", "p_value",
            "significant", "n_images", "n_cases"]
    return pd.DataFrame(rows, columns=cols)
