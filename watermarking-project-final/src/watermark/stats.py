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

# Unique composite index identifying a single paired experimental trial
CASE = ["image", "attack", "severity"]


def _paired(df, a, b, metric):
    """Pivots the DataFrame to align paired observations of configs `a` and `b`.

    Extracts matched trials for the two configurations across identical host images,
    attack types, and severity levels to guarantee case-by-case comparability.

    Args:
        df (pd.DataFrame): Raw evaluation results containing config and metric columns.
        a (str): Primary configuration name (e.g., 'adaptive+ecc').
        b (str): Baseline/comparison configuration name (e.g., 'uniform+ecc').
        metric (str): Column name of the metric to compare (e.g., 'success', 'bit_acc').

    Returns:
        tuple[pd.Series, pd.Series]: Two aligned Series containing metric values for `a` and `b`.

    Raises:
        ValueError: If either configuration `a` or `b` is missing from the results.
    """
    sub = df[df.config.isin([a, b])]
    wide = sub.pivot_table(index=CASE, columns="config", values=metric, aggfunc="first").dropna()
    if a not in wide or b not in wide:
        raise ValueError(f"need both configs {a!r} and {b!r} in the results")
    return wide[a].astype(float), wide[b].astype(float)


def paired_bootstrap(df, a, b, metric="success", n_boot=2000, seed=0):
    """Calculate 95% bootstrap confidence interval for mean(metric | A) - mean(metric | B).

    Resamples whole images with replacement to respect the clustered/nested correlation
    structure (trials belonging to the same host image share texture and frequency properties).

    Args:
        df (pd.DataFrame): Evaluation results containing case trials.
        a (str): Focus configuration name.
        b (str): Control baseline configuration name.
        metric (str, optional): Target performance metric. Defaults to "success".
        n_boot (int, optional): Number of bootstrap resamples. Defaults to 2000.
        seed (int, optional): Deterministic random seed for resampling. Defaults to 0.

    Returns:
        dict: Summary statistics including observed difference (`diff`), 2.5% quantile (`lo`),
              97.5% quantile (`hi`), image count (`n_images`), and total cases (`n_cases`).
    """
    xa, xb = _paired(df, a, b, metric)
    # Aggregate metric differences per image cluster to handle intra-image correlation
    per_image = (xa - xb).groupby(level="image").agg(["sum", "count"])
    sums, counts = per_image["sum"].values, per_image["count"].values

    # Resample image indices with replacement
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))

    # Compute ratio of sums for each bootstrap replicate
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)

    return {
        "diff": float(sums.sum() / counts.sum()),
        "lo": float(np.percentile(boot, 2.5)),
        "hi": float(np.percentile(boot, 97.5)),
        "n_images": int(len(sums)),
        "n_cases": int(counts.sum()),
    }


def mcnemar(df, a, b, metric="success"):
    """Compute the exact two-sided McNemar test on discordant paired outcomes.

    Assesses whether the proportion of cases won only by config `a` differs significantly
    from the proportion won only by config `b` using an exact binomial test with p = 0.5.

    Args:
        df (pd.DataFrame): Evaluation results containing case trials.
        a (str): Focus configuration name.
        b (str): Control baseline configuration name.
        metric (str, optional): Binary success metric column name. Defaults to "success".

    Returns:
        dict: Counts of discordant trials (`a_only`, `b_only`) and the exact two-sided `p_value`.
    """
    xa, xb = _paired(df, a, b, metric)
    xa, xb = xa.astype(bool), xb.astype(bool)

    # Count discordant pairs (cases where only one configuration succeeded)
    a_only, b_only = int((xa & ~xb).sum()), int((~xa & xb).sum())
    n = a_only + b_only

    # Exact binomial test under null hypothesis p = 0.5 (equal failure/success rates)
    p = 1.0 if n == 0 else float(binomtest(a_only, n, 0.5).pvalue)
    return {"a_only": a_only, "b_only": b_only, "p_value": p}


def compare(df, pairs, metric="success", by=None):
    """Generate a structured comparative statistical table across multiple configuration pairs.

    Executes paired bootstrap estimation and exact McNemar testing for every designated
    configuration pair, optionally partitioning the analysis across subsets (e.g., severity).

    Args:
        df (pd.DataFrame): Evaluation results DataFrame.
        pairs (list[tuple[str, str]]): List of (config_A, config_B) tuples to evaluate.
        metric (str, optional): Metric column name to compare. Defaults to "success".
        by (str or list[str], optional): Optional column(s) to partition by before comparing.
            Defaults to None (evaluates all rows together under group 'all').

    Returns:
        pd.DataFrame: Tabular summary containing paired difference estimates, 95% CIs,
                      discordant counts, McNemar p-values, and statistical significance flags.
    """
    rows = []
    groups = [("all", df)] if by is None else list(df.groupby(by))

    for key, part in groups:
        for a, b in pairs:
            try:
                # Merge cluster bootstrap metrics and exact McNemar test results
                r = {**paired_bootstrap(part, a, b, metric), **mcnemar(part, a, b, metric)}
            except ValueError:
                # Skip pair if one of the configurations is absent in the subset
                continue

            # Flag as statistically significant at alpha = 0.05 if 95% CI excludes zero
            r.update({
                "A": a,
                "B": b,
                "group": key,
                "significant": bool(r["lo"] > 0 or r["hi"] < 0),
            })
            rows.append(r)

    cols = [
        "group", "A", "B", "diff", "lo", "hi", "a_only", "b_only",
        "p_value", "significant", "n_images", "n_cases",
    ]
    return pd.DataFrame(rows, columns=cols)
