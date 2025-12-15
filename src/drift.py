import numpy as np
import pandas as pd

def psi(expected: pd.Series, actual: pd.Series, buckets=10) -> float:
    expected = expected.dropna()
    actual = actual.dropna()
    if expected.empty or actual.empty:
        return 0.0

    qs = np.linspace(0, 1, buckets + 1)
    cuts = expected.quantile(qs).values
    cuts = np.unique(cuts)
    if len(cuts) < 3:
        return 0.0

    e_counts = pd.cut(expected, bins=cuts, include_lowest=True).value_counts(normalize=True)
    a_counts = pd.cut(actual, bins=cuts, include_lowest=True).value_counts(normalize=True)

    # align bins
    e, a = e_counts.align(a_counts, fill_value=1e-6)
    e = e.replace(0, 1e-6)
    a = a.replace(0, 1e-6)

    return float(((a - e) * np.log(a / e)).sum())

def drift_report(train_df: pd.DataFrame, live_df: pd.DataFrame, top_n=10):
    rows = []
    common = [c for c in train_df.columns if c in live_df.columns]
    for c in common:
        if pd.api.types.is_numeric_dtype(train_df[c]) and pd.api.types.is_numeric_dtype(live_df[c]):
            rows.append({"feature": c, "psi": psi(train_df[c], live_df[c])})
    out = pd.DataFrame(rows).sort_values("psi", ascending=False)
    return out.head(top_n)
