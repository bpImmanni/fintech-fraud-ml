import numpy as np
import pandas as pd

def simulate(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "user_id": rng.integers(1, 800, size=n),
        "merchant_id": rng.integers(1, 200, size=n),
        "device_id": rng.integers(1, 1200, size=n),
        "hour": rng.integers(0, 24, size=n),
        "amount": np.round(rng.lognormal(mean=3.5, sigma=0.7, size=n), 2),
        "is_international": rng.integers(0, 2, size=n),
        "is_new_device": rng.integers(0, 2, size=n),
    })

    # velocity proxy (not time-series perfect, but good demo)
    df["user_txn_count_24h"] = rng.poisson(lam=2.0, size=n) + (df["is_new_device"] * rng.integers(0, 5, size=n))
    df["merchant_txn_count_1h"] = rng.poisson(lam=5.0, size=n)

    # create a synthetic fraud label for demo (optional)
    score = (
        0.02 * df["amount"] +
        1.5 * df["is_international"] +
        1.0 * df["is_new_device"] +
        0.4 * df["user_txn_count_24h"] +
        0.2 * df["merchant_txn_count_1h"] +
        rng.normal(0, 1.0, size=n)
    )
    prob = 1 / (1 + np.exp(-(score - 7.0)))
    df["Class"] = (rng.random(n) < prob).astype(int)
    return df

if __name__ == "__main__":
    df = simulate()
    df.to_csv("data/processed/simulated_transactions.csv", index=False)
    print("Saved data/processed/simulated_transactions.csv")
