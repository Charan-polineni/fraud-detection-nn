"""Create a synthetic credit-card transaction dataset (no real or private data)."""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def generate(n_rows: int = 50_000, fraud_rate: float = 0.02) -> pd.DataFrame:
    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    legit = pd.DataFrame({
        "amount": RNG.lognormal(mean=3.5, sigma=0.9, size=n_legit),
        "hour": RNG.choice(range(24), size=n_legit,
                           p=np.array([1] * 6 + [3] * 14 + [2] * 4) / (6 + 42 + 8)),
        "distance_from_home_km": RNG.exponential(scale=8, size=n_legit),
        "transactions_last_24h": RNG.poisson(lam=2, size=n_legit),
        "is_online": RNG.binomial(1, 0.35, size=n_legit),
        "is_fraud": 0,
    })

    fraud = pd.DataFrame({
        "amount": RNG.lognormal(mean=5.0, sigma=1.0, size=n_fraud),
        "hour": RNG.choice([0, 1, 2, 3, 4, 22, 23], size=n_fraud),
        "distance_from_home_km": RNG.exponential(scale=120, size=n_fraud),
        "transactions_last_24h": RNG.poisson(lam=7, size=n_fraud),
        "is_online": RNG.binomial(1, 0.8, size=n_fraud),
        "is_fraud": 1,
    })

    df = pd.concat([legit, fraud], ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    data = generate()
    data.to_csv("data/transactions.csv", index=False)
    print(f"Saved {len(data):,} rows. Fraud rate: {data['is_fraud'].mean():.2%}")
