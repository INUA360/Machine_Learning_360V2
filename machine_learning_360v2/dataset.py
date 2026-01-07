from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
import typer

from machine_learning_360v2.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    output_path: Path = PROCESSED_DATA_DIR / "synthetic_onboarding_data.csv",
):
    np.random.seed(42)
    n = 2000

    # --- Revenue: realistic SME range (30k – 1M), right-skewed ---
    raw_revenue = np.random.lognormal(mean=-0.5, sigma=1.0, size=n)
    revenue = 30_000 + (raw_revenue - raw_revenue.min()) / (
        raw_revenue.max() - raw_revenue.min()
    ) * (1_000_000 - 30_000)

    dataset = pd.DataFrame(
        {
            "sme_id": range(1, n + 1),
            "revenue": revenue,
            "profit_margin": np.clip(np.random.normal(0.15, 0.07, n), 0, 1),
            "employee_count": np.random.randint(1, 100, size=n),
            "sector": np.random.choice(
                [
                    "agriculture",
                    "retail",
                    "tech",
                    "manufacturing",
                    "beauty",
                    "transport",
                    "services",
                ],
                size=n,
                p=[0.2, 0.15, 0.15, 0.15, 0.1, 0.1, 0.15],
            ),
            "age_of_business": np.random.randint(1, 25, size=n),
            "late_payments": np.random.binomial(1, 0.25, n),
            "debt_ratio": np.clip(np.random.normal(0.3, 0.15, n), 0, 1),
            "marketing_spend": np.random.lognormal(mean=5, sigma=0.8, size=n),
            "credit_history": np.random.randint(100, 850, size=n),
            "collateral_value": np.random.lognormal(mean=8, sigma=1.5, size=n),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    logger.success(f"Synthetic onboarding dataset saved to {output_path}")


if __name__ == "__main__":
    app()
