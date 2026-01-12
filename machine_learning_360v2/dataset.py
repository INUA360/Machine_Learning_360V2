from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
import typer

from machine_learning_360v2.config import PROCESSED_DATA_DIR

app = typer.Typer()


# ==============================
# CONFIG
# ==============================
N_SAMPLES = 2000
SEED = 42


# ==============================
# GENERATION
# ==============================
def generate_base_data(n: int) -> pd.DataFrame:
    np.random.seed(SEED)

    raw_revenue = np.random.lognormal(mean=-0.5, sigma=1.0, size=n)
    revenue = 30_000 + (raw_revenue - raw_revenue.min()) / (
        raw_revenue.max() - raw_revenue.min()
    ) * (1_000_000 - 30_000)

    df = pd.DataFrame({
        "sme_id": range(1, n + 1),
        "business_name": [f"SME_{i}" for i in range(1, n + 1)],

        # Financials
        "revenue": revenue,
        "profit_margin": np.clip(np.random.normal(0.15, 0.07, n), 0, 1),
        "debt_ratio": np.clip(np.random.normal(0.3, 0.15, n), 0, 1),

        # Structure
        "employee_count": np.random.randint(1, 100, n),
        "staff_count": np.random.randint(0, 50, n),
        "age_of_business": np.random.randint(1, 25, n),
        "sector": np.random.choice(
            ["agriculture", "retail", "tech", "manufacturing", "beauty", "transport", "services"],
            size=n,
            p=[0.2, 0.15, 0.15, 0.15, 0.1, 0.1, 0.15]
        ),

        # Funding history
        "previous_funding_received": np.random.binomial(1, 0.4, n),
        "loan_applications_count": np.random.randint(0, 10, n),

        # Funding request
        "funding_requested_amount": np.random.uniform(20_000, 1_000_000, n),
        "project_duration_months": np.random.randint(3, 36, n),
        "business_stage": np.random.choice(["startup", "growth", "mature"], n, p=[0.3, 0.5, 0.2]),
        "collateral_offered": np.random.binomial(1, 0.7, n),

        # Compliance
        "tax_registered": np.random.binomial(1, 0.9, n),
        "tax_paid_last_year": np.random.binomial(1, 0.85, n),
        "licenses_required": np.random.randint(1, 5, n),
        "licenses_up_to_date": np.random.binomial(1, 0.8, n),

        # Finance
        "bank_balance": np.random.uniform(5_000, 200_000, n),
        "m_pesa_balance": np.random.uniform(1_000, 50_000, n),
        "pending_invoices": np.random.randint(0, 20, n),
        "paid_invoices": np.random.randint(0, 50, n),

        # Marketing
        "campaign_spend": np.random.uniform(0, 50_000, n),
        "clicks": np.random.randint(0, 5000, n),
        "impressions": np.random.randint(0, 100_000, n),
        "conversions": np.random.randint(0, 500, n),

        # Customers
        "total_customers": np.random.randint(50, 5000, n),
        "active_customers": np.random.randint(20, 4000, n),
        "repeat_customers": np.random.randint(10, 3000, n),

        # Owner
        "owner_age": np.random.randint(20, 60, n),
        "owner_gender": np.random.choice(["male", "female"], n, p=[0.6, 0.4]),
        "education_level": np.random.choice(["none", "highschool", "bachelor", "master", "phd"], n),
        "employment_status": np.random.choice(["self-employed", "employed"], n, p=[0.7, 0.3]),
        "location": np.random.choice(["Nairobi", "Mombasa", "Kisumu", "Eldoret", "Other"], n)
    })

    return df


# ==============================
# CONSTRAINT LAYER (CRITICAL)
# ==============================
def apply_constraints(df: pd.DataFrame) -> pd.DataFrame:

    # --- Funding dependencies ---
    df["previous_funding_amount"] = np.where(
        df["previous_funding_received"] == 1,
        np.random.uniform(10_000, 500_000, len(df)),
        0.0
    )

    df["default_history"] = np.where(
        df["previous_funding_received"] == 1,
        np.random.binomial(1, 0.1, len(df)),
        0
    )

    # --- Collateral ---
    df["collateral_value"] = np.where(
        df["collateral_offered"] == 1,
        np.random.lognormal(mean=8, sigma=1.5, size=len(df)),
        0.0
    )

    # --- Payroll ---
    df["total_payroll"] = np.where(
        df["staff_count"] > 0,
        np.random.uniform(50_000, 500_000, len(df)),
        0.0
    )

    # --- Marketing ---
    df.loc[df["campaign_spend"] == 0, ["clicks", "impressions", "conversions"]] = 0

    # --- Customers ---
    df["active_customers"] = np.minimum(df["active_customers"], df["total_customers"])
    df["repeat_customers"] = np.minimum(df["repeat_customers"], df["active_customers"])

    return df


# ==============================
# TARGET GENERATION
# ==============================
def generate_targets(df: pd.DataFrame) -> pd.DataFrame:

    eligibility_score = (
        (df["revenue"] > 100_000).astype(int) * 0.3 +
        (df["profit_margin"] > 0.12).astype(int) * 0.2 +
        (df["debt_ratio"] < 0.45).astype(int) * 0.2 +
        (df["default_history"] == 0).astype(int) * 0.2 +
        (df["tax_registered"] == 1).astype(int) * 0.1
    ) + np.random.normal(0, 0.1, len(df))

    df["eligible_for_funding"] = (eligibility_score > 0.55).astype(int)

    risk_score = (
        df["debt_ratio"] * 0.4 +
        df["default_history"] * 0.4 +
        df["profit_margin"].rsub(1) * 0.2
    ) + np.random.normal(0, 0.1, len(df))

    df["default_risk"] = pd.cut(
        np.clip(risk_score, 0, 1),
        bins=[0, 0.33, 0.66, 1],
        labels=["low", "medium", "high"]
    )

    df["business_health_score"] = np.clip(
        (df["profit_margin"] * 40 +
         (1 - df["debt_ratio"]) * 30 +
         np.log1p(df["revenue"]) / np.log1p(1_000_000) * 30)
        + np.random.normal(0, 8, len(df)),
        0, 100
    ).round(1)

    return df


# ==============================
# CLI ENTRYPOINT
# ==============================
@app.command()
def main(
    output_path: Path = PROCESSED_DATA_DIR / "synthetic_sme_agents_data.csv"
):
    df = generate_base_data(N_SAMPLES)
    df = apply_constraints(df)
    df = generate_targets(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.success(f"Synthetic SME dataset generated → {output_path}")


if __name__ == "__main__":
    app()
