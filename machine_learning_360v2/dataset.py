from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
import typer

from machine_learning_360v2.config import PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    output_path: Path = PROCESSED_DATA_DIR / "synthetic_sme_agents_data.csv",
):
    np.random.seed(42)
    n = 2000  
    # ------------------------
    # Base SME & Financial Info
    # ------------------------
    raw_revenue = np.random.lognormal(mean=-0.5, sigma=1.0, size=n)
    revenue = 30_000 + (raw_revenue - raw_revenue.min()) / (raw_revenue.max() - raw_revenue.min()) * (1_000_000 - 30_000)

    dataset = pd.DataFrame(
        {
            # --- SME identifiers ---
            "sme_id": range(1, n + 1),
            "business_name": [f"SME_{i}" for i in range(1, n + 1)],

            # --- Financials ---
            "revenue": revenue,  # total revenue
            "profit_margin": np.clip(np.random.normal(0.15, 0.07, n), 0, 1),
            "debt_ratio": np.clip(np.random.normal(0.3, 0.15, n), 0, 1),
            "collateral_value": np.random.lognormal(mean=8, sigma=1.5, size=n),
            "marketing_spend": np.random.lognormal(mean=5, sigma=0.8, size=n),
            "employee_count": np.random.randint(1, 100, size=n),
            "age_of_business": np.random.randint(1, 25, size=n),
            "sector": np.random.choice(
                ["agriculture", "retail", "tech", "manufacturing", "beauty", "transport", "services"],
                size=n,
                p=[0.2, 0.15, 0.15, 0.15, 0.1, 0.1, 0.15]
            ),

            # ------------------------
            # Funding Agent Inputs
            # ------------------------
            "previous_funding_received": np.random.binomial(1, 0.4, n),
            "previous_funding_amount": np.random.uniform(10_000, 500_000, n),
            "default_history": np.random.binomial(1, 0.1, n),
            "loan_applications_count": np.random.randint(0, 10, size=n),
            "funding_requested_amount": np.random.uniform(20_000, 1_000_000, n),
            "expected_roi": np.clip(np.random.normal(0.2, 0.1, n), 0, 2),
            "project_duration_months": np.random.randint(3, 36, n),
            "business_stage": np.random.choice(["startup", "growth", "mature"], size=n, p=[0.3,0.5,0.2]),
            "collateral_offered": np.random.choice([0,1], size=n, p=[0.3,0.7]),

            # Document uploads (for funding / compliance)
            "business_registration_uploaded": np.random.binomial(1, 0.9, n),
            "tax_clearance_uploaded": np.random.binomial(1, 0.85, n),
            "financial_statements_uploaded": np.random.binomial(1, 0.8, n),

            # ------------------------
            # Compliance Agent Inputs
            # ------------------------
            "tax_registered": np.random.binomial(1, 0.9, n),
            "tax_paid_last_year": np.random.binomial(1, 0.85, n),
            "licenses_required": np.random.randint(1, 5, n),
            "licenses_up_to_date": np.random.binomial(1, 0.8, n),

            # ------------------------
            # HR Agent Inputs
            # ------------------------
            "staff_count": np.random.randint(1, 50, n),
            "total_payroll": np.random.uniform(50_000, 500_000, n),
            "cost_per_hire": np.random.uniform(5_000, 50_000, n),
            "nssf_contribution_percent": np.random.uniform(0.05,0.12,n),
            "nhif_contribution_percent": np.random.uniform(0.02,0.05,n),
            "staff_turnover_rate": np.clip(np.random.normal(0.15, 0.05, n), 0, 1),

            # ------------------------
            # Finance Agent Inputs
            # ------------------------
            "bank_balance": np.random.uniform(5_000, 200_000, n),
            "m_pesa_balance": np.random.uniform(1_000, 50_000, n),
            "pending_invoices": np.random.randint(0, 20, n),
            "paid_invoices": np.random.randint(0, 50, n),

            # ------------------------
            # Sales Agent Inputs
            # ------------------------
            "total_customers": np.random.randint(50, 5000, n),
            "active_customers": np.random.randint(20, 4000, n),
            "repeat_customers": np.random.randint(10, 3000, n),
            "channels_used": np.random.choice(["physical","online","mobile"], size=n),

            # ------------------------
            # Marketing Agent Inputs
            # ------------------------
            "campaign_spend": np.random.uniform(1_000, 50_000, n),
            "clicks": np.random.randint(10, 5000, n),
            "impressions": np.random.randint(100, 100_000, n),
            "conversions": np.random.randint(0, 500, n),
            "target_segment": np.random.choice(["low","medium","high"], n),

            # ------------------------
            # Owner / Personal Info
            # ------------------------
            "owner_age": np.random.randint(20, 60, n),
            "owner_gender": np.random.choice(["male", "female"], n, p=[0.6, 0.4]),
            "education_level": np.random.choice(["none","highschool","bachelor","master","phd"], n),
            "employment_status": np.random.choice(["self-employed","employed"], n, p=[0.7,0.3]),
            "location": np.random.choice(["Nairobi","Mombasa","Kisumu","Eldoret","Other"], n)
        }
    )

    # ------------------------
    # TARGET VARIABLES (with noise and complexity)
    # ------------------------

    # 1. ELIGIBLE FOR FUNDING
    # Use complex conditions with some randomness
    eligibility_base = (
        (dataset['revenue'] > 100_000).astype(float) * 0.25 +
        (dataset['profit_margin'] > 0.12).astype(float) * 0.20 +
        (dataset['debt_ratio'] < 0.45).astype(float) * 0.15 +
        (dataset['default_history'] == 0).astype(float) * 0.20 +
        (dataset['tax_registered'] == 1).astype(float) * 0.10 +
        (dataset['licenses_up_to_date'] == 1).astype(float) * 0.10
    )
    # Add random noise to simulate real-world uncertainty
    eligibility_noise = np.random.normal(0, 0.15, n)
    eligibility_score = np.clip(eligibility_base + eligibility_noise, 0, 1)
    dataset['eligible_for_funding'] = (eligibility_score > 0.55).astype(int)

    # 2. DEFAULT RISK (low/medium/high)
    # Complex non-linear risk calculation
    risk_base = (
        dataset['debt_ratio'] ** 1.5 * 0.30 +  # non-linear
        dataset['default_history'] * 0.35 +
        (1 - dataset['profit_margin']) * 0.20 +
        dataset['staff_turnover_rate'] * 0.15
    )
    # Add sector-specific risk adjustments
    sector_risk = {'agriculture': 0.1, 'retail': 0.05, 'tech': -0.05, 
                   'manufacturing': 0, 'beauty': 0.08, 'transport': 0.12, 'services': 0.02}
    dataset['sector_risk_adj'] = dataset['sector'].map(sector_risk)
    risk_base += dataset['sector_risk_adj']

    # Random shocks (simulate unexpected events)
    random_shocks = np.random.choice([0, 0.2, -0.1], size=n, p=[0.85, 0.10, 0.05])
    risk_score = np.clip(risk_base + random_shocks + np.random.normal(0, 0.1, n), 0, 1)

    dataset['default_risk'] = pd.cut(risk_score, bins=[0, 0.33, 0.66, 1.0], 
                                      labels=['low', 'medium', 'high'])
    dataset.drop('sector_risk_adj', axis=1, inplace=True)  # clean up temp column

    # 3. BUSINESS HEALTH SCORE (0-100)
    # Multi-factor health with interactions
    financial_health = (
        dataset['profit_margin'] * 35 +
        (1 - dataset['debt_ratio']) * 25 +
        np.log1p(dataset['revenue']) / np.log1p(1_000_000) * 20
    )

    operational_health = (
        (dataset['age_of_business'] / 25) * 10 +
        (dataset['repeat_customers'] / dataset['total_customers'].clip(lower=1)) * 10
    )

    # Interaction effects (e.g., good marketing + good sales = bonus)
    marketing_efficiency = dataset['conversions'] / dataset['clicks'].clip(lower=1)
    interaction_bonus = np.where(
        (marketing_efficiency > 0.05) & (dataset['active_customers'] > 1000),
        5, 0
    )

    health_base = financial_health + operational_health + interaction_bonus

    # Add realistic noise and random events
    health_noise = np.random.normal(0, 8, n)
    random_events = np.random.choice([-15, -5, 0, 5, 10], size=n, 
                                      p=[0.05, 0.15, 0.60, 0.15, 0.05])

    dataset['business_health_score'] = np.clip(
        health_base + health_noise + random_events,
        0, 100
    ).round(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    logger.success(f"Synthetic SME agents dataset saved to {output_path}")


if __name__ == "__main__":
    app()