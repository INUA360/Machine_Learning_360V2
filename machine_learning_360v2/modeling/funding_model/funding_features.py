# feature_engineering.py
from machine_learning_360v2.config import PROCESSED_DATA_DIR
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
import typer

app = typer.Typer()

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "synthetic_sme_agents_data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "synthetic_onboarding_features.csv"
):

    df = pd.read_csv(input_path)
    logger.info(f"Loaded data: {df.shape}")

     
    df['log_revenue'] = np.log1p(df['revenue'])
    df['profit_per_employee'] = df['revenue'] / df['employee_count'].replace(0, 1)
    df['late_payment_rate'] = df['late_payments'] / df['age_of_business'].replace(0, 1)
    df['debt_profit_ratio'] = df['debt_ratio'] / df['profit_margin'].replace(0, np.nan)
    df['marketing_efficiency'] = df['revenue'] / df['marketing_spend'].replace(0, 1)
    df['funding_request_ratio'] = df['funding_requested_amount'] / df['revenue'].replace(0, 1)
    
 
    df['customer_retention_rate'] = df['repeat_customers'] / df['active_customers'].replace(0, 1)
    df['invoice_completion_rate'] = df['paid_invoices'] / (df['paid_invoices'] + df['pending_invoices']).replace(0, 1)
    df['liquidity_ratio'] = (df['bank_balance'] + df['m_pesa_balance']) / df['revenue'].replace(0, 1)
    df['payroll_to_revenue'] = df['total_payroll'] / df['revenue'].replace(0, 1)
    df['marketing_conversion_rate'] = df['conversions'] / df['clicks'].replace(0, 1)

    logger.info("Created numerical features")

    df['profitability'] = pd.cut(
        df['profit_margin'],
        bins=[-np.inf, 0, 0.15, np.inf],
        labels=['Loss-making', 'Thin Margin', 'Healthy Margin']
    )

    df['financial_health'] = pd.cut(
        df['debt_profit_ratio'],
        bins=[-np.inf, 1, 3, np.inf],
        labels=['Stable', 'Manageable', 'Risky']
    )

    df['payment_behavior'] = pd.cut(
        df['late_payment_rate'],
        bins=[-np.inf, 0.5, 2, np.inf],
        labels=['Reliable', 'Occasional', 'Problematic']
    )

    df['operational_efficiency'] = pd.cut(
        df['profit_per_employee'],
        bins=[-np.inf, 50000, 100000, np.inf],
        labels=['Lean', 'Balanced', 'High-performing']
    )

    df['marketing_roi'] = pd.cut(
        df['marketing_efficiency'],
        bins=[-np.inf, 5, 15, np.inf],
        labels=['Low ROI', 'Moderate ROI', 'High ROI']
    )

    df['business_maturity'] = pd.cut(
        df['age_of_business'],
        bins=[0, 2, 5, np.inf],
        labels=['Startup', 'Growth', 'Mature']
    )

    df['customer_base_size'] = pd.cut(
        df['total_customers'],
        bins=[0, 500, 2000, np.inf],
        labels=['Small', 'Medium', 'Large']
    )

    df['funding_size_category'] = pd.cut(
        df['funding_requested_amount'],
        bins=[0, 100000, 500000, np.inf],
        labels=['Small', 'Medium', 'Large']
    )

    logger.info("Created categorical features")

      
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        logger.warning(f"Removing {null_count} null values from dataset")
        df = df.dropna()

   
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Saved engineered features to {output_path}")
    logger.info(f"Final shape: {df.shape}")


if __name__ == "__main__":
    app()
