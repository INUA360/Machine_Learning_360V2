from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import typer
from machine_learning_360v2.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    INPUT_PATH: Path = PROCESSED_DATA_DIR / "synthetic_sme_agents_data.csv",
    OUTPUT_PATH: Path = PROCESSED_DATA_DIR / "health_score_features.csv"

):
    df = pd.read_csv(INPUT_PATH)
    logger.info(f"Loaded data: {df.shape}")

    # Financial ratios
    df['liquidity_ratio'] = (df['bank_balance'] + df['m_pesa_balance']) / df['revenue'].replace(0, 1)
    df['cash_months'] = df['liquidity_ratio'] * 12
    df['revenue_per_employee'] = df['revenue'] / df['employee_count'].replace(0, 1)
    
    # Operational ratios
    df['payroll_to_revenue'] = df['total_payroll'] / df['revenue'].replace(0, 1)
    df['invoice_paid_rate'] = df['paid_invoices'] / (df['paid_invoices'] + df['pending_invoices']).replace(0, 1)
    df['staff_retention_rate'] = 1 - df['staff_turnover_rate']
    
    # Customer ratios
    df['customer_active_rate'] = df['active_customers'] / df['total_customers'].replace(0, 1)
    df['customer_repeat_rate'] = df['repeat_customers'] / df['active_customers'].replace(0, 1)
    
    # Marketing ratios
    df['marketing_roi'] = df['revenue'] / df['marketing_spend'].replace(0, 1)
    df['conversion_rate'] = df['conversions'] / df['clicks'].replace(0, 1)
    
    logger.info("✓ Ratios calculated")

        # Risk flags 
    
    df['low_cash'] = (df['cash_months'] < 2).astype(int)
    df['high_debt'] = (df['debt_ratio'] > 0.6).astype(int)
    df['low_profit'] = (df['profit_margin'] < 0.05).astype(int)
    df['late_payer'] = (df['late_payments'] > df['age_of_business']).astype(int)
    
    # Compliance flags
    df['missing_docs'] = (
        (df['business_registration_uploaded'] == 0) |
        (df['tax_clearance_uploaded'] == 0) |
        (df['financial_statements_uploaded'] == 0)
    ).astype(int)
    
    df['tax_noncompliant'] = (
        (df['tax_registered'] == 0) |
        (df['tax_paid_last_year'] == 0)
    ).astype(int)
    
    df['license_expired'] = (df['licenses_up_to_date'] == 0).astype(int)
    
    logger.info(" Risk flags created")
    
    # risk flags 
    
    np.random.seed(42)  
    
    # Calculate base health probability (0 to 1)
    base_health = (
        (df['profit_margin'] / 0.30) * 0.20 +           # Profitability weight
        (df['cash_months'] / 12) * 0.20 +                # Liquidity weight
        ((1 - df['debt_ratio']) / 1.0) * 0.15 +          # Debt management
        (df['invoice_paid_rate']) * 0.10 +               # Operations
        (df['customer_repeat_rate']) * 0.10 +            # Customer health
        ((df['revenue'] / 500_000).clip(0, 1)) * 0.10 +  # Scale
        (1 - df['missing_docs']) * 0.08 +                # Compliance
        (1 - df['tax_noncompliant']) * 0.07              # Tax compliance
    )
    
    # Clip to 0-1 range
    base_health = np.clip(base_health, 0, 1)
    
    # Add realistic noise (market conditions, luck, external factors)
    noise = np.random.normal(0, 0.15, len(df))
    health_probability = np.clip(base_health + noise, 0, 1)
    
    # Convert probability to category 
    def prob_to_category(prob):
        """
        Convert health probability to category with randomness.
        
        Even high-probability businesses can fail (bad luck)
        Even low-probability businesses can survive (good luck)
        """
        # Base category from probability
        if prob < 0.25:
            base_cat = 'Critical'
        elif prob < 0.50:
            base_cat = 'At Risk'
        elif prob < 0.75:
            base_cat = 'Stable'
        else:
            base_cat = 'Thriving'
        
        # Add 10% chance of "surprise" outcome (randomness!)
        if np.random.random() < 0.10:
            # Random jump to adjacent category
            categories = ['Critical', 'At Risk', 'Stable', 'Thriving']
            current_idx = categories.index(base_cat)
            
            # Move up or down one category randomly
            if current_idx == 0:
                return np.random.choice(['Critical', 'At Risk'])
            elif current_idx == 3:
                return np.random.choice(['Stable', 'Thriving'])
            else:
                return np.random.choice([
                    categories[current_idx - 1],
                    categories[current_idx],
                    categories[current_idx + 1]
                ])
        
        return base_cat
    
    df['health_category'] = [prob_to_category(p) for p in health_probability]
    
    logger.info("✓ Health categories labeled (with randomness)")
    logger.info(f"\nCategory distribution:\n{df['health_category'].value_counts()}")
    
    # Also save the probability )
    df['health_probability'] = health_probability
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    
    logger.success(f"Saved to {OUTPUT_PATH}")
    logger.info(f"Final shape: {df.shape}")


if __name__ == "__main__":
    app()