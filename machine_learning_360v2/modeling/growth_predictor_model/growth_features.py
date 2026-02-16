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
    OUTPUT_PATH: Path = PROCESSED_DATA_DIR / "growth_predictor_features.csv"
):
    df = pd.read_csv(INPUT_PATH)
    logger.info(f"Loaded data: {df.shape}")

    
    df['current_revenue_category'] = pd.cut(
        df['revenue'],
        bins=[0, 100_000, 500_000, np.inf],
        labels=['Small', 'Medium', 'Large']
    )
    
    logger.info(" Revenue categories created")
    logger.info(f" small: {(df['current_revenue_category'] == 'Small').sum()}businesses")
    logger.info(f" medium: {(df['current_revenue_category'] == 'Medium').sum()} businesses")
    logger.info(f" large: {(df['current_revenue_category'] == 'Large').sum()} businesses")


    
    # Business age momentum (young businesses grow faster)
    df['age_momentum'] = np.where(
        df['age_of_business'] < 3, 1.5,  
        np.where(df['age_of_business'] < 7, 1.2,  
                 1.0)
    )
    logger.info(" Age momentum calculated")
    
    # Customer acquisition momentum
    df['customer_acquisition_rate'] = (
        df['active_customers'] / df['total_customers'].replace(0, 1)
    )
    df['customer_retention_power'] = (
        df['repeat_customers'] / df['active_customers'].replace(0, 1)
    )
    
    # Combined customer momentum
    df['customer_momentum'] = (
        df['customer_acquisition_rate'] * 0.6 + 
        df['customer_retention_power'] * 0.4
    )
    
    logger.info(" Momentum indicators created")

    # Financial capacity for growth
    df['cash_runway_months'] = (
        (df['bank_balance'] + df['m_pesa_balance']) / 
        (df['revenue'] / 12).replace(0, 1)
    )
    
    df['debt_headroom'] = np.clip(1 - df['debt_ratio'], 0, 1)
    
    df['financial_capacity_score'] = (
        np.clip(df['cash_runway_months'] / 6, 0, 1) * 0.5 +  # 6 months = ideal
        df['debt_headroom'] * 0.5
    ) * 100
    
    # Operational capacity
    df['employee_utilization'] = np.where(
        df['employee_count'] > 0,
        df['revenue'] / (df['employee_count'] * 50_000),  # 50K per employee baseline
        0
    )
    
    df['operational_capacity_score'] = np.clip(
        (2 - df['employee_utilization']) * 50,  # Room to grow if < 2x baseline
        0, 100
    )
    
    # Market capacity (room to expand customer base)
    df['market_penetration'] = np.clip(
        df['total_customers'] / 10_000,  # Assume 10K max market
        0, 1
    )
    df['market_capacity_score'] = (1 - df['market_penetration']) * 100
    
    logger.info("✓ Capacity features created")

    
    # Marketing investment index
    df['marketing_to_revenue_ratio'] = (
        df['marketing_spend'] / df['revenue'].replace(0, 1)
    )
    df['marketing_efficiency'] = (
        df['conversions'] / df['clicks'].replace(0, 1)
    )
    df['marketing_growth_potential'] = (
        np.clip(df['marketing_to_revenue_ratio'] * 10, 0, 1) * 0.5 +
        np.clip(df['marketing_efficiency'], 0, 1) * 0.5
    ) * 100
    
    # Funding access indicator
    df['has_funding_access'] = (
        (df['previous_funding_received'] == 1) & 
        (df['default_history'] == 0)
    ).astype(int)
    
    df['funding_growth_boost'] = np.where(
        df['has_funding_access'] == 1,
        1.3,  # 30% boost if has good funding history
        np.where(df['eligible_for_funding'] == 1, 1.15, 1.0)  # 15% if eligible
    )
    
    # Profitability enabler (profitable businesses can reinvest)
    df['reinvestment_capacity'] = np.clip(
        df['profit_margin'] * 100,
        0, 30  # Max 30 points
    )
    
    logger.info("✓ Growth enabler features created")


    
    # Sector growth multipliers (Kenya-specific)
    sector_growth_map = {
        'tech': 1.4,           # High growth
        'agriculture': 1.2,    # Moderate-high
        'services': 1.15,      # Moderate
        'retail': 1.1,         # Moderate
        'manufacturing': 1.1,  # Moderate
        'transport': 1.05,     # Stable
        'beauty': 1.05         # Stable
    }
    df['sector_growth_factor'] = df['sector'].map(sector_growth_map).fillna(1.0)
    
    # Location growth factors
    location_growth_map = {
        'Nairobi': 1.3,   # Highest growth
        'Mombasa': 1.15,
        'Kisumu': 1.1,
        'Eldoret': 1.1,
        'Other': 1.0
    }
    df['location_growth_factor'] = df['location'].map(location_growth_map).fillna(1.0)
    
    logger.info("✓ External growth factors created")

    
    df['composite_growth_score'] = (
        df['financial_capacity_score'] * 0.25 +
        df['operational_capacity_score'] * 0.20 +
        df['market_capacity_score'] * 0.15 +
        df['marketing_growth_potential'] * 0.15 +
        df['reinvestment_capacity'] * 0.15 +
        (df['customer_momentum'] * 100) * 0.10
    )
    
    # Adjust by external factors
    df['composite_growth_score'] = (
        df['composite_growth_score'] * 
        df['sector_growth_factor'] * 
        df['location_growth_factor'] *
        df['age_momentum'] *
        df['funding_growth_boost']
    )
    
    # Normalize to 0-100
    df['composite_growth_score'] = np.clip(
        df['composite_growth_score'] / 2,  # Scale down from boosted values
        0, 100
    ).round(1)
    
    logger.info("✓ Composite growth score calculated")

    
    # Simulate 6-month growth rate based on growth score
    base_growth = df['composite_growth_score'] / 200  # Convert to decimal
    noise = np.random.normal(0, 0.05, len(df))  # Add realistic variance
    
    df['predicted_6m_growth_rate'] = np.clip(
        base_growth + noise,
        -0.2,  # Max 20% decline
        0.5    # Max 50% growth
    ).round(3)
    
    # Calculate predicted revenue
    df['predicted_6m_revenue'] = df['revenue'] * (1 + df['predicted_6m_growth_rate'])
    
    # Predicted revenue category
    df['predicted_revenue_category'] = pd.cut(
        df['predicted_6m_revenue'],
        bins=[0, 100_000, 500_000, np.inf],
        labels=['Small', 'Medium', 'Large']
    )
    
    # Category jump indicator
    df['will_jump_category'] = (
        df['predicted_revenue_category'] != df['current_revenue_category']
    ).astype(int)
    
    logger.info("✓ Growth targets created")

    
    df['growth_stage'] = pd.cut(
        df['predicted_6m_growth_rate'],
        bins=[-1, 0, 0.1, 0.25, 1],
        labels=['Declining', 'Stable', 'Growing', 'Rapid Growth']
    )
    
    logger.info("✓ Growth stages classified")

    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    
    logger.success(f" Growth predictor features saved to {OUTPUT_PATH}")
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"\nGrowth Rate Distribution:")
    logger.info(f"  Mean: {df['predicted_6m_growth_rate'].mean():.1%}")
    logger.info(f"  Median: {df['predicted_6m_growth_rate'].median():.1%}")
    logger.info(f"\nGrowth Stages:\n{df['growth_stage'].value_counts()}")
    logger.info(f"\nCategory Jumps: {df['will_jump_category'].sum()} SMEs ({df['will_jump_category'].mean():.1%})")


if __name__ == "__main__":
    app()