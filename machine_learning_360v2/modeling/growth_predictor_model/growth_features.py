from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import typer
from machine_learning_360v2.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "synthetic_sme_agents_data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "growth_predictor_features.csv"
):
    df = pd.read_csv(input_path)
    logger.info(f"Loaded data: {df.shape}")

    np.random.seed(42)
    
    # Generate realistic historical growth
    base_growth = np.where(
        df['age_of_business'] < 3,
        np.random.normal(0.15, 0.10, len(df)),
        np.where(
            df['age_of_business'] < 7,
            np.random.normal(0.10, 0.08, len(df)),
            np.random.normal(0.05, 0.05, len(df))
        )
    )
    
    historical_growth = np.clip(base_growth, -0.30, 0.50)
    
    df['revenue_6m_ago'] = df['revenue'] / (1 + historical_growth)
    df['customers_6m_ago'] = (df['total_customers'] / (1 + np.abs(historical_growth) * 0.5)).round()
    df['marketing_spend_6m_ago'] = df['marketing_spend'] / (1 + np.random.normal(0, 0.2, len(df)))
    
    logger.info("Historical data generated for training")
    
    # Momentum indicators
    df['revenue_growth_6m'] = (df['revenue'] - df['revenue_6m_ago']) / df['revenue_6m_ago'].replace(0, 1)
    df['revenue_velocity'] = df['revenue_growth_6m'] / 6
    
    df['customer_growth_6m'] = (df['total_customers'] - df['customers_6m_ago']) / df['customers_6m_ago'].replace(0, 1)
    df['customer_velocity'] = df['customer_growth_6m'] / 6
    
    df['customer_retention_rate'] = df['repeat_customers'] / df['active_customers'].replace(0, 1)
    df['customer_activation_rate'] = df['active_customers'] / df['total_customers'].replace(0, 1)
    df['customer_acquisition_rate'] = (df['total_customers'] - df['customers_6m_ago']) / 6
    
    logger.info("Momentum indicators created")

    # Capacity scores
    total_liquid = df['bank_balance'] + df['m_pesa_balance']
    monthly_burn = df['revenue'] / 12
    df['cash_runway_months'] = total_liquid / monthly_burn.replace(0, 1)
    df['debt_capacity'] = np.clip(1 - df['debt_ratio'], 0, 1)
    df['reinvestment_capacity'] = df['profit_margin'] * df['revenue']
    
    df['revenue_per_employee'] = df['revenue'] / df['employee_count'].replace(0, 1)
    df['operational_leverage'] = np.clip(df['revenue_per_employee'] / 50_000, 0, 5)
    df['staff_stability'] = 1 - df['staff_turnover_rate']
    
    df['customer_base_growth'] = (df['total_customers'] - df['customers_6m_ago']) / df['customers_6m_ago'].replace(0, 1)
    df['market_penetration_velocity'] = df['customer_base_growth'] / 6
    
    logger.info("Capacity scores created")

    # Growth enablers
    df['marketing_roi'] = df['revenue'] / df['marketing_spend'].replace(0, 1)
    df['marketing_efficiency'] = np.clip(np.log1p(df['marketing_roi']) / 10, 0, 1)
    df['marketing_intensity'] = df['marketing_spend'] / df['revenue'].replace(0, 1)
    df['conversion_rate'] = df['conversions'] / df['clicks'].replace(0, 1)
    df['marketing_growth_potential'] = np.clip(df['marketing_roi'] * df['marketing_intensity'], 0, 1)
    
    df['has_funding_track'] = df['previous_funding_received'].astype(int)
    df['funding_growth_boost'] = np.where(
        df['previous_funding_received'] == 1,
        np.clip(df['previous_funding_amount'] / df['revenue'].replace(0, 1), 0, 1),
        0
    )
    
    logger.info("Growth enablers created")

    # Composite growth score
    df['composite_growth_score'] = np.clip(
        (df['revenue_velocity'] * 6) * 20 +
        (df['customer_velocity'] * 6) * 15 +
        np.clip(df['cash_runway_months'] / 6, 0, 1) * 15 +
        df['debt_capacity'] * 15 +
        np.clip(df['operational_leverage'] / 2, 0, 1) * 15 +
        df['marketing_efficiency'] * 20,
        0, 100
    ).round(1)
    
    logger.info("Composite growth score calculated")

    # Future growth prediction (REALISTIC DISTRIBUTION)
    
    # Start with historical momentum
    momentum_factor = np.clip(df['revenue_growth_6m'], -0.30, 0.50)
    
    # Business health adjustments
    health_boost = np.where(
        (df['profit_margin'] > 0.10) & (df['debt_ratio'] < 0.4),
        0.05,
        np.where(
            df['profit_margin'] < 0,
            -0.10,
            0
        )
    )
    
    # Stage-based expectations
    stage_factor = np.where(
        df['age_of_business'] < 3,
        np.random.normal(0.03, 0.08, len(df)),
        np.where(
            df['age_of_business'] < 7,
            np.random.normal(0.02, 0.06, len(df)),
            np.random.normal(0, 0.04, len(df))
        )
    )
    
    # Market noise
    market_noise = np.random.normal(0, 0.05, len(df))
    
    # Final prediction: mean reversion toward realistic growth
    raw_prediction = (
        momentum_factor * 0.40 +
        health_boost +
        stage_factor +
        market_noise
    )
    
    # Mean reversion: pull extreme predictions toward realistic center
    mean_growth = 0.10
    df['predicted_6m_growth_rate'] = (
        raw_prediction * 0.70 + mean_growth * 0.30
    )
    
    # Clip to realistic range
    df['predicted_6m_growth_rate'] = np.clip(
        df['predicted_6m_growth_rate'],
        -0.30,
        0.40
    ).round(3)
    
    df['predicted_6m_revenue'] = df['revenue'] * (1 + df['predicted_6m_growth_rate'])
    
    # Revenue categories
    df['current_revenue_category'] = pd.cut(
        df['revenue'],
        bins=[0, 100_000, 500_000, np.inf],
        labels=['Small', 'Medium', 'Large']
    )
    
    df['predicted_revenue_category'] = pd.cut(
        df['predicted_6m_revenue'],
        bins=[0, 100_000, 500_000, np.inf],
        labels=['Small', 'Medium', 'Large']
    )
    
    df['will_jump_category'] = (
        df['predicted_revenue_category'] != df['current_revenue_category']
    ).astype(int)
    
    # Growth stage (REALISTIC BINS)
    df['growth_stage'] = pd.cut(
        df['predicted_6m_growth_rate'],
        bins=[-1, -0.05, 0.05, 0.15, 1],
        labels=['Declining', 'Stable', 'Growing', 'Rapid Growth']
    )
    
    logger.info("Growth targets created")
    logger.info(f"\nGrowth Statistics:")
    logger.info(f"  Mean: {df['predicted_6m_growth_rate'].mean():.1%}")
    logger.info(f"  Median: {df['predicted_6m_growth_rate'].median():.1%}")
    logger.info(f"  Std Dev: {df['predicted_6m_growth_rate'].std():.1%}")
    logger.info(f"  Min: {df['predicted_6m_growth_rate'].min():.1%}")
    logger.info(f"  Max: {df['predicted_6m_growth_rate'].max():.1%}")
    logger.info(f"\nGrowth Stages:")
    logger.info(f"\n{df['growth_stage'].value_counts().sort_index()}")
    logger.info(f"\nCategory Jumps: {df['will_jump_category'].sum()} SMEs ({df['will_jump_category'].mean():.1%})")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.success(f"Features saved to {output_path}")
    logger.info(f"Final shape: {df.shape}")


if __name__ == "__main__":
    app()