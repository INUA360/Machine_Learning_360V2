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
    
    historical_growth = np.clip(base_growth, -0.30, 0.50) # Realistic range: -30% (decline) to +50% (rapid growth)
    
    df['revenue_6m_ago'] = df['revenue'] / (1 + historical_growth)
    df['customers_6m_ago'] = (df['total_customers'] / (1 + np.abs(historical_growth) * 0.5)).round()
    df['marketing_spend_6m_ago'] = df['marketing_spend'] / (1 + np.random.normal(0, 0.2, len(df)))
    
    logger.info("Historical data generated for training")
    
    # Momentum indicators
    df['revenue_growth_6m'] = (df['revenue'] - df['revenue_6m_ago']) / df['revenue_6m_ago'].replace(0, 1) # Calculate: % change in revenue over last 6 months
    # Example: revenue $100k today, $80k 6 months ago → growth=25%
    df['revenue_velocity'] = df['revenue_growth_6m'] / 6
    # Example: 25% growth over 6 months → 4.2% per month velocity

    df['customer_growth_6m'] = (df['total_customers'] - df['customers_6m_ago']) / df['customers_6m_ago'].replace(0, 1) # Calculate: % change in customer count

    df['customer_velocity'] = df['customer_growth_6m'] / 6
    # Per-month customer acquisition rate

    df['customer_retention_rate'] = df['repeat_customers'] / df['active_customers'].replace(0, 1)
    # Repeat customers / total active = % returning customers
    # Higher = better (loyal customer base)

    df['customer_activation_rate'] = df['active_customers'] / df['total_customers'].replace(0, 1)
    # Active / total = % of all customers who are currently active
    # Higher = healthier engagement
    df['customer_acquisition_rate'] = (df['total_customers'] - df['customers_6m_ago']) / 6 # New customers per month
    
    logger.info("Momentum indicators created")

    # Capacity scores
    # Measure: "Does this business have resources to grow?"
    total_liquid = df['bank_balance'] + df['m_pesa_balance']
    monthly_burn = df['revenue'] / 12
    df['cash_runway_months'] = total_liquid / monthly_burn.replace(0, 1)
    # How many months can the business operate with current cash?
    # Example: $10k cash, $5k/month burn → 2 months runway
    df['debt_capacity'] = np.clip(1 - df['debt_ratio'], 0, 1)
    # How much more debt can business take?
    # Example: debt_ratio=30% → capacity=70% (can borrow more)
    df['reinvestment_capacity'] = df['profit_margin'] * df['revenue']
    # How much profit can be reinvested in growth?
    # Example: $100k revenue × 15% margin = $15k to reinvest
    
    df['revenue_per_employee'] = df['revenue'] / df['employee_count'].replace(0, 1)
    df['operational_leverage'] = np.clip(df['revenue_per_employee'] / 50_000, 0, 5)
    # Normalized efficiency (capped at 5x)
    # Example: $100k/employee → 2x leverage (twice ideal)
    df['staff_stability'] = 1 - df['staff_turnover_rate']
    # Retention rate (1 = perfect retention, 0 = complete turnover)
    
    df['customer_base_growth'] = (df['total_customers'] - df['customers_6m_ago']) / df['customers_6m_ago'].replace(0, 1)
    # % increase in customer base
    df['market_penetration_velocity'] = df['customer_base_growth'] / 6
    #Per-month customer addition rate
    
    logger.info("Capacity scores created")

    # Growth enablers
    df['marketing_roi'] = df['revenue'] / df['marketing_spend'].replace(0, 1) # Revenue per marketing dollar spent
    df['marketing_efficiency'] = np.clip(np.log1p(df['marketing_roi']) / 10, 0, 1)
    # What % of revenue is spent on marketing?
    # Example: $100k revenue, $20k marketing → 20% intensity
    df['marketing_intensity'] = df['marketing_spend'] / df['revenue'].replace(0, 1)
    
    df['conversion_rate'] = df['conversions'] / df['clicks'].replace(0, 1)     # % of visitors who buy

    df['marketing_growth_potential'] = np.clip(df['marketing_roi'] * df['marketing_intensity'], 0, 1)
    # Combined: "If we increase marketing spend, how much growth?"

    df['has_funding_track'] = df['previous_funding_received'].astype(int)
    df['funding_growth_boost'] = np.where(
        df['previous_funding_received'] == 1,
        np.clip(df['previous_funding_amount'] / df['revenue'].replace(0, 1), 0, 1),
        0
    )
    # If received funding before, boost = (funding amount / current revenue)
# Example: raised $50k when revenue was $100k → 0.5x boost potential
    logger.info("Growth enablers created")

    # Composite growth score
    df['composite_growth_score'] = np.clip(
        (df['revenue_velocity'] * 6) * 20 +  # 20 points for revenue momentum
        (df['customer_velocity'] * 6) * 15 +  # 15 points for customer growth
        np.clip(df['cash_runway_months'] / 6, 0, 1) * 15 +  # 15 points for cash runway
        df['debt_capacity'] * 15 + # 15 points for borrowing capacity
        np.clip(df['operational_leverage'] / 2, 0, 1) * 15 + # 15 points for efficiency
        df['marketing_efficiency'] * 20, # 20 points for marketing ROI
        0, 100
    ).round(1)
    
    # Interpretation:
    # 0-20: Very limited growth potential
    # 20-40: Slow growth trajectory
    # 40-60: Moderate growth potential
    # 60-80: Strong growth prospects
    # 80-100: Excellent growth potential
    logger.info("Composite growth score calculated")

    # Future growth prediction (REALISTIC DISTRIBUTION)
    
    # Start with historical momentum
    momentum_factor = np.clip(df['revenue_growth_6m'], -0.30, 0.50) #Assume future growth = past growth (with bounds)
    
    # Business health adjustments
    health_boost = np.where(
        (df['profit_margin'] > 0.10) & (df['debt_ratio'] < 0.4),
        0.05, # Healthy business gets +5% boost
        np.where(
            df['profit_margin'] < 0,
            -0.10,  # Loss-making, gets a -10% penalty
            0
        )
    )
    
    # Stage-based expectations  Age-based expectations (startups grow faster than mature companies)
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
    market_noise = np.random.normal(0, 0.05, len(df)) #Market randomness (sometimes luck matters)
    
    # Final prediction: mean reversion toward realistic growth
    raw_prediction = (
        momentum_factor * 0.40 + # 40% weight on past momentum
        health_boost +
        stage_factor +
        market_noise
    )
    
    # Mean reversion: pull extreme predictions toward realistic center
    mean_growth = 0.10
    df['predicted_6m_growth_rate'] = (
        raw_prediction * 0.70 + mean_growth * 0.30
    )

    # Problem: Raw formula could predict 200% growth (unrealistic)
    # Solution: Pull extreme predictions toward realistic center (10% growth)
    
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

    # Interpretation:
    # < -5%: Declining (business shrinking)
    # -5% to 5%: Stable (flat growth)
    # 5% to 15%: Growing (healthy growth)
    # > 15%: Rapid Growth (accelerating)
        
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

    # OUTPUT example:
    # Declining:      120 SMEs (6%)
    # Stable:         800 SMEs (40%)
    # Growing:        980 SMEs (49%)
    # Rapid Growth:   100 SMEs (5%)

# Example: 340 SMEs will jump to larger revenue bracket (17%)