from pathlib import Path
import pandas as pd 
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer
from machine_learning_360v2.config import PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "synthetic_sme_agents_data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "health_score_features.csv",
):
    df = pd.read_csv(input_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")

    # 1: financial health (0-100)

    # profitability score (0-30)
    df['probability_score']=np.clip(
        df['profit_margin']*100*0.3,
        0, 30
    )

    # liquidity score
    total_liquid=df['bank_balance'] +df['m_pesa_balance']
    monthly_revenue = df['revenue'] / 12
    df['liquidity_months'] = total_liquid / monthly_revenue.replace(0, 1)
    df['liquidity_score'] = np.clip(
        df['liquidity_months']*5,
          0, 25)
    
    # debt management score
    df['debt_management_score'] = np.clip(
        (1 - df['debt_ratio'])*25,
        0, 25
    )

    # revenue stability score based on revenue growth potential and revenue size 
    df['revenue_stability_score'] = np.clip(
        (np.log1p(df['revenue'])/np.log1p(1_000_000))*20,
        0, 20
    )

    # financial health = sum of all scores
    df['financial_health_score']=(
        df['probability_score'] + 
        df['liquidity_score'] + 
        df['debt_management_score'] + 
        df['revenue_stability_score']
    )

    logger.info("financial health features created")

    # 2: operational efficiency (0-100)
    # employee productivity 

    df['revenue_per_employee'] = df['revenue'] /df['employee_count'].replace(0, 1)
    df['employee_productivity_score'] = np.clip(
        (df['revenue_per_employee']/50000)*30,
        0, 30   
    )

    # payroll efficiency
    df['payroll_ratio'] = df['total_payroll'] / df['revenue'].replace(0, 1)
    df['payroll_efficiency_score'] = np.clip(
        (1 - df['payroll_ratio']) * 25,
        0, 25
    )

    # invoice management
    df['invoice_completion_rate'] = df['paid_invoices']/(
        df['paid_invoices'] + df['pending_invoices']
    ).replace(0, 1)
    df['invoice_mgmt_score'] = df['invoice_completion_rate']*25
    
    # staff retention
    df['staff_retention_score'] = np.clip(
        (1 - df['staff_turnover_rate'])*20,
        0, 20
    )

    # operational efficiency = sum of all scores
    df['operational_efficiency_score'] = (
        df['employee_productivity_score'] + 
        df['payroll_efficiency_score'] + 
        df['invoice_mgmt_score'] + 
        df['staff_retention_score']
    )
    logger.info("operational efficiency features created")

    # 3: compliance health

    # documentation completeness
    doc_score = (
        df['business_registration_uploaded']*10 +
        df['tax_clearance_uploaded']*10 +
        df['financial_statements_uploaded']*10
    )
    df['documentation_score'] = doc_score

    # tax compliance
    df['tax_compliance_score'] = (
        df['tax_registered'] * 20 +
        df['tax_paid_last_year'] * 15
    )

    # license compliance
    df['license_compliance_score'] = df['licenses_up_to_date'] * 30
    
    # compliance health = sum of all scores
    df['compliance_health_score'] = (
        df['documentation_score'] + 
        df['tax_compliance_score'] + 
        df['license_compliance_score']
    )
    logger.info("compliance health features created")

    # 4: growth potential
    # market expansion based on customer acquisition and retention
    df['customer_growth_rate']=(
        df['active_customers']/df['total_customers'].replace(0, 1)
    )
    df['customer_retention_rate'] = (
        df['repeat_customers']/df['active_customers'].replace(0, 1)
    )
    df['market_expansion_score'] = np.clip(
        (df['customer_growth_rate']*15 + df['customer_retention_rate']*15),
        0, 30
    )
    # market effectiveness
    df['marketing_roi'] = df['revenue'] / df['marketing_spend'].replace(0, 1)
    df['marketing_conversion']=df['conversions']/df['clicks'].replace(0, 1)
    df['market_effectiveness_score'] = np.clip(
        (np.log1p(df['marketing_roi'])*10 + df['marketing_conversion']*100*0.15),
        0, 25
    )
    # business maturity 
    df['maturity_score'] = np.clip(
        (df['age_of_business']/10)*25,
        0, 25
    )
    # funding track record
    df['funding_track_score'] = (
        df['previous_funding_received']*10 +
        (1-df['default_history'])*10
    )

    # growth potential = sum of all scores
    df['growth_potential_score'] = (
        df['market_expansion_score'] + 
        df['market_effectiveness_score'] + 
        df['maturity_score'] +  
        df['funding_track_score']
    )
    logger.info("growth potential features created")

    # comprehnensive health score = weighted sum of all dimensions
    df['comprehensive_health_score'] = (
        df['financial_health_score']*0.35 + 
        df['operational_efficiency_score']*0.25 + 
        df['compliance_health_score']*0.20 + 
        df['growth_potential_score']*0.20
    )
    # normalize 
    df['comprehensive_health_score'] = np.clip(
        df['comprehensive_health_score'], 
        0, 100
        ).round(1)

    # health score categories
    df['health_category'] = pd.cut(
        df['comprehensive_health_score'],
        bins=[0,40,60, 80, 100],
        labels=['Critical', 'At Risk', 'Stable', 'Thriving']
    )
    logger.info("comprehensive health score calculated")

    # risk flags
    df['liquidity_risk']= (df['liquidity_months'] < 2).astype(int)
    df['debt_risk'] = (df['debt_ratio'] > 0.6).astype(int)
    df['profitability_risk'] = (df['profit_margin'] < 0.05).astype(int)
    df['compliance_risk'] = (df['compliance_health_score'] < 50).astype(int)
    df['growth_stagnation_risk'] = (df['growth_potential_score'] < 40).astype(int)

    df['total_risk_flags'] = (
        df['liquidity_risk'] +
        df['debt_risk'] +
        df['profitability_risk'] +
        df['compliance_risk'] +
        df['growth_stagnation_risk']
    )
    logger.info("risk flags created")

    # save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Health score features saved to {output_path}")
    logger.info(f"Final dataset shape: {df.shape}")
    logger.info(f"Health Score Stats:")
    logger.info(f"  Mean: {df['comprehensive_health_score'].mean():.1f}")
    logger.info(f"  Median: {df['comprehensive_health_score'].median():.1f}")
    logger.info(f"  Categories:\n{df['health_category'].value_counts()}")


if __name__ == "__main__":
    app()