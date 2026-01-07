from machine_learning_360v2.config import PROCESSED_DATA_DIR
import pandas as pd
import numpy as np

def main(
    input_path=PROCESSED_DATA_DIR / "synthetic_onboarding_data.csv",
    output_path=PROCESSED_DATA_DIR / "synthetic_onboarding_features.csv"
):
    df = pd.read_csv(input_path)

    top_5 = df.head()
    print(f"The top 5 rows are:\n{top_5}\n")
 
    df['log_revenue'] = np.log1p(df['revenue'])
    df['profit_per_employee'] = df['revenue'] / df['employee_count']
    df['late_payment_rate'] = df['late_payments'] / df['age_of_business'].replace(0,1)
    df['debt_profit_ratio'] = df['debt_ratio'] / df['profit_margin'].replace(0, np.nan)
    df['marketing_efficiency'] = df['revenue'] / df['marketing_spend'].replace(0,1)

  
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

    df['business_stage'] = pd.cut(
        df['age_of_business'],
        bins=[0,2,5,np.inf],
        labels=['Startup', 'Growth', 'Mature']
    )

    df.to_csv(output_path, index=False)
    print(f"SME feature dataset saved to: {output_path}")

    print(f'The new dataset: {df}')

    return df

if __name__ == "__main__":
    main()
