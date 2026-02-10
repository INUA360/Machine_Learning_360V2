# train_models.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from loguru import logger
from pathlib import Path
import typer
import joblib

from machine_learning_360v2.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()

@app.command()
def main(
    INPUT_PATH: Path = PROCESSED_DATA_DIR / "synthetic_onboarding_features.csv",
    OUTPUT_PATH: Path = MODELS_DIR / "funding_model" / "best_models_funding.pkl"
):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    logger.info(f"Loaded engineered data: {df.shape}")

    ordinal_features = ['business_stage', 'education_level']
    ord_categories = [
        ['startup', 'growth', 'mature'],
        ['none', 'highschool', 'bachelor', 'master', 'phd']
    ]
    ord_encoder = OrdinalEncoder(categories=ord_categories)
    X_ord = ord_encoder.fit_transform(df[ordinal_features])

    unordered_features = [
        'sector', 'channels_used', 'target_segment',
        'owner_gender', 'employment_status', 'location'
    ]
    ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    X_ohe = ohe.fit_transform(df[unordered_features])

    numeric_features = [
        'revenue', 'profit_margin', 'debt_ratio', 'collateral_value',
        'marketing_spend', 'employee_count', 'age_of_business',
        'previous_funding_amount', 'loan_applications_count',
        'funding_requested_amount', 'expected_roi', 'project_duration_months',
        'staff_count', 'total_payroll', 'cost_per_hire',
        'staff_turnover_rate', 'bank_balance', 'm_pesa_balance',
        'pending_invoices', 'paid_invoices', 'total_customers',
        'active_customers', 'repeat_customers', 'campaign_spend',
        'clicks', 'impressions', 'conversions', 'owner_age'
    ]

    binary_features = [
        'previous_funding_received', 'collateral_offered',
        'business_registration_uploaded', 'tax_clearance_uploaded',
        'financial_statements_uploaded', 'tax_registered',
        'tax_paid_last_year', 'licenses_up_to_date'
    ]

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(df[numeric_features])
    X_binary = df[binary_features].values

    X = np.hstack([X_ord, X_ohe, X_num_scaled, X_binary])

    y_eligibility = df['eligible_for_funding'].values
    y_risk = df['default_risk'].values
    y_health = df['business_health_score'].values

    # Eligibility model
    logger.info("Training eligibility model")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_eligibility, test_size=0.3, stratify=y_eligibility, random_state=42
    )
    funding_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    funding_model.fit(X_train, y_train)
    y_pred = funding_model.predict(X_test)
    logger.info(f"Eligibility accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

    # Default risk model
    logger.info("Training default risk model")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_risk, test_size=0.2, stratify=y_risk, random_state=42
    )
    risk_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    risk_model.fit(X_train, y_train)
    y_pred = risk_model.predict(X_test)
    logger.info(f"Default risk accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

    # Business health regression model
    logger.info("Training business health regression model")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_health, test_size=0.2, random_state=42
    )
    health_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    health_model.fit(X_train, y_train)
    y_pred = health_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    logger.success(f"Business health RMSE: {rmse:.2f}, R²: {r2:.3f}")

    # Save models & encoders
    joblib.dump({
        'models': {
            'eligibility_model': funding_model,
            'risk_model': risk_model,
            'health_model': health_model
        },
        'ord_encoder': ord_encoder,
        'ohe': ohe,
        'scaler': scaler,
        'feature_names': {
            'ordinal': ordinal_features,
            'onehot': unordered_features,
            'numeric': numeric_features,
            'binary': binary_features
        }
    }, OUTPUT_PATH)

    logger.success(f"All models and encoders saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    app()
