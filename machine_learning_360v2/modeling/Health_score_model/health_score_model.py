from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder ,OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report
from loguru import logger
import typer
import joblib
from machine_learning_360v2.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()
@app.command()
def main(
    INPUT_PATH: Path = PROCESSED_DATA_DIR / "health_score_features.csv",
    OUTPUT_PATH: Path = MODELS_DIR / "health_score_model" / "best_health_score_model.pkl",
):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)
    logger.info(f"Loaded {INPUT_PATH} with shape: {df.shape}")

    ordinal_features = ['business_stage', 'education_level']
    ord_categories = [
        ['startup', 'growth', 'mature'],
        ['none', 'highschool', 'bachelor', 'master', 'phd'],
    ]

    existing_ordinal = [f for f in ordinal_features if f in df.columns]
    existing_categories = [ord_categories[i] for i, f in enumerate(ordinal_features) if f in df.columns]

    ord_encoder = OrdinalEncoder(categories=existing_categories, handle_unknown='use_encoded_value', unknown_value=-1)
    X_ord = ord_encoder.fit_transform(df[existing_ordinal]) 

    # categorical features
    unordered_features = [
        'sector', 'channels_used', 'target_segment',
        'owner_gender', 'employment_status', 'location'
    ]
    existing_unordered = [f for f in unordered_features if f in df.columns]
    ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    X_ohe = ohe.fit_transform(df[existing_unordered])

    # numeric features
    numeric_features = [
        'revenue', 'profit_margin', 'debt_ratio', 'collateral_value',
        'marketing_spend', 'employee_count', 'age_of_business',
        'previous_funding_amount', 'loan_applications_count',
        'funding_requested_amount', 'expected_roi', 'project_duration_months',
        'staff_count', 'total_payroll', 'cost_per_hire',
        'staff_turnover_rate', 'bank_balance', 'm_pesa_balance',
        'pending_invoices', 'paid_invoices', 'total_customers',
        'active_customers', 'repeat_customers', 'campaign_spend',
        'clicks', 'impressions', 'conversions', 'owner_age',
        'late_payments',
        # engineered features
        'revenue_per_employee','liquidity_months','payroll_ratio',
        'invoice_completion_rate','customer_growth_rate',
        'customer_retention_rate','marketing_roi','marketing_conversion',
    ]
    existing_numeric = [f for f in numeric_features if f in df.columns]
    # binary features
    binary_features = [
        'previous_funding_received', 'collateral_offered','default_history',
        'business_registration_uploaded', 'tax_clearance_uploaded',
        'financial_statements_uploaded', 'tax_registered',
        'tax_paid_last_year', 'licenses_up_to_date',
        # risk flags
        'liquidity_risk','profitability_risk','debt_risk',
        'compliance_risk','growth_stagnation_risk'
    ]
    existing_binary = [f for f in binary_features if f in df.columns]
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(df[existing_numeric])
    X_binary = df[existing_binary].values

    X = np.hstack([X_ord, X_ohe, X_num_scaled, X_binary])
    logger.info(f"Total feature : {X.shape[1]}")

    targets_columns=[
        'financial_health_score', 
        'operational_efficiency_score',
        'compliance_health_score', 
        'growth_potential_score',
        'comprehensive_health_score'
    ]
    y = df[targets_columns].values
    logger.info(f"Target columns: {targets_columns}")
    logger.info(f"Target shape: {y.shape}")

    # Train multi-output model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info("Training multi-output health score model")
    base_model = GradientBoostingRegressor(
        n_estimators=100, 
        max_depth=8, 
        learning_rate=0.1,
        random_state=42)
    
    health_model = MultiOutputRegressor(base_model)
    health_model.fit(X_train, y_train)

    y_pred = health_model.predict(X_test)
    logger.info("\n" + "="*60)
    logger.info("Health score model evaluation:")
    logger.info("="*60)

    for i, col in enumerate(targets_columns):
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])

        logger.info(f"\n{col}:")
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  R²: {r2:.3f}")
        logger.info(f"  MAE: {mae:.2f}")

    # comprehensive health score evaluation (once, after loop)
    comp_idx = targets_columns.index('comprehensive_health_score')
    comp_rmse = np.sqrt(mean_squared_error(y_test[:, comp_idx], y_pred[:, comp_idx]))
    comp_r2 = r2_score(y_test[:, comp_idx], y_pred[:, comp_idx])
    comp_mae = mean_absolute_error(y_test[:, comp_idx], y_pred[:, comp_idx])
    logger.success(f"\n COMPREHENSIVE HEALTH SCORE: RMSE={comp_rmse:.2f}, R²={comp_r2:.3f} MAE={comp_mae:.2f}")

    # train health category classification model
    y_category = df['health_category'].values
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
        X, y_category, test_size=0.2, stratify=y_category, random_state=42
    )
    category_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42)
    category_model.fit(X_train_cat, y_train_cat)
    y_pred_cat = category_model.predict(X_test_cat)
    accuracy = accuracy_score(y_test_cat, y_pred_cat)
    logger.info("\n" + "="*60)
    logger.info("HEALTH CATEGORY CLASSIFIER")
    logger.info("="*60)
    logger.info(f"Accuracy: {accuracy:.3f}")
    logger.info("\n" + classification_report(y_test_cat, y_pred_cat))

    # save model & artifacts
    model_bundle = {
        'models': {
            'health_score_model': health_model,
            'health_category_model': category_model
        },
        'encoders': {
            'ord_encoder': ord_encoder,
            'ohe': ohe,
            'scaler': scaler
        },
        'feature_names': {
            'ordinal': existing_ordinal,
            'onehot': existing_unordered,
            'numeric': existing_numeric,
            'binary': existing_binary
        },
        'target_columns': targets_columns,
        'metadata': {
            'comprehensive_rmse': comp_rmse,
            'comprehensive_r2': comp_r2,
            'category_accuracy': accuracy
        }
    }

    joblib.dump(model_bundle, OUTPUT_PATH)
    logger.success(f"Health score model and encoders saved to {OUTPUT_PATH}")
    logger.info(f"Bundle includes:")
    logger.info(f"  - Multi-output health score regressor")
    logger.info(f"  - Health category classifier")
    logger.info(f"  - All encoders and scalers")
    logger.info(f"  - Feature mappings")


if __name__ == "__main__":
    app()
