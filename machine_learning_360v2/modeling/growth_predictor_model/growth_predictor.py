from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report
from loguru import logger
import typer
import joblib
from machine_learning_360v2.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()


@app.command()
def main(
    INPUT_PATH: Path = PROCESSED_DATA_DIR / "growth_predictor_features.csv",
    OUTPUT_PATH: Path = MODELS_DIR / "growth_predictor_model" / "best_growth_predictor_model.pkl"

):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)
    logger.info(f"Loaded {INPUT_PATH} with shape: {df.shape}")

    # Ordinal features
    ordinal_features = ['business_stage', 'education_level']
    ord_categories = [
        ['startup', 'growth', 'mature'],
        ['none', 'highschool', 'bachelor', 'master', 'phd']
    ]

    existing_ordinal = [f for f in ordinal_features if f in df.columns]
    existing_categories = [ord_categories[i] for i, f in enumerate(ordinal_features) if f in df.columns]

    ord_encoder = OrdinalEncoder(categories=existing_categories, handle_unknown='use_encoded_value', unknown_value=-1)
    X_ord = ord_encoder.fit_transform(df[existing_ordinal])
    
    logger.info(f"Encoded {len(existing_ordinal)} ordinal features")

    # Categorical features
    unordered_features = [
        'sector', 'channels_used', 'target_segment',
        'owner_gender', 'employment_status', 'location',
        'current_revenue_category'
    ]
    existing_unordered = [f for f in unordered_features if f in df.columns]
    
    ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    X_ohe = ohe.fit_transform(df[existing_unordered])
    
    logger.info(f"One-hot encoded {len(existing_unordered)} categorical features to {X_ohe.shape[1]} columns")

    # Numeric features
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
        'revenue_6m_ago', 'customers_6m_ago', 'marketing_spend_6m_ago',
        'revenue_growth_6m', 'revenue_velocity',
        'customer_growth_6m', 'customer_velocity',
        'customer_retention_rate', 'customer_activation_rate', 'customer_acquisition_rate',
        'cash_runway_months', 'debt_capacity', 'reinvestment_capacity',
        'revenue_per_employee', 'operational_leverage', 'staff_stability',
        'customer_base_growth', 'market_penetration_velocity',
        'marketing_roi', 'marketing_efficiency', 'marketing_intensity',
        'conversion_rate', 'marketing_growth_potential',
        'funding_growth_boost', 'composite_growth_score'
    ]
    existing_numeric = [f for f in numeric_features if f in df.columns]
    
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(df[existing_numeric])
    
    logger.info(f"Scaled {len(existing_numeric)} numeric features")

    # Binary features
    binary_features = [
        'previous_funding_received', 'collateral_offered', 'default_history',
        'business_registration_uploaded', 'tax_clearance_uploaded',
        'financial_statements_uploaded', 'tax_registered',
        'tax_paid_last_year', 'licenses_up_to_date',
        'has_funding_track', 'will_jump_category'
    ]
    existing_binary = [f for f in binary_features if f in df.columns]
    X_binary = df[existing_binary].values
    
    logger.info(f"Extracted {len(existing_binary)} binary features")

    X = np.hstack([X_ord, X_ohe, X_num_scaled, X_binary])
    
    logger.info(f"Total features: {X.shape[1]}")

    # Regression targets
    regression_targets = ['predicted_6m_growth_rate']
    y_regression = df[regression_targets].values
    
    logger.info(f"Regression target: {regression_targets}")

    # Train-test split for regression
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training: {X_train.shape[0]} SMEs, Testing: {X_test.shape[0]} SMEs")

    # Train growth rate model
    logger.info("Training growth rate predictor")
    
    growth_model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )
    
    growth_model.fit(X_train, y_train.ravel())
    y_pred = growth_model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info("Growth Rate Model Performance")
    logger.info(f"  RMSE: {rmse:.4f} ({rmse*100:.2f}%)")
    logger.info(f"  MAE:  {mae:.4f} ({mae*100:.2f}%)")
    logger.info(f"  R²:   {r2:.3f}")

    # Train category jump classifier
    logger.info("Training category jump classifier")
    
    y_category_jump = df['will_jump_category'].values
    
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
        X, y_category_jump, test_size=0.2, random_state=42, stratify=y_category_jump
    )
    
    category_jump_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight='balanced',
        random_state=42
    )
    
    category_jump_model.fit(X_train_cat, y_train_cat)
    y_pred_cat = category_jump_model.predict(X_test_cat)
    
    accuracy = accuracy_score(y_test_cat, y_pred_cat)
    
    logger.info("Category Jump Classifier")
    logger.info(f"  Accuracy: {accuracy:.3f}")
    logger.info(f"\n{classification_report(y_test_cat, y_pred_cat, zero_division=0)}")

    # Train growth stage classifier
    logger.info("Training growth stage classifier")
    
    y_stage = df['growth_stage'].values
    
    X_train_stage, X_test_stage, y_train_stage, y_test_stage = train_test_split(
        X, y_stage, test_size=0.2, random_state=42, stratify=y_stage
    )
    
    stage_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )
    
    stage_model.fit(X_train_stage, y_train_stage)
    y_pred_stage = stage_model.predict(X_test_stage)
    
    stage_accuracy = accuracy_score(y_test_stage, y_pred_stage)
    
    logger.info("Growth Stage Classifier")
    logger.info(f"  Accuracy: {stage_accuracy:.3f}")
    logger.info(f"\n{classification_report(y_test_stage, y_pred_stage, zero_division=0)}")

    # Save model bundle
    model_bundle = {
        'models': {
            'growth_rate_model': growth_model,
            'category_jump_model': category_jump_model,
            'growth_stage_model': stage_model
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
        'metadata': {
            'growth_rate_rmse': rmse,
            'growth_rate_r2': r2,
            'category_jump_accuracy': accuracy,
            'growth_stage_accuracy': stage_accuracy
        }
    }

    joblib.dump(model_bundle, OUTPUT_PATH)
    
    logger.success(f"Models saved to {OUTPUT_PATH}")
    logger.info("Bundle includes:")
    logger.info("  - Growth rate regressor")
    logger.info("  - Category jump classifier")
    logger.info("  - Growth stage classifier")
    logger.info("  - All encoders and scalers")


if __name__ == "__main__":
    app()