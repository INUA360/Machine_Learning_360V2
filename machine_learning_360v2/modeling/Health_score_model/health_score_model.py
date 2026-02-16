from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from loguru import logger
import typer
import joblib
from machine_learning_360v2.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()


@app.command()
def main(
    INPUT_PATH: Path = PROCESSED_DATA_DIR / "health_score_features.csv",
    OUTPUT_PATH: Path = MODELS_DIR / "health_score_model" / "health_model.pkl"
):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)
    logger.info(f"Loaded data: {df.shape}")

    # Ordinal features
    ordinal_features = ['business_stage', 'education_level']
    ord_categories = [
        ['startup', 'growth', 'mature'],
        ['none', 'highschool', 'bachelor', 'master', 'phd']
    ]
    
    ord_encoder = OrdinalEncoder(categories=ord_categories, handle_unknown='use_encoded_value', unknown_value=-1)
    X_ord = ord_encoder.fit_transform(df[ordinal_features])
    
    # Categorical features
    categorical_features = ['sector', 'channels_used', 'owner_gender', 'location']
    ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    X_ohe = ohe.fit_transform(df[categorical_features])
    
    # Numeric features (RAW metrics + simple ratios only!)
    numeric_features = [
        # Raw financials
        'revenue', 'profit_margin', 'debt_ratio', 'collateral_value',
        'marketing_spend', 'employee_count', 'age_of_business',
        'bank_balance', 'm_pesa_balance',
        # Raw operations
        'total_payroll', 'staff_turnover_rate',
        'pending_invoices', 'paid_invoices', 'late_payments',
        # Raw customer
        'total_customers', 'active_customers', 'repeat_customers',
        'clicks', 'conversions',
        # Simple ratios (NOT scores!)
        'liquidity_ratio', 'cash_months',
        'revenue_per_employee', 'payroll_to_revenue',
        'invoice_paid_rate', 'staff_retention_rate',
        'customer_active_rate', 'customer_repeat_rate',
        'marketing_roi', 'conversion_rate'
    ]
    
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[numeric_features])
    
    # Binary features 
    binary_features = [
        'previous_funding_received', 'default_history',
        'business_registration_uploaded', 'tax_clearance_uploaded',
        'financial_statements_uploaded', 'tax_registered',
        'tax_paid_last_year', 'licenses_up_to_date',
        # Individual risk flags 
        'low_cash', 'high_debt', 'low_profit',
        'missing_docs', 'tax_noncompliant', 'license_expired'
    ]
    
    X_bin = df[binary_features].values
    
    # Combine
    X = np.hstack([X_ord, X_ohe, X_num, X_bin])
    
    logger.info(f"Total features: {X.shape[1]}")

    
    y = df['health_category'].values
    
    logger.info(f"\nTarget distribution:\n{pd.Series(y).value_counts()}")

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"\nTraining on {len(X_train)} SMEs...")
    
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info("\n" + "="*60)
    logger.info("MODEL PERFORMANCE")
    logger.info("="*60)
    logger.info(f"Accuracy: {accuracy:.3f}")
    logger.info("\n" + classification_report(y_test, y_pred, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['Critical', 'At Risk', 'Stable', 'Thriving'])
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    logger.success(f"Model trained successfully")

    
    bundle = {
        'model': model,
        'encoders': {
            'ord_encoder': ord_encoder,
            'ohe': ohe,
            'scaler': scaler
        },
        'feature_names': {
            'ordinal': ordinal_features,
            'categorical': categorical_features,
            'numeric': numeric_features,
            'binary': binary_features
        },
        'metadata': {
            'accuracy': accuracy,
            'n_features': X.shape[1]
        }
    }
    
    joblib.dump(bundle, OUTPUT_PATH)
    logger.success(f" Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    app()