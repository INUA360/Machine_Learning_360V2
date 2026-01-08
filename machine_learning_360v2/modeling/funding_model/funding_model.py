import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from machine_learning_360v2.config import PROCESSED_DATA_DIR 
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import typer
import joblib

app = typer.Typer()

@app.command()
def main(
    INPUT_PATH: Path = PROCESSED_DATA_DIR/"synthetic_sme_agents_data.csv",
    OUTPUT_PATH: Path = PROCESSED_DATA_DIR/"best_models_funding.pkl"
):
    df = pd.read_csv(INPUT_PATH)
    logger.info(f"Loaded data: {df.shape}")

    # Encoding the data
    # Ordinal features (ordered categories)
    ordinal_features = ['business_stage', 'education_level']
    
    categories = [
        ['startup', 'growth', 'mature'],
        ['none', 'highschool', 'bachelor', 'master', 'phd']
    ]
    
    # # SYNTAX TIP: fit_transform = learn + apply, transform = just apply
    ord_encoder = OrdinalEncoder(categories=categories)
    X_ord = ord_encoder.fit_transform(df[ordinal_features])
    logger.info(f"Ordinal encoded: {X_ord.shape}")

    # One-hot encoding for unordered features
    unordered_features = ['sector', 'channels_used', 'target_segment', 
                          'owner_gender', 'employment_status', 'location', 'default_risk']
    ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore') #sparse_output=False: Returns normal array (not sparse matrix)
    
    """ 
    Sparse_output=False: Returns normal array (not sparse matrix)
    # - drop='first': Avoids dummy variable trap (if not A and not B, must be C)
    # - handle_unknown='ignore': Won't crash if new data has unseen categories
    
    # SYNTAX TIP: Always set sparse_output=False or you'll get array type errors later!
    """
    X_ohe = ohe.fit_transform(df[unordered_features])
    logger.info(f"One-hot encoded: {X_ohe.shape}")

    # Numerical features
    numeric_features = ['revenue', 'profit_margin', 'debt_ratio', 'collateral_value', 
                        'marketing_spend', 'employee_count', 'age_of_business',
                        'previous_funding_amount', 'loan_applications_count',
                        'funding_requested_amount', 'expected_roi', 'project_duration_months',
                        'staff_count', 'total_payroll', 'cost_per_hire', 
                        'staff_turnover_rate', 'bank_balance', 'm_pesa_balance',
                        'pending_invoices', 'paid_invoices', 'total_customers',
                        'active_customers', 'repeat_customers', 'campaign_spend',
                        'clicks', 'impressions', 'conversions', 'owner_age']
    
    # Binary features (already 0/1, no scaling needed)
    binary_features = ['previous_funding_received', 'default_history', 'collateral_offered',
                       'business_registration_uploaded', 'tax_clearance_uploaded',
                       'financial_statements_uploaded', 'tax_registered', 'tax_paid_last_year',
                       'licenses_up_to_date']
    
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(df[numeric_features])
    X_binary = df[binary_features].values #.values converts DataFrame to numpy array (needed for hstack)
    logger.info(f"Numerical scaled: {X_num_scaled.shape}, Binary: {X_binary.shape}")

    #StandardScaler makes each feature have mean=0, std=1
    # WHY? Revenue (500k) vs profit_margin (0.15) - without scaling, 
    # model thinks revenue is 3 million times more important
    
    #Always fit on TRAINING data only, then transform test data

    # Combine all features
    X = np.hstack([X_ord, X_ohe, X_num_scaled, X_binary]) #np.hstack = horizontal stack (combines columns)
    logger.info(f"Final feature matrix: {X.shape}")

    # Targets
    y_eligibility = df['eligible_for_funding'].values
    y_health = df['business_health_score'].values
    y_risk = df['default_risk'].values

    #values removes the index

    logger.info("Training the eligibility model")

    X_train, X_test, y_train, y_test = train_test_split(X, y_eligibility, test_size=0.3, random_state=42)

    funding_model = RandomForestClassifier(
        n_estimators=100, #Number of trees
        max_depth = 10, #prevents overfitting
        random_state=42
    )
    funding_model.fit(X_train, y_train)

    y_pred = funding_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"The accuracy for the eligibility model is {accuracy}")
    print(classification_report(y_test,y_pred))

 
 #Default risk classification
    logger.info("Training the defaukt risk model")

    X_train, X_test, y_train, y_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)
    default_risk = RandomForestClassifier(
        n_estimators=100, #Number of trees
        max_depth = 10, #prevents overfitting
        random_state=42
    )
    default_risk.fit(X_train, y_train)
    y_pred=default_risk.predict(X_test)

    accauracy = accuracy_score(y_test, y_pred)
    logger.info(f"The accuracy for the default risk is {accuracy}")
    print(classification_report(y_test,y_pred))

#Business health model   logger.info("\n" + "="*50)
    logger.info("Training Business Health Model (Regression)")
    logger.info("="*50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_health, test_size=0.2, random_state=42
    )
    
    # NOTE: Using RandomForestREGRESSOR (not Classifier!)
    reg_health = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    reg_health.fit(X_train, y_train)
    y_pred = reg_health.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    logger.success(f"Business Health RMSE: {rmse:.2f}, R²: {r2:.3f}")

    #RMSE: Average prediction error (lower = better)
    models = {
        'eligibility_model': funding_model,
        'risk_model':default_risk,
        'health_model': reg_health
    }

    #it is critical to save encoders with models for later deployment
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'models': models,
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

    logger.success(f"\n All models and encoders saved to {OUTPUT_PATH}")
    logger.info(f"\n Summary:")
    logger.info(f"   - 3 models trained successfully")
    logger.info(f"   - Eligibility: Binary Classification")
    logger.info(f"   - Default Risk: Multi-class Classification") 
    logger.info(f"   - Business Health: Regression")


if __name__ == "__main__":
    app()