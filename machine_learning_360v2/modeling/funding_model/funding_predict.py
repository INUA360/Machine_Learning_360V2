from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import joblib

from machine_learning_360v2.config import PROCESSED_DATA_DIR, MODELS_DIR

FUNDING_MODEL_PATH = (
    Path(r"C:\Users\USER\Desktop\Machine_Learning_360V2")
    / "machine_learning_360v2"
    / "modeling"
    / "funding_model"
    / "best_models_funding.pkl"
)


def load_funding_model() -> dict:
    """
    Load the funding model bundle from disk.
    """
    logger.info("Loading funding model bundle ...")
    bundle = joblib.load(FUNDING_MODEL_PATH)
    logger.success("Funding model loaded successfully.")
    return bundle


def _build_funding_features(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    fn = bundle["feature_names"]

    X_ord = bundle["ord_encoder"].transform(df[fn["ordinal"]])
    X_ohe = bundle["ohe"].transform(df[fn["onehot"]])
    X_num = bundle["scaler"].transform(df[fn["numeric"]])
    X_bin = df[fn["binary"]].values

    return np.hstack([X_ord, X_ohe, X_num, X_bin])



def predict_funding(sme_data: dict, bundle: dict) -> dict:
    """
    Predict funding outcomes for a single SME.

    Returns
    {
        "eligible_for_funding": true,
        "default_risk": false,
        "business_health_score": 73.4
    }
    """
    df = pd.DataFrame([sme_data])
    X  = _build_funding_features(df, bundle)

    models = bundle["models"]

    eligible = bool(models["eligibility_model"].predict(X)[0])
    risk     = bool(models["risk_model"].predict(X)[0])
    health   = float(models["health_model"].predict(X)[0])

    return {
        "eligible_for_funding": eligible,
        "default_risk": risk,
        "business_health_score": round(health, 2),
    }


def predict_all(sme_data: dict, bundle: dict) -> dict:
    return {
        "sme_id": sme_data.get("sme_id", "unknown"),
        "funding": predict_funding(sme_data, bundle),
    }

if __name__ == "__main__":
    logger.info("Running funding predictor test (no CSV) ...")

    bundle = load_funding_model()

    sample_sme = {
        "sme_id": "TEST_001",

        # ----- ordinal -----
        "business_stage": "growth",
        "education_level": "bachelor",

        # ----- categorical -----
        "sector": "retail",
        "channels_used": "online",
        "target_segment": "b2c",
        "owner_gender": "female",
        "employment_status": "self_employed",
        "location": "nairobi",

        # ----- numeric -----
        "revenue": 500000,
        "profit_margin": 0.18,
        "debt_ratio": 0.25,
        "collateral_value": 200000,
        "marketing_spend": 30000,
        "employee_count": 5,
        "age_of_business": 3,
        "previous_funding_amount": 100000,
        "loan_applications_count": 1,
        "funding_requested_amount": 250000,
        "expected_roi": 0.3,
        "project_duration_months": 12,
        "staff_count": 5,
        "total_payroll": 120000,
        "cost_per_hire": 20000,
        "staff_turnover_rate": 0.1,
        "bank_balance": 80000,
        "m_pesa_balance": 40000,
        "pending_invoices": 30000,
        "paid_invoices": 150000,
        "total_customers": 120,
        "active_customers": 80,
        "repeat_customers": 40,
        "campaign_spend": 20000,
        "clicks": 1200,
        "impressions": 15000,
        "conversions": 60,
        "owner_age": 29,

        # ----- binary -----
        "previous_funding_received": 1,
        "collateral_offered": 1,
        "business_registration_uploaded": 1,
        "tax_clearance_uploaded": 1,
        "financial_statements_uploaded": 1,
        "tax_registered": 1,
        "tax_paid_last_year": 1,
        "licenses_up_to_date": 1,
    }

    result = predict_all(sample_sme, bundle)

    logger.info(f"SME ID : {result['sme_id']}")
    logger.info("FUNDING")
    logger.info(f"  Eligible : {result['funding']['eligible_for_funding']}")
    logger.info(f"  Risk     : {result['funding']['default_risk']}")
    logger.info(f"  Health   : {result['funding']['business_health_score']}")

    logger.success("Funding predictor test passed.")