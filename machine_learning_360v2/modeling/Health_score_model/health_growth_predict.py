from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import joblib
from machine_learning_360v2.config import PROCESSED_DATA_DIR, MODELS_DIR

HEALTH_MODEL_PATH = MODELS_DIR / "health_score_model" / "health_model.pkl"
GROWTH_MODEL_PATH = MODELS_DIR / "growth_predictor_model" / "best_growth_predictor_model.pkl"

# Risk flags extracted from incoming SME data for the health result
RISK_FLAG_COLS = [
    "low_cash", "high_debt", "low_profit",
    "missing_docs", "tax_noncompliant", "license_expired",
]


def load_models() -> dict:
    """
    Load both model bundles from disk.
    Returns
    {
        "health": <health model bundle>,
        "growth": <growth model bundle>
    }
    """
    logger.info("Loading health model ...")
    health_bundle = joblib.load(HEALTH_MODEL_PATH)

    logger.info("Loading growth model ...")
    growth_bundle = joblib.load(GROWTH_MODEL_PATH)

    logger.success("Both models loaded successfully.")
    return {
        "health": health_bundle,
        "growth": growth_bundle,
    }


def _build_health_features(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    enc = bundle["encoders"]
    fn  = bundle["feature_names"]

    X_ord = enc["ord_encoder"].transform(df[[f for f in fn["ordinal"]     if f in df.columns]])
    X_ohe = enc["ohe"].transform(        df[[f for f in fn["categorical"] if f in df.columns]])
    X_num = enc["scaler"].transform(     df[[f for f in fn["numeric"]     if f in df.columns]])
    X_bin =                              df[[f for f in fn["binary"]      if f in df.columns]].values

    return np.hstack([X_ord, X_ohe, X_num, X_bin])


def _build_growth_features(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    enc = bundle["encoders"]
    fn  = bundle["feature_names"]

    X_ord = enc["ord_encoder"].transform(df[[f for f in fn["ordinal"] if f in df.columns]])
    X_ohe = enc["ohe"].transform(        df[[f for f in fn["onehot"]  if f in df.columns]])
    X_num = enc["scaler"].transform(     df[[f for f in fn["numeric"] if f in df.columns]])
    X_bin =                              df[[f for f in fn["binary"]  if f in df.columns]].values

    return np.hstack([X_ord, X_ohe, X_num, X_bin])


def _growth_action(rate: float) -> str:
    if rate >= 0.25: return "fast_track_funding"
    if rate >= 0.10: return "growth_support"
    if rate >= 0.00: return "stability_support"
    return "intervention_required"


def predict_health(sme_data: dict, health_bundle: dict) -> dict:
    """
    Predict the health category for a single SME.
    Returns
    {
        "health_category": "Thriving" | "Stable" | "At Risk" | "Critical",
        "risk_flags": {
            "low_cash": 0,
            "high_debt": 1,
            "low_profit": 0,
            "missing_docs": 0,
            "tax_noncompliant": 0,
            "license_expired": 0,
            "total": 1
        }
    }
    """
    df       = pd.DataFrame([sme_data])
    X        = _build_health_features(df, health_bundle)
    category = health_bundle["model"].predict(X)[0]

    risk_flags         = {col: int(sme_data.get(col, 0)) for col in RISK_FLAG_COLS}
    risk_flags["total"] = sum(risk_flags.values())

    return {
        "health_category": category,
        "risk_flags": risk_flags,
    }


def predict_growth(sme_data: dict, growth_bundle: dict) -> dict:
    """
    Predict 6-month growth for a single SME.
    Returns
    {
        "predicted_6m_growth_rate": 12.5,          # percentage
        "predicted_6m_revenue": 281250.0,           # KES
        "growth_stage": "Growing",
        "will_jump_category": false,
        "current_revenue_category": "Micro",
        "composite_growth_score": 67.3,
        "growth_action": "growth_support"           
    }
    """
    df         = pd.DataFrame([sme_data])
    X          = _build_growth_features(df, growth_bundle)
    models     = growth_bundle["models"]

    growth_rate  = float(models["growth_rate_model"].predict(X)[0])
    growth_stage = models["growth_stage_model"].predict(X)[0]
    will_jump    = bool(models["category_jump_model"].predict(X)[0])

    current_revenue   = float(sme_data.get("revenue", 0))
    predicted_revenue = round(current_revenue * (1 + growth_rate), 0)

    return {
        "predicted_6m_growth_rate":  round(growth_rate * 100, 1),
        "predicted_6m_revenue":      predicted_revenue,
        "growth_stage":              growth_stage,
        "will_jump_category":        will_jump,
        "current_revenue_category":  sme_data.get("current_revenue_category", "Unknown"),
        "composite_growth_score":    round(float(sme_data.get("composite_growth_score", 0)), 1),
        "growth_action":             _growth_action(growth_rate),
    }


def predict_all(sme_data: dict, bundles: dict) -> dict:
  
    return {
        "sme_id": sme_data.get("sme_id", "unknown"),
        "health": predict_health(sme_data, bundles["health"]),
        "growth": predict_growth(sme_data, bundles["growth"]),
    }


# test

if __name__ == "__main__":
    logger.info("Running test ...")

    bundles = load_models()

    health_df = pd.read_csv(PROCESSED_DATA_DIR / "health_score_features.csv")
    growth_df = pd.read_csv(PROCESSED_DATA_DIR / "growth_predictor_features.csv")
    merged_df = pd.merge(health_df, growth_df, on="sme_id", suffixes=("", "_growth"))

    sample = merged_df.iloc[0].to_dict()
    result = predict_all(sample, bundles)

    
    logger.info(f"SME ID : {result['sme_id']}")
    logger.info("HEALTH")
    logger.info(f"  Category   : {result['health']['health_category']}")
    logger.info(f"  Risk Flags : {result['health']['risk_flags']}")
    logger.info("GROWTH")
    logger.info(f"  6m Growth  : {result['growth']['predicted_6m_growth_rate']}%")
    logger.info(f"  6m Revenue : KES {result['growth']['predicted_6m_revenue']:,.0f}")
    logger.info(f"  Stage      : {result['growth']['growth_stage']}")
    logger.info(f"  Jump Cat.  : {result['growth']['will_jump_category']}")
    logger.info(f"  Action     : {result['growth']['growth_action']}")
    logger.success("test passed.")