from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from machine_learning_360v2.config import MODELS_DIR, PROCESSED_DATA_DIR
import uvicorn

app = FastAPI(
    title="SME Intelligence API",
    version="1.0.0",
    description="ML-powered SME growth, health, funding, and compliance predictions"
)

GROWTH_MODEL_PATH  = MODELS_DIR / "growth_predictor_model" / "best_growth_predictor_model.pkl"
HEALTH_MODEL_PATH  = MODELS_DIR / "health_score_model"     / "health_model.pkl"
FUNDING_MODEL_PATH = MODELS_DIR / "funding_model"          / "best_models_funding.pkl"

logger.info("Loading models ...")
growth_bundle  = joblib.load(GROWTH_MODEL_PATH)
health_bundle  = joblib.load(HEALTH_MODEL_PATH)
funding_bundle = joblib.load(FUNDING_MODEL_PATH)
logger.success("All models loaded")

RISK_FLAG_COLS = [
    "low_cash", "high_debt", "low_profit",
    "missing_docs", "tax_noncompliant", "license_expired",
]

class SMEInput(BaseModel):
    sme_id: Optional[str] = "unknown"
    revenue: Optional[float] = 0.0
    current_revenue_category: Optional[str] = "Unknown"
    composite_growth_score: Optional[float] = 0.0
    low_cash: int = 0
    high_debt: int = 0
    low_profit: int = 0
    missing_docs: int = 0
    tax_noncompliant: int = 0
    license_expired: int = 0

def _build_growth_features(df, bundle):
    enc = bundle["encoders"]
    fn  = bundle["feature_names"]
    for col in fn["ordinal"] + fn["onehot"] + fn["numeric"] + fn["binary"]:
        if col not in df.columns:
            df[col] = 0
    X_ord = enc["ord_encoder"].transform(df[fn["ordinal"]])
    X_ohe = enc["ohe"].transform(df[fn["onehot"]])
    X_num = enc["scaler"].transform(df[fn["numeric"]])
    X_bin = df[fn["binary"]].values
    return np.hstack([X_ord, X_ohe, X_num, X_bin])

def _build_health_features(df, bundle):
    enc = bundle["encoders"]
    fn  = bundle["feature_names"]
    for col in fn["ordinal"] + fn.get("categorical", []) + fn["numeric"] + fn["binary"]:
        if col not in df.columns:
            df[col] = 0
    X_ord = enc["ord_encoder"].transform(df[fn["ordinal"]])
    X_ohe = enc["ohe"].transform(df.get("categorical", [])) if "categorical" in fn else np.empty((len(df), 0))
    X_num = enc["scaler"].transform(df[fn["numeric"]])
    X_bin = df[fn["binary"]].values
    return np.hstack([X_ord, X_ohe, X_num, X_bin])

def _build_funding_features(df, bundle):
    fn = bundle["feature_names"]
    for col in fn["ordinal"] + fn["onehot"] + fn["numeric"] + fn["binary"]:
        if col not in df.columns:
            df[col] = 0
    X_ord = bundle["ord_encoder"].transform(df[fn["ordinal"]])
    X_ohe = bundle["ohe"].transform(df[fn["onehot"]])
    X_num = bundle["scaler"].transform(df[fn["numeric"]])
    X_bin = df[fn["binary"]].values
    return np.hstack([X_ord, X_ohe, X_num, X_bin])

def _growth_action(rate):
    if rate >= 0.25: return "fast_track_funding"
    if rate >= 0.10: return "growth_support"
    if rate >= 0.00: return "stability_support"
    return "intervention_required"

def _compliance_decision(p_default, eligible, health_score):
    if not eligible:
        return "HIGH", ["FUNDING_POLICY_VIOLATION"]
    reasons = []
    if p_default >= 0.6:  reasons.append("HIGH_DEFAULT_RISK")
    if health_score < 50: reasons.append("POOR_BUSINESS_HEALTH")
    if not reasons:
        return "LOW", ["ALL_OK"]
    if "HIGH_DEFAULT_RISK" in reasons:
        return "HIGH", reasons
    return "MEDIUM", reasons

@app.get("/")
def index():
    return {
        "service": "SME Intelligence API",
        "status": "Online",
        "docs": "/docs"
    }

@app.post("/api/v1/predict/growth")
def predict_growth(data: SMEInput):
    df = pd.DataFrame([data.dict()])
    X = _build_growth_features(df, growth_bundle)
    models = growth_bundle["models"]
    growth_rate = float(models["growth_rate_model"].predict(X)[0])
    stage = models["growth_stage_model"].predict(X)[0]
    will_jump = bool(models["category_jump_model"].predict(X)[0])
    return {
        "sme_id": data.sme_id,
        "predicted_6m_growth_rate": round(growth_rate * 100, 1),
        "predicted_6m_revenue": round(data.revenue * (1 + growth_rate), 0),
        "growth_stage": stage,
        "will_jump_category": will_jump,
        "current_revenue_category": data.current_revenue_category,
        "composite_growth_score": round(data.composite_growth_score, 1),
        "growth_action": _growth_action(growth_rate),
    }

@app.post("/api/v1/predict/health")
def predict_health(data: SMEInput):
    df = pd.DataFrame([data.dict()])
    X = _build_health_features(df, health_bundle)
    category = health_bundle["model"].predict(X)[0]
    risk_flags = {col: getattr(data, col) for col in RISK_FLAG_COLS}
    risk_flags["total"] = sum(risk_flags.values())
    return {
        "sme_id": data.sme_id,
        "health_category": category,
        "risk_flags": risk_flags,
    }

@app.post("/api/v1/predict/funding")
def predict_funding(data: SMEInput):
    df = pd.DataFrame([data.dict()])
    X = _build_funding_features(df, funding_bundle)
    models = funding_bundle["models"]
    eligible = bool(models["eligibility_model"].predict(X)[0])
    risk = bool(models["risk_model"].predict(X)[0])
    health = float(models["health_model"].predict(X)[0])
    return {
        "sme_id": data.sme_id,
        "eligible_for_funding": eligible,
        "default_risk": risk,
        "business_health_score": round(health, 2),
    }

@app.post("/api/v1/predict/compliance")
def predict_compliance(data: SMEInput):
    df = pd.DataFrame([data.dict()])
    X = _build_funding_features(df, funding_bundle)
    models = funding_bundle["models"]
    eligible = bool(models["eligibility_model"].predict(X)[0])
    p_default = float(models["risk_model"].predict_proba(X)[0, 1])
    health = float(models["health_model"].predict(X)[0])
    risk_level, reasons = _compliance_decision(p_default, eligible, health)
    return {
        "sme_id": data.sme_id,
        "eligible_for_funding": eligible,
        "default_risk_probability": round(p_default, 3),
        "business_health_score": round(health, 1),
        "compliance_risk": risk_level,
        "reasons": reasons,
    }

@app.post("/api/v1/predict/all")
def predict_all(data: SMEInput):
    growth = predict_growth(data)
    health = predict_health(data)
    funding = predict_funding(data)
    compliance = predict_compliance(data)
    return {
        "sme_id": data.sme_id,
        "growth": growth,
        "health": health,
        "funding": funding,
        "compliance": compliance,
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )