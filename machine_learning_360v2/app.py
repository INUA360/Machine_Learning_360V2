'''Every time you get a new pkl file, this is the entire thought process:
```
pkl file
  ↓
inspect bundle.keys()          → find encoders, feature_names, models
  ↓
read feature_names             → these become SMEInput fields
  ↓
read training file             → look at the feature lists, it tells you exactly what columns the model was trained on, then the encoder order,copy encoder order exactly
  ↓
read features file             → If these columns are in numeric_features, the model was trained expecting them. You must compute them at prediction time too — find computed columns → write _compute_ratios()
  ↓
build SMEInput                 → one field per feature, safe defaults
  ↓
build _build_features()        → missing → fillna → encode → stack → clean
  ↓
write route                    → df → ratios → X → predict → return dict
  ↓
run server → test in /docs
'''


from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Literal
from machine_learning_360v2.config import MODELS_DIR
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
#  SMEInput
#
#  Built by combining all fields from:
#  - funding_predict.py  sample_sme dict  (ordinal, categorical, numeric, binary)
#  - health_score_model.py feature lists  (ordinal, categorical, numeric, binary)
#  - health_score_features.py             (computed ratio fields)
#  - growth_predict.py                    (current_revenue_category, composite_growth_score)
#
#  Every field has a safe default so missing fields never
#  crash the encoders.

class SMEInput(BaseModel):
    #the pattern is field_name: type = default
    sme_id: Optional[str] = "unknown"

    # ordinal (health + funding models)
    # exact values from health_score_model.py ord_categories
    business_stage:  Literal["startup", "growth", "mature"] = "startup"
    education_level: Literal["none", "highschool", "bachelor", "master", "phd"] = "highschool"

    # categorical (health + funding models)
    sector:            Optional[str] = "retail"
    channels_used:     Optional[str] = "online"
    target_segment:    Optional[str] = "b2c"
    owner_gender:      Optional[str] = "female"
    employment_status: Optional[str] = "self_employed"
    location:          Optional[str] = "nairobi"

    # growth model specific
    current_revenue_category: Optional[str]   = "Low"
    composite_growth_score:   Optional[float] = 0.0

    # numeric — from funding_predict.py sample_sme + health_score_model.py numeric_features
    revenue:                  Optional[float] = 0.0
    profit_margin:            Optional[float] = 0.0
    debt_ratio:               Optional[float] = 0.0
    collateral_value:         Optional[float] = 0.0
    marketing_spend:          Optional[float] = 0.0
    employee_count:           Optional[float] = 1.0
    age_of_business:          Optional[float] = 1.0
    previous_funding_amount:  Optional[float] = 0.0
    loan_applications_count:  Optional[float] = 0.0
    funding_requested_amount: Optional[float] = 0.0
    expected_roi:             Optional[float] = 0.0
    project_duration_months:  Optional[float] = 0.0
    staff_count:              Optional[float] = 1.0
    total_payroll:            Optional[float] = 0.0
    cost_per_hire:            Optional[float] = 0.0
    staff_turnover_rate:      Optional[float] = 0.0
    bank_balance:             Optional[float] = 0.0
    m_pesa_balance:           Optional[float] = 0.0
    pending_invoices:         Optional[float] = 0.0
    paid_invoices:            Optional[float] = 0.0
    late_payments:            Optional[float] = 0.0
    total_customers:          Optional[float] = 0.0
    active_customers:         Optional[float] = 0.0
    repeat_customers:         Optional[float] = 0.0
    campaign_spend:           Optional[float] = 0.0
    clicks:                   Optional[float] = 0.0
    impressions:              Optional[float] = 0.0
    conversions:              Optional[float] = 0.0
    owner_age:                Optional[float] = 30.0

    # computed ratios from health_score_features.py
    # set to None so _compute_ratios() knows to calculate them
    liquidity_ratio:      Optional[float] = None
    cash_months:          Optional[float] = None
    revenue_per_employee: Optional[float] = None
    payroll_to_revenue:   Optional[float] = None
    invoice_paid_rate:    Optional[float] = None
    staff_retention_rate: Optional[float] = None
    customer_active_rate: Optional[float] = None
    customer_repeat_rate: Optional[float] = None
    marketing_roi:        Optional[float] = None
    conversion_rate:      Optional[float] = None

    # binary — from funding_predict.py sample_sme + health_score_model.py binary_features
    previous_funding_received:      int = 0
    collateral_offered:             int = 0
    default_history:                int = 0
    business_registration_uploaded: int = 1
    tax_clearance_uploaded:         int = 1
    financial_statements_uploaded:  int = 1
    tax_registered:                 int = 1
    tax_paid_last_year:             int = 1
    licenses_up_to_date:            int = 1

    # risk flags — from health_score_features.py
    low_cash:         int = 0
    high_debt:        int = 0
    low_profit:       int = 0
    missing_docs:     int = 0
    tax_noncompliant: int = 0
    license_expired:  int = 0

#  RATIO CALCULATOR
#  Mirrors health_score_features.py exactly.
#  Runs before health feature building.
#  Only calculates a ratio if the caller didn't provide it.

#these computed columns are in `numeric_features` — meaning the model was **trained on these computed values**. So at prediction 
#time you must compute them too, which is why `_compute_ratios()` exists.
def _compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    safe_rev   = df["revenue"].replace(0, 1)
    safe_click = df["clicks"].replace(0, 1)
    safe_cust  = df["total_customers"].replace(0, 1)
    safe_act   = df["active_customers"].replace(0, 1)
    safe_inv   = (df["paid_invoices"] + df["pending_invoices"]).replace(0, 1)
    safe_mktg  = df["marketing_spend"].replace(0, 1)
    safe_emp   = df["employee_count"].replace(0, 1)

    if df["liquidity_ratio"].isna().any():
        df["liquidity_ratio"] = (df["bank_balance"] + df["m_pesa_balance"]) / safe_rev
    if df["cash_months"].isna().any():
        df["cash_months"] = df["liquidity_ratio"] * 12
    if df["revenue_per_employee"].isna().any():
        df["revenue_per_employee"] = df["revenue"] / safe_emp
    if df["payroll_to_revenue"].isna().any():
        df["payroll_to_revenue"] = df["total_payroll"] / safe_rev
    if df["invoice_paid_rate"].isna().any():
        df["invoice_paid_rate"] = df["paid_invoices"] / safe_inv
    if df["staff_retention_rate"].isna().any():
        df["staff_retention_rate"] = 1 - df["staff_turnover_rate"]
    if df["customer_active_rate"].isna().any():
        df["customer_active_rate"] = df["active_customers"] / safe_cust
    if df["customer_repeat_rate"].isna().any():
        df["customer_repeat_rate"] = df["repeat_customers"] / safe_act
    if df["marketing_roi"].isna().any():
        df["marketing_roi"] = df["revenue"] / safe_mktg
    if df["conversion_rate"].isna().any():
        df["conversion_rate"] = df["conversions"] / safe_click

    return df

#  NaN / inf CLEANER
#  Runs after encoding, right before model.predict().
#  Catches NaN from: missing columns, division by zero,
#  scaler edge cases, and inf from ratio calculations.
#  np.nan_to_num replaces NaN→0, +inf→0, -inf→0.
#

def _clean(X) -> np.ndarray:
    """
    Convert to dense numpy array and replace any NaN/inf with 0.
    Handles both numpy arrays and scipy sparse matrices (from OHE).
    Runs after EVERY encoder step so NaN never reaches the model.
    """
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.array(X, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
#  FEATURE BUILDERS
#  Each one mirrors the predict logic from its source file,
#  with three layers of NaN protection:
#    1. Fill missing columns before encoding
#    2. fillna() on existing columns that contain NaN
#    3. _clean() after stacking — catches anything that slipped through


def _build_growth_features(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Mirrors predict_single() in growth_predict.py"""
    enc = bundle["encoders"]
    fn  = bundle["feature_names"]

    for col in fn["ordinal"]:
        if col not in df.columns: df[col] = "startup"
    for col in fn["onehot"]:
        if col not in df.columns: df[col] = "unknown"
    for col in fn["numeric"]:
        if col not in df.columns: df[col] = 0.0
    for col in fn["binary"]:
        if col not in df.columns: df[col] = 0

    df[fn["ordinal"]] = df[fn["ordinal"]].fillna("startup").astype(str)
    df[fn["onehot"]]  = df[fn["onehot"]].fillna("unknown").astype(str)
    df[fn["numeric"]] = df[fn["numeric"]].fillna(0.0)
    df[fn["binary"]]  = df[fn["binary"]].fillna(0)

    X_ord = _clean(enc["ord_encoder"].transform(df[fn["ordinal"]]))
    X_ohe = _clean(enc["ohe"].transform(df[fn["onehot"]]))
    X_num = _clean(enc["scaler"].transform(df[fn["numeric"]]))
    X_bin = np.nan_to_num(df[fn["binary"]].values.astype(float), nan=0.0)
    return _clean(np.hstack([X_ord, X_ohe, X_num, X_bin]))


def _build_health_features(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Mirrors _build_feature_matrix() in health_score_predict.py"""
    enc = bundle["encoders"]
    fn  = bundle["feature_names"]

    for col in fn["ordinal"]:
        if col not in df.columns:
            df[col] = enc["ord_encoder"].categories_[fn["ordinal"].index(col)][0]
    for col in fn["categorical"]:
        if col not in df.columns: df[col] = "missing"
    for col in fn["numeric"]:
        if col not in df.columns: df[col] = 0.0
    for col in fn["binary"]:
        if col not in df.columns: df[col] = 0

    df[fn["ordinal"]]     = df[fn["ordinal"]].fillna("startup").astype(str)
    df[fn["categorical"]] = df[fn["categorical"]].fillna("missing").astype(str)
    df[fn["numeric"]]     = df[fn["numeric"]].fillna(0.0)
    df[fn["binary"]]      = df[fn["binary"]].fillna(0)

    X_ord = _clean(enc["ord_encoder"].transform(df[fn["ordinal"]]))
    X_ohe = _clean(enc["ohe"].transform(df[fn["categorical"]]))
    X_num = _clean(enc["scaler"].transform(df[fn["numeric"]]))
    X_bin = np.nan_to_num(df[fn["binary"]].values.astype(float), nan=0.0)
    X     = _clean(np.hstack([X_ord, X_ohe, X_num, X_bin]))

    # pad/trim to exact training shape (from bundle metadata)
    n = bundle["metadata"]["n_features"]
    if X.shape[1] < n:
        X = np.hstack([X, np.zeros((X.shape[0], n - X.shape[1]))])
    elif X.shape[1] > n:
        X = X[:, :n]
    return X


def _build_funding_features(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Mirrors _build_funding_features() in funding_predict.py"""
    fn = bundle["feature_names"]

    for col in fn["ordinal"] + fn["onehot"] + fn["numeric"] + fn["binary"]:
        if col not in df.columns: df[col] = 0

    df[fn["ordinal"]] = df[fn["ordinal"]].fillna("startup").astype(str)
    df[fn["onehot"]]  = df[fn["onehot"]].fillna("unknown").astype(str)
    df[fn["numeric"]] = df[fn["numeric"]].fillna(0.0)
    df[fn["binary"]]  = df[fn["binary"]].fillna(0)

    X_ord = _clean(bundle["ord_encoder"].transform(df[fn["ordinal"]]))
    X_ohe = _clean(bundle["ohe"].transform(df[fn["onehot"]]))
    X_num = _clean(bundle["scaler"].transform(df[fn["numeric"]]))
    X_bin = np.nan_to_num(df[fn["binary"]].values.astype(float), nan=0.0)
    return _clean(np.hstack([X_ord, X_ohe, X_num, X_bin]))



def _growth_action(rate: float) -> str:
    """From growth_predict.py growth_action logic"""
    if rate >= 0.25: return "fast_track_funding"
    if rate >= 0.10: return "growth_support"
    if rate >= 0.00: return "stability_support"
    return "intervention_required"


def _compliance_decision(p_default: float, eligible: bool, health_score: float):
    """From compliance_score_sme.py compliance_decision()"""
    if not eligible:
        return "HIGH", ["FUNDING_POLICY_VIOLATION"]
    reasons = []
    if p_default >= 0.6:  reasons.append("HIGH_DEFAULT_RISK")
    if health_score < 50: reasons.append("POOR_BUSINESS_HEALTH")
    if not reasons:
        return "LOW", ["ALL_OK"]
    if "HIGH_DEFAULT_RISK" in reasons or "FUNDING_POLICY_VIOLATION" in reasons:
        return "HIGH", reasons
    return "MEDIUM", reasons


@app.get("/")
def index():
    return {
        "service": "SME Intelligence API",
        "status":  "Online",
        "docs":    "/docs"
    }

'''The critical rule: Whatever transformations were applied to data during training must be applied in exactly the same order at prediction time. 
The pkl file contains the fitted transformers that know how to do this.'''

@app.post("/api/v1/predict/growth")
def predict_growth(data: SMEInput):
    """
    From growth_predict.py predict_single()
    Models: growth_rate_model, growth_stage_model, category_jump_model
    Bundle keys: encoders.ord_encoder, encoders.ohe, encoders.scaler
                 feature_names.ordinal/onehot/numeric/binary
                 models.growth_rate_model/growth_stage_model/category_jump_model
    """
    df          = pd.DataFrame([data.dict()])
    X           = _build_growth_features(df, growth_bundle)
    models      = growth_bundle["models"]

    growth_rate = float(models["growth_rate_model"].predict(X)[0])
    stage       = models["growth_stage_model"].predict(X)[0]
    will_jump   = bool(int(models["category_jump_model"].predict(X)[0]))

    return {
        "sme_id":                   data.sme_id,
        "predicted_6m_growth_rate": round(growth_rate * 100, 1),
        "predicted_6m_revenue":     round(data.revenue * (1 + growth_rate), 0),
        "growth_stage":             stage,
        "will_jump_category":       will_jump,
        "current_revenue_category": data.current_revenue_category,
        "composite_growth_score":   round(data.composite_growth_score, 1),
        "growth_action":            _growth_action(growth_rate),
    }


@app.post("/api/v1/predict/health")
def predict_health(data: SMEInput):
    """
    From health_score_predict.py predict_single()
    Models: health_bundle['model']  (GradientBoostingClassifier)
    Bundle keys: encoders.ord_encoder, encoders.ohe, encoders.scaler
                 feature_names.ordinal/categorical/numeric/binary
                 metadata.n_features
    """
    df       = pd.DataFrame([data.dict()])
    df       = _compute_ratios(df)
    X        = _build_health_features(df, health_bundle)
    category = health_bundle["model"].predict(X)[0]

    risk_flags          = {col: getattr(data, col) for col in RISK_FLAG_COLS}
    risk_flags["total"] = sum(risk_flags.values())

    return {
        "sme_id":          data.sme_id,
        "health_category": category,
        "risk_flags":      risk_flags,
    }


@app.post("/api/v1/predict/funding")
def predict_funding(data: SMEInput):
    """
    From funding_predict.py predict_funding()
    Models: eligibility_model, risk_model, health_model
    Bundle keys: ord_encoder, ohe, scaler (top-level, not nested under 'encoders')
                 feature_names.ordinal/onehot/numeric/binary
                 models.eligibility_model/risk_model/health_model
    """
    df       = pd.DataFrame([data.dict()])
    X        = _build_funding_features(df, funding_bundle)
    models   = funding_bundle["models"]

    eligible = bool(models["eligibility_model"].predict(X)[0])
    risk     = bool(models["risk_model"].predict(X)[0])
    health   = float(models["health_model"].predict(X)[0])

    return {
        "sme_id":                data.sme_id,
        "eligible_for_funding":  eligible,
        "default_risk":          risk,
        "business_health_score": round(health, 2),
    }


@app.post("/api/v1/predict/compliance")
def predict_compliance(data: SMEInput):
    """
    From compliance_score_sme.py compliance_decision()
    Reuses funding bundle — same encoders and models.
    Key difference: uses predict_proba()[:,1] for probability
    instead of predict() for binary label.
    """
    df        = pd.DataFrame([data.dict()])
    X         = _build_funding_features(df, funding_bundle)
    models    = funding_bundle["models"]

    eligible  = bool(models["eligibility_model"].predict(X)[0])
    p_default = float(models["risk_model"].predict_proba(X)[0, 1])
    health    = float(models["health_model"].predict(X)[0])

    risk_level, reasons = _compliance_decision(p_default, eligible, health)

    return {
        "sme_id":                   data.sme_id,
        "eligible_for_funding":     eligible,
        "default_risk_probability": round(p_default, 3),
        "business_health_score":    round(health, 1),
        "compliance_risk":          risk_level,
        "reasons":                  reasons,
    }


@app.post("/api/v1/predict/all")
def predict_all(data: SMEInput):
    """
    Runs all four models in one request.
    Mirrors health_growth_predict.py predict_all() pattern.
    """
    return {
        "sme_id":     data.sme_id,
        "growth":     predict_growth(data),
        "health":     predict_health(data),
        "funding":    predict_funding(data),
        "compliance": predict_compliance(data),
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)