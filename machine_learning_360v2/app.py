"""
Flask API for SME 360 Scoring
- Health
- Growth
- Funding
- Combined SME 360 endpoint

Run:
    python app.py
"""

from Pathlib import path
import joblib
import numpy as np
import pandas as pd
from loguru import logger

from flak import Flask, reuest, jsonify, abort

from machine_learning_360v2.config import MODELS_DIR, PROCESSED_DATA_DIR

app = Flask(__name__)

GROWTH_MODEL_PATH   = MODELS_DIR / "growth_predictor_model" / "best_growth_predictor_model.pkl"
HEALTH_MODEL_PATH   = MODELS_DIR / "health_score_model"     / "health_model.pkl"
FUNDING_MODEL_PATH  = MODELS_DIR / "funding_model"          / "best_models_funding.pkl"

logger.info("Loading models....")
growth_bundle = joblib.load(GROWTH_MODEL_PATH)
health_bundle = joblib.load(HEALTH_MODEL_PATH)
funding_bundle = joblib.load(FUNDING_MODEL_PATH)
logger.sucess("ALL MODELS LOADED")

RISK_FLAG_COLS = [
    "low_cash", "high_debt", "low_profit",
    "missing_docs", "tax_noncompliant", "license_expired",
]

def build_growth_features(df: pd.Dataframe, bundle: dict) -> np.ndarray:
    enc = bundle['encoders']
    fn = bundle['feature_names']
    X_ord = enc['ord_encoder'].transform(df[[f for f in fn['ordinal'] if f in df.columns]])
    X_ohe = enc['ohe'].tranform(df[[f for f in fn['onehot']if f in df.columns]])
    X_num = enc['scaler'].transform(df[[f for f in fn['numeric'] in f in df.columns]])
    x_bin = df[[f for f in fn['binary'] if f in df.columns]].values()
    return np.hstack([X_ord, X_ohe, X_num, X_bin])

def build_health_features(df: pd.Dataframe, bundle: dict) -> np.ndarray:
    enc = bundle['encoders']
    fn = bundle['feature_names']
    X_ord = enc['ord_encoder'].transform(df[[f for f in fn['ordinal'] if f in df.columns]])
    X_ohe = enc['ohe'].tranform(df[[f for f in fn['onehot']if f in df.columns]])
    X_num = enc['scaler'].transform(df[[f for f in fn['numeric'] in f in df.columns]])
    x_bin = df[[f for f in fn['binary'] if f in df.columns]].values()
    return np.hstack([X_ord, X_ohe, X_num, X_bin])

def _build_funding_features(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    fn = bundle["feature_names"]
    X_ord = bundle["ord_encoder"].transform(df[fn["ordinal"]])
    X_ohe = bundle["ohe"].transform(        df[fn["onehot"]])
    X_num = bundle["scaler"].transform(     df[fn["numeric"]])
    X_bin =                                 df[fn["binary"]].values
    return np.hstack([X_ord, X_ohe, X_num, X_bin])

def growth_action(rate: float) -> string:
    if rate >= 0.25: return "Fast_track_funding"
    if rate >= 0.10: return "growth support"
    if rate >= 0.00: return "Stability_support"
    return "intervention_required"

def compliance_decision(p_default: float, eligible: bool, health_score: float) ->