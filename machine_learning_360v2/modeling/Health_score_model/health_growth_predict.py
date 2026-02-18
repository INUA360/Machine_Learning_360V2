from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Any, Union, List,Tuple, Optional
import typer
import joblib
from machine_learning_360v2.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()

RISK_FLAG_COLS = [
    'low_cash', 'high_debt', 'low_profit',
    'missing_docs', 'tax_noncompliant', 'license_expired'
]


def load_models():
    
    health_bundle = joblib.load(MODELS_DIR / "health_score_model" / "health_model.pkl")
    growth_bundle = joblib.load(MODELS_DIR / "growth_predictor_model" / "best_growth_predictor_model.pkl")
    
    return {
        'health': health_bundle,
        'growth': growth_bundle
    }


def _build_health_features(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Build feature matrix for health model."""
    encoders = bundle['encoders']
    feature_names = bundle['feature_names']

    existing_ordinal = [f for f in feature_names['ordinal'] if f in df.columns]
    existing_categorical = [f for f in feature_names['categorical'] if f in df.columns]
    existing_numeric = [f for f in feature_names['numeric'] if f in df.columns]
    existing_binary = [f for f in feature_names['binary'] if f in df.columns]

    X_ord = encoders['ord_encoder'].transform(df[existing_ordinal])
    X_ohe = encoders['ohe'].transform(df[existing_categorical])
    X_num = encoders['scaler'].transform(df[existing_numeric])
    X_bin = df[existing_binary].values

    return np.hstack([X_ord, X_ohe, X_num, X_bin])


def _build_growth_features(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Build feature matrix for growth model."""
    encoders = bundle['encoders']
    feature_names = bundle['feature_names']

    existing_ordinal = [f for f in feature_names['ordinal'] if f in df.columns]
    existing_unordered = [f for f in feature_names['onehot'] if f in df.columns]
    existing_numeric = [f for f in feature_names['numeric'] if f in df.columns]
    existing_binary = [f for f in feature_names['binary'] if f in df.columns]

    X_ord = encoders['ord_encoder'].transform(df[existing_ordinal])
    X_ohe = encoders['ohe'].transform(df[existing_unordered])
    X_num = encoders['scaler'].transform(df[existing_numeric])
    X_bin = df[existing_binary].values

    return np.hstack([X_ord, X_ohe, X_num, X_bin])


def predict_health(sme_data: dict, health_bundle: dict) -> dict:
    """
    Predict health category for a single SME.
    Returns:
        {
            'health_category': 'Thriving',
            'risk_flags': {...}
        }
    """
    df = pd.DataFrame([sme_data])
    
    model = health_bundle['model']
    X = _build_health_features(df, health_bundle)
    category = model.predict(X)[0]
    
    risk_flags = {col: int(sme_data.get(col, 0)) for col in RISK_FLAG_COLS}
    risk_flags['total'] = sum(risk_flags.values())
    
    return {
        'health_category': category,
        'risk_flags': risk_flags
    }


def predict_growth(sme_data: dict, growth_bundle: dict) -> dict:
    """
    Predict 6-month growth for a single SME.
    Returns:
        {
            'predicted_6m_growth_rate': 12.5,
            'predicted_6m_revenue': 281250,
            'growth_stage': 'Growing',
            'will_jump_category': false,
            'growth_action': 'growth_support'
        }
    """
    df = pd.DataFrame([sme_data])
    
    models = growth_bundle['models']
    X = _build_growth_features(df, growth_bundle)
    
    growth_rate = float(models['growth_rate_model'].predict(X)[0])
    growth_stage = models['growth_stage_model'].predict(X)[0]
    will_jump = int(models['category_jump_model'].predict(X)[0])
    
    current_revenue = float(sme_data.get('revenue', 0))
    predicted_revenue = round(current_revenue * (1 + growth_rate), 0)
    
    growth_action = (
        'fast_track_funding' if growth_rate >= 0.25 else
        'growth_support' if growth_rate >= 0.10 else
        'stability_support' if growth_rate >= 0 else
        'intervention_required'
    )
    
    return {
        'predicted_6m_growth_rate': round(growth_rate * 100, 1),
        'predicted_6m_revenue': predicted_revenue,
        'growth_stage': growth_stage,
        'will_jump_category': bool(will_jump),
        'current_revenue_category': sme_data.get('current_revenue_category', 'Unknown'),
        'composite_growth_score': round(float(sme_data.get('composite_growth_score', 0)), 1),
        'growth_action': growth_action
    }


def predict_all(sme_data: dict, bundles: dict) -> dict:
    """
    Combined prediction: health + growth for a single SME
    Returns:
        {
            'sme_id': 123,
            'health': {...},
            'growth': {...}
        }
    """
    health_result = predict_health(sme_data, bundles['health'])
    growth_result = predict_growth(sme_data, bundles['growth'])
    
    return {
        'sme_id': sme_data.get('sme_id', 'unknown'),
        'health': health_result,
        'growth': growth_result
    }


@app.command()
def test():

    logger.info("Loading models...")
    bundles = load_models()
    logger.success("Models loaded successfully")
    
    logger.info("Loading sample data...")
    health_df = pd.read_csv(PROCESSED_DATA_DIR / "health_score_features.csv")
    growth_df = pd.read_csv(PROCESSED_DATA_DIR / "growth_predictor_features.csv")
    
    # Merge on sme_id to get all features
    merged_df = pd.merge(health_df, growth_df, on='sme_id', suffixes=('', '_growth'))
    
    logger.info(f"Testing on {len(merged_df)} SMEs")
    
    # Test single prediction
    sample = merged_df.iloc[0].to_dict()
    result = predict_all(sample, bundles)
    
    logger.info("\n" + "="*60)
    logger.info("SAMPLE PREDICTION OUTPUT")
    logger.info("="*60)
    logger.info(f"\nSME ID: {result['sme_id']}")
    logger.info(f"\nHEALTH PREDICTION:")
    logger.info(f"  Category:   {result['health']['health_category']}")
    logger.info(f"  Risk Flags: {result['health']['risk_flags']}")
    logger.info(f"\nGROWTH PREDICTION:")
    logger.info(f"  6m Growth Rate:     {result['growth']['predicted_6m_growth_rate']}%")
    logger.info(f"  6m Revenue:         KES {result['growth']['predicted_6m_revenue']:,.0f}")
    logger.info(f"  Growth Stage:       {result['growth']['growth_stage']}")
    logger.info(f"  Will Jump Category: {result['growth']['will_jump_category']}")
    logger.info(f"  Recommended Action: {result['growth']['growth_action']}")
    
    logger.info("\n" + "="*60)
    logger.info("This is what POST /api/predict/all will return")
    logger.info("="*60)


if __name__ == "__main__":
    app()

  