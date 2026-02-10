# compliance_score_sme.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from loguru import logger
import typer
from machine_learning_360v2.config import PROCESSED_DATA_DIR

app = typer.Typer()

# Default paths
INPUT_DEFAULT = PROCESSED_DATA_DIR / "synthetic_onboarding_features.csv"
FUNDING_MODEL_PATH = Path(__file__).parents[1] / "funding_model" / "best_models_funding.pkl"
OUTPUT_DIR = Path(__file__).resolve().parent / "compliance_feature_set"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load trained models & encoders
artifacts = joblib.load(FUNDING_MODEL_PATH)
eligibility_model = artifacts["models"]["eligibility_model"]
risk_model = artifacts["models"]["risk_model"]
health_model = artifacts["models"]["health_model"]
ord_encoder = artifacts["ord_encoder"]
ohe = artifacts["ohe"]
scaler = artifacts["scaler"]
feature_names = artifacts["feature_names"]

def build_feature_vector(df):
    """Transform SME dataframe to ML-ready feature vector."""
    X_ord = ord_encoder.transform(df[feature_names['ordinal']])
    X_ohe = ohe.transform(df[feature_names['onehot']])
    X_num = scaler.transform(df[feature_names['numeric']])
    X_bin = df[feature_names['binary']].values
    return np.hstack([X_ord, X_ohe, X_num, X_bin])

def compliance_decision(p_default, eligible_pred, health_score):
    """Determine compliance risk using simple rule logic."""
    reasons = []

    if not eligible_pred:
        return "HIGH", ["FUNDING_POLICY_VIOLATION"]
    if p_default >= 0.6:
        reasons.append("HIGH_DEFAULT_RISK")
    if health_score < 50:
        reasons.append("POOR_BUSINESS_HEALTH")

    if not reasons:
        return "LOW", ["ALL_OK"]
    elif "HIGH_DEFAULT_RISK" in reasons or "FUNDING_POLICY_VIOLATION" in reasons:
        return "HIGH", reasons
    else:
        return "MEDIUM", reasons

@app.command()
def score(input_path: Path = INPUT_DEFAULT):
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} SMEs from {input_path}")

    results = []

    # Vectorized approach for speed
    X = build_feature_vector(df)
    eligible_preds = eligibility_model.predict(X)
    default_probs = risk_model.predict_proba(X)[:, 1]
    health_scores = health_model.predict(X)

    for idx, row in df.iterrows():
        eligible_pred = int(eligible_preds[idx])
        p_default = float(default_probs[idx])
        health_score = float(health_scores[idx])

        compliance_risk, reasons = compliance_decision(p_default, eligible_pred, health_score)

        results.append({
            "sme_id": row.get("sme_id", f"UNKNOWN_{idx}"),
            "eligible_for_funding": bool(eligible_pred),
            "default_risk_probability": round(p_default, 3),
            "business_health_score": round(health_score, 1),
            "compliance_risk": compliance_risk,
            "reasons": "|".join(reasons)
        })

    output_file = OUTPUT_DIR / "compliance_decisions.csv"
    pd.DataFrame(results).to_csv(output_file, index=False)
    logger.success(f"Compliance decisions saved to {output_file}")
    print(f"{len(results)} SMEs scored and saved to {output_file}")

if __name__ == "__main__":
    app()
