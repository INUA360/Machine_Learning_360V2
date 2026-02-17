from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import typer
import joblib
from machine_learning_360v2.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()

OUTPUT_DIR = Path(__file__).parent / "health_score_feature_set"

RISK_FLAG_COLS = [
    'low_cash', 'high_debt', 'low_profit',
    'missing_docs', 'tax_noncompliant', 'license_expired'
]


def _build_feature_matrix(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    """
    Shared helper to build the feature matrix from a dataframe.
    Used by both predict_single and the batch score command.
    """
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


def predict_single(sme_data: dict, bundle: dict) -> dict:
    """
    Predict health category for a single SME.
    This is the core logic that will be wrapped in a Flask endpoint.
    Input:  dict of SME features
    Output: dict of predictions
    """
    df = pd.DataFrame([sme_data])

    model = bundle['model']
    X = _build_feature_matrix(df, bundle)
    category = model.predict(X)[0]

    risk_flags = {col: int(sme_data.get(col, 0)) for col in RISK_FLAG_COLS}
    risk_flags['total'] = sum(risk_flags.values())

    return {
        'health_category': category,
        'risk_flags': risk_flags
    }


@app.command()
def score(
    input_path: Path = PROCESSED_DATA_DIR / "health_score_features.csv",
    model_path: Path = MODELS_DIR / "health_score_model" / "health_model.pkl",
    output_path: Path = OUTPUT_DIR / "health_decisions.csv",
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} SMEs from {input_path}")

    model = bundle['model']
    X = _build_feature_matrix(df, bundle)
    all_categories = model.predict(X)

    results = pd.DataFrame({
        'sme_id': df['sme_id'],
        'health_category': all_categories,
        'low_cash': df['low_cash'],
        'high_debt': df['high_debt'],
        'low_profit': df['low_profit'],
        'missing_docs': df['missing_docs'],
        'tax_noncompliant': df['tax_noncompliant'],
        'license_expired': df['license_expired'],
        'total_risk_flags': df[RISK_FLAG_COLS].sum(axis=1),
    })

    results.to_csv(output_path, index=False)

    logger.success(f"Health decisions saved to {output_path}")
    print(f"{len(results)} SMEs scored and saved to {output_path}")

    logger.info(f"\nHealth Category Distribution:\n{results['health_category'].value_counts()}")
    logger.info(f"\nRisk Summary:")
    logger.info(f"  0 flags:   {(results['total_risk_flags'] == 0).sum()} SMEs")
    logger.info(f"  1-2 flags: {((results['total_risk_flags'] >= 1) & (results['total_risk_flags'] <= 2)).sum()} SMEs")
    logger.info(f"  3+ flags:  {(results['total_risk_flags'] >= 3).sum()} SMEs")

    # Test single prediction using first SME as sample
    logger.info("\nSample single prediction (SME #1):")
    sample = df.iloc[0].to_dict()
    result = predict_single(sample, bundle)

    logger.info(f"\nSME ID: {sample.get('sme_id', 'N/A')}")
    logger.info(f"  Category:   {result['health_category']}")
    logger.info(f"  Risk Flags: {result['risk_flags']}")


if __name__ == "__main__":
    app()