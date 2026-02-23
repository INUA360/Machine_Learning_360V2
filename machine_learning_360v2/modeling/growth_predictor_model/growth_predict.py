from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import typer
import joblib
from machine_learning_360v2.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()

OUTPUT_DIR = Path(__file__).parent / "growth_predictor_feature_set"


def predict_single(sme_data: dict, bundle: dict) -> dict:
    """
    Predict growth for a single SME.
    Input:  dict of SME features
    Output: dict of predictions
    """
    df = pd.DataFrame([sme_data])

    models = bundle['models']
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

    X = np.hstack([X_ord, X_ohe, X_num, X_bin])

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
#the expected output
    {
  'predicted_6m_growth_rate': 18.0,      ← Will grow 18%
  'predicted_6m_revenue': 118000,        ← Revenue becomes $118k
  'growth_stage': 'Growing',             ← In growth phase
  'will_jump_category': True,            ← Jumps to larger bracket
  'current_revenue_category': 'Medium',  ← Currently Medium
  'composite_growth_score': 72.5,        ← Score: 72.5/100
  'growth_action': 'growth_support'      ← Recommendation: Support them
}


@app.command()
def score(
    input_path: Path = PROCESSED_DATA_DIR / "growth_predictor_features.csv",
    model_path: Path = MODELS_DIR / "growth_predictor_model" / "best_growth_predictor_model.pkl",
    output_path: Path = OUTPUT_DIR / "growth_decisions.csv",
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} SMEs from {input_path}")

    models = bundle['models']
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

    X = np.hstack([X_ord, X_ohe, X_num, X_bin])

    all_growth_rates = models['growth_rate_model'].predict(X)
    all_stages = models['growth_stage_model'].predict(X)
    all_jumps = models['category_jump_model'].predict(X)

    results = pd.DataFrame({
        'sme_id': df['sme_id'],
        'current_revenue': df['revenue'],
        'predicted_6m_growth_rate': (all_growth_rates * 100).round(1),
        'predicted_6m_revenue': (df['revenue'] * (1 + all_growth_rates)).round(0),
        'growth_stage': all_stages,
        'will_jump_category': all_jumps,
        'current_revenue_category': df['current_revenue_category'],
        'composite_growth_score': df['composite_growth_score'],
        'growth_action': np.where(
            all_growth_rates >= 0.25, 'fast_track_funding',
            np.where(
                all_growth_rates >= 0.10, 'growth_support',
                np.where(
                    all_growth_rates >= 0, 'stability_support',
                    'intervention_required'
                )
            )
        )
    })

    results.to_csv(output_path, index=False)

    logger.success(f"Growth decisions saved to {output_path}")
    print(f"{len(results)} SMEs scored and saved to {output_path}")

    logger.info(f"\nGrowth Stage Distribution:\n{results['growth_stage'].value_counts()}")
    logger.info(f"\nGrowth Statistics:")
    logger.info(f"  Mean:   {results['predicted_6m_growth_rate'].mean():.1f}%")
    logger.info(f"  Median: {results['predicted_6m_growth_rate'].median():.1f}%")
    logger.info(f"  SMEs jumping category: {results['will_jump_category'].sum()} ({results['will_jump_category'].mean():.1%})")
    logger.info(f"\nRecommended Actions:\n{results['growth_action'].value_counts()}")

    # Test single prediction using first SME as sample
    logger.info("\nSample single prediction (SME #1):")
    sample = df.iloc[0].to_dict()
    result = predict_single(sample, bundle)

    logger.info(f"\nSME ID: {sample.get('sme_id', 'N/A')}")
    logger.info(f"  Predicted 6m Growth:    {result['predicted_6m_growth_rate']}%")
    logger.info(f"  Predicted 6m Revenue:   KES {result['predicted_6m_revenue']:,.0f}")
    logger.info(f"  Growth Stage:           {result['growth_stage']}")
    logger.info(f"  Will Jump Category:     {result['will_jump_category']}")
    logger.info(f"  Current Category:       {result['current_revenue_category']}")
    logger.info(f"  Composite Score:        {result['composite_growth_score']}")
    logger.info(f"  Recommended Action:     {result['growth_action']}")


if __name__ == "__main__":
    app()