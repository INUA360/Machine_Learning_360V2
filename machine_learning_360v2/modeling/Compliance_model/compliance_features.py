import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from machine_learning_360v2.config import PROCESSED_DATA_DIR
from loguru import logger
from pathlib import Path
import joblib
import typer

app = typer.Typer()

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "synthetic_sme_agents_data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "compliance_feature_set"
):
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
 
    target_cols = ["default_risk", "eligible_for_funding"]
    y_default_risk = df["default_risk"]
    y_eligible_for_funding = df["eligible_for_funding"]
 
    X = df.drop(columns=target_cols + ["sme_id", "business_name"], errors="ignore")

 
    numeric_features = X.select_dtypes(include=["number"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    logger.info(f"Numeric features: {list(numeric_features)}")
    logger.info(f"Categorical features: {list(categorical_features)}")
 
    X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
    X[categorical_features] = X[categorical_features].fillna("unknown")

 
 
    scaler = StandardScaler()
    X_numeric_scaled = pd.DataFrame(
        scaler.fit_transform(X[numeric_features]),
        columns=numeric_features,
        index=X.index
    )

 
    encoder = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore"
    )

    X_categorical_encoded = pd.DataFrame(
        encoder.fit_transform(X[categorical_features]),
        columns=encoder.get_feature_names_out(categorical_features),
        index=X.index
    )

    X_final = pd.concat(
        [X_numeric_scaled, X_categorical_encoded],
        axis=1
    )

    X_final.to_csv(output_path / "X_compliance.csv", index=False)
    y_default_risk.to_csv(output_path / "y_default_risk.csv", index=False)
    y_eligible_for_funding.to_csv(output_path / "y_eligible_for_funding.csv", index=False)

    joblib.dump(scaler, output_path / "numeric_scaler.joblib")
    joblib.dump(encoder, output_path / "categorical_encoder.joblib")

    logger.success("Compliance feature engineering completed successfully")

if __name__ == "__main__":
    app()
