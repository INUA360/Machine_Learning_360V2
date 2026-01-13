import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from machine_learning_360v2.config import PROCESSED_DATA_DIR,mo
from loguru import logger
from pathlib import Path
import joblib 
import typer

app = typer.Typer()

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR/"sythetic_sme_agents",
    output_path: Path = PROCESSED_DATA_DIR/"compliace_feature_set"
):
    df = pd.read_csv(input_path)
    logger.info("The firts firve features of the dataset are: {df.head}")

    target_cols = ['default_risk', 'eligible_for_funding']

    y_default_risk = df['default_risk']
    y_eligible_for_funding = df['eligible_for_funding']

    X= df.drop(columns = target_cols +["sme_id","business_name"], errors="ignore")

    numeric_features = df.select_dtypes(include=['number'])
    logger.info("The numeric features are {numeric_features}")
    
    categorical_features = df.select_dtypes(include = 'object')
    logger.info("The categorical features are: {categorical_features}")

    X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
    X[categorical_features] = X[categorical_features].fillna('unknown')
    
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(numeric_features)

    X_numeric_scaled = pd.DataFrame(
        X_numeric_scaled,
        columns = numeric_features,
        index=X.index
    )

    encoder = OneHotEncoder()
    X_categorical_encoded = encoder.fit_transform(X[categorical_features])
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)

    X_categorical_encoded = pd.DataFrame(
        X_categorical_encoded,
        columns = encoded_feature_names,
        index = X.index
    )

    X_final = pd.concat([X_numeric_scaled, X_categorical_encoded], axis = 1)

    X_final.to_csv(output_path/ "X_compliance.csv", index = False)
    y_default_risk(output_path/"y_default_risk.csv", index = False)
    y_eligible_for_funding(output_path/"y_eligible_for_funding.csv", index = False)

    joblib.dump(scaler, output_path/ "numeric_scaler.joblib")
    joblib.dump(encoder, output_path/ "categorical_encoder.csv", index = False)

    #Rule enf

    if __name__ == "__main__":
        app()


    