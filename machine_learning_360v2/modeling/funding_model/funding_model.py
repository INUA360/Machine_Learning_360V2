import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from machine_learning_360v2.config import PROCESSED_DATA_DIR 
from lorugu import logger
from pathlib import Pathath
from tdqm import tdqm
import typer

app = typer.Typer()
@app.command()

def main(
    INPUT_PATH: Path = PROCESSED_DATA_DIR/"synthetic_onboarding_features.csv",
    OUTPUT_PATH: Path = PROCESSED_DATA_DIR/"best_models_funding.pkl"
):
    df = pd.read_csv(INPUT_PATH)

    #Encoding the daata
    #ordinal_features (they are ordered and we have to keep that fact alive)
    #chosen teh ordinalencoder

    ordinal_features = ['financial_health', 'profitability', 'payment_behavior',
                    'business_stage', 'operational_efficiency', 'marketing_roi']
    
    categories = [
    ['Stable','Manageable','Risky'],             
    ['Loss-making','Thin Margin','Healthy Margin'], 
    ['Reliable','Occasional','Problematic'],     
    ['Startup','Growth','Mature'],           
    ['Lean','Balanced','High-performing'],     
    ['Low ROI','Moderate ROI','High ROI']   

    ord_encoder = Ordinal_encoder(categories=categories)
    X_ord = ord_encoder.fit_transform(df[ordinal_features])


    #onehot encoding for unordered bins
    unordered_features= ['sector']
    ohe=OneHotEncoder(sparse=False, drop = 'first')
    x_ohe = ohe.fit_transform(df[unordered_features])

    #numerical features
    numeric_features = ['log_revenue', 'profit_per_employee', 'debt_profit_ratio', 'marketing_spend', 'employee_count']
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(df[numeric_features])

    X = np.hstack(X_ord, x_ohe, x_num_scaled)

    #modelling part
    #funding eligibility
    #Funding ammount
    #Risk assessment

    y_eligibility = 

    






