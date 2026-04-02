#!/bin/bash
# build.sh - Runs before the server starts on Railway/Render
# Generates synthetic data, engineers features, and trains all models.
# Only runs training if models don't already exist.

set -e  # Exit immediately if any command fails

GROWTH_MODEL="models/growth_predictor_model/best_growth_predictor_model.pkl"
HEALTH_MODEL="models/health_score_model/health_model.pkl"
FUNDING_MODEL="models/funding_model/best_models_funding.pkl"

echo "======================================"
echo " SME Intelligence API - Build Script"
echo "======================================"

if [ -f "$GROWTH_MODEL" ] && [ -f "$HEALTH_MODEL" ] && [ -f "$FUNDING_MODEL" ]; then
    echo ">>> Models already exist. Skipping training."
else
    echo ">>> Models not found. Starting full training pipeline..."

    echo ""
    echo "[1/6] Generating synthetic SME dataset..."
    python -m machine_learning_360v2.dataset

    echo ""
    echo "[2/6] Engineering health score features..."
    python -m machine_learning_360v2.modeling.Health_score_model.health_score_features

    echo ""
    echo "[3/6] Engineering growth predictor features..."
    python -m machine_learning_360v2.modeling.growth_predictor_model.growth_features

    echo ""
    echo "[4/6] Engineering funding features..."
    python -m machine_learning_360v2.modeling.funding_model.funding_features

    echo ""
    echo "[5/6] Training health score model..."
    python -m machine_learning_360v2.modeling.Health_score_model.health_score_model

    echo ""
    echo "[6/6] Training growth + funding models..."
    python -m machine_learning_360v2.modeling.growth_predictor_model.growth_predictor
    python -m machine_learning_360v2.modeling.funding_model.funding_model

    echo ""
    echo ">>> All models trained successfully."
fi

echo ""
echo ">>> Build complete. Starting API server..."
