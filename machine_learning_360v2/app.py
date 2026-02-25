"""
╔══════════════════════════════════════════════════════════════╗
║               SME Intelligence Flask API                     ║
║  Wraps: growth, funding, health, compliance predictions      ║
╚══════════════════════════════════════════════════════════════╝

SYNTAX TIPS & TRICKS (read these — they'll save you hours)
─────────────────────────────────────────────────────────────
① @app.route('/path', methods=['POST'])
    ↑ The decorator "registers" a URL with Flask.
    ↑ Always declare methods=['POST'] for data-receiving endpoints.

② request.get_json()
    ↑ Reads the incoming JSON body. Returns None if Content-Type
      header is missing "application/json".
    ↑ Safer alternative: request.get_json(force=True, silent=True)
      force=True  → parse even if Content-Type is wrong
      silent=True → return None instead of raising an error

③ jsonify({...})
    ↑ Converts a Python dict → proper JSON HTTP response.
    ↑ Also sets Content-Type: application/json automatically.

④ abort(400) / abort(404)
    ↑ Immediately stops the request and returns an HTTP error code.
    ↑ Use it for guard clauses at the top of your route functions.

⑤ app.errorhandler(404)
    ↑ A decorator that catches unhandled HTTP errors globally.
    ↑ Keeps error responses consistent across the whole API.

⑥ Blueprint   (used in larger apps, not here but good to know)
    ↑ Like a "mini Flask app" — groups routes by feature.
    ↑ e.g.  growth_bp = Blueprint('growth', __name__)
"""

# ─────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from loguru import logger

from flask import Flask, request, jsonify, abort

# Your project config  ← adjust if paths differ
from machine_learning_360v2.config import MODELS_DIR, PROCESSED_DATA_DIR


# ─────────────────────────────────────────────
#  APP INITIALISATION
#  Flask(__name__)  tells Flask "this file is the root of the app"
#  __name__ == "__main__" when run directly, else the module name.
# ─────────────────────────────────────────────
app = Flask(__name__)


# ═══════════════════════════════════════════════════════════════
#  MODEL LOADING  (happens ONCE at startup, not per request)
#  Keeping models in module-level variables is the standard
#  pattern — loading from disk on every request would be ~10×
#  slower.
# ═══════════════════════════════════════════════════════════════

GROWTH_MODEL_PATH   = MODELS_DIR / "growth_predictor_model" / "best_growth_predictor_model.pkl"
HEALTH_MODEL_PATH   = MODELS_DIR / "health_score_model"     / "health_model.pkl"
FUNDING_MODEL_PATH  = MODELS_DIR / "funding_model"          / "best_models_funding.pkl"

logger.info("Loading models …")
growth_bundle  = joblib.load(GROWTH_MODEL_PATH)
health_bundle  = joblib.load(HEALTH_MODEL_PATH)
funding_bundle = joblib.load(FUNDING_MODEL_PATH)
logger.success("All models loaded ✓")


# ═══════════════════════════════════════════════════════════════
#  HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════

RISK_FLAG_COLS = [
    "low_cash", "high_debt", "low_profit",
    "missing_docs", "tax_noncompliant", "license_expired",
]

# ── Helper: validate that the request body is present ─────────
def get_body() -> dict:
    """
    Pull JSON from the request.
    Raises HTTP 400 if body is missing or not valid JSON.

    TRICK: Always call this at the top of POST handlers.
           It gives users a clear error instead of a cryptic crash.
    """
    body = request.get_json(force=True, silent=True)
    if not body:
        abort(400, description="Request body must be valid JSON.")
    return body


# ── Feature builders (mirror your existing pipeline) ──────────
def _build_growth_features(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    enc = bundle["encoders"]
    fn  = bundle["feature_names"]
    X_ord = enc["ord_encoder"].transform(df[[f for f in fn["ordinal"] if f in df.columns]])
    X_ohe = enc["ohe"].transform(        df[[f for f in fn["onehot"]  if f in df.columns]])
    X_num = enc["scaler"].transform(     df[[f for f in fn["numeric"] if f in df.columns]])
    X_bin =                              df[[f for f in fn["binary"]  if f in df.columns]].values
    return np.hstack([X_ord, X_ohe, X_num, X_bin])


def _build_health_features(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    enc = bundle["encoders"]
    fn  = bundle["feature_names"]
    X_ord = enc["ord_encoder"].transform(df[[f for f in fn["ordinal"]     if f in df.columns]])
    X_ohe = enc["ohe"].transform(        df[[f for f in fn["categorical"] if f in df.columns]])
    X_num = enc["scaler"].transform(     df[[f for f in fn["numeric"]     if f in df.columns]])
    X_bin =                              df[[f for f in fn["binary"]      if f in df.columns]].values
    return np.hstack([X_ord, X_ohe, X_num, X_bin])


def _build_funding_features(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    fn = bundle["feature_names"]
    X_ord = bundle["ord_encoder"].transform(df[fn["ordinal"]])
    X_ohe = bundle["ohe"].transform(        df[fn["onehot"]])
    X_num = bundle["scaler"].transform(     df[fn["numeric"]])
    X_bin =                                 df[fn["binary"]].values
    return np.hstack([X_ord, X_ohe, X_num, X_bin])


# ── Growth action label ────────────────────────────────────────
def _growth_action(rate: float) -> str:
    if rate >= 0.25: return "fast_track_funding"
    if rate >= 0.10: return "growth_support"
    if rate >= 0.00: return "stability_support"
    return "intervention_required"


# ── Compliance decision rule ───────────────────────────────────
def _compliance_decision(p_default: float, eligible: bool, health_score: float):
    if not eligible:
        return "HIGH", ["FUNDING_POLICY_VIOLATION"]
    reasons = []
    if p_default >= 0.6:  reasons.append("HIGH_DEFAULT_RISK")
    if health_score < 50: reasons.append("POOR_BUSINESS_HEALTH")
    if not reasons:
        return "LOW", ["ALL_OK"]
    if "HIGH_DEFAULT_RISK" in reasons:
        return "HIGH", reasons
    return "MEDIUM", reasons


# ═══════════════════════════════════════════════════════════════
#  ROUTES
#  Convention: prefix all routes with /api/v1/ so you can
#  version the API later without breaking existing clients.
# ═══════════════════════════════════════════════════════════════

# ── Health check ───────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    """
    GET /
    Quick ping to confirm the server is alive.
    No auth, no body needed.
    """
    return jsonify({
        "service": "SME Intelligence API",
        "status": "online",
        "endpoints": [
            "POST /api/v1/predict/growth",
            "POST /api/v1/predict/health",
            "POST /api/v1/predict/funding",
            "POST /api/v1/predict/compliance",
            "POST /api/v1/predict/all",
        ]
    })


# ── 1. Growth Prediction ───────────────────────────────────────
@app.route("/api/v1/predict/growth", methods=["POST"])
def predict_growth():
    """
    POST /api/v1/predict/growth
    Body: { SME feature dict }

    Returns 6-month growth forecast.

    SYNTAX NOTE:
        pd.DataFrame([sme_data])   ← wrapping a dict in a list
        creates a single-row DataFrame — the standard trick for
        using batch-trained sklearn models on one sample.
    """
    sme_data = get_body()

    df = pd.DataFrame([sme_data])
    X  = _build_growth_features(df, growth_bundle)

    models      = growth_bundle["models"]
    growth_rate = float(models["growth_rate_model"].predict(X)[0])
    stage       = models["growth_stage_model"].predict(X)[0]
    will_jump   = bool(models["category_jump_model"].predict(X)[0])

    current_revenue = float(sme_data.get("revenue", 0))

    result = {
        "sme_id":                    sme_data.get("sme_id", "unknown"),
        "predicted_6m_growth_rate":  round(growth_rate * 100, 1),
        "predicted_6m_revenue":      round(current_revenue * (1 + growth_rate), 0),
        "growth_stage":              stage,
        "will_jump_category":        will_jump,
        "current_revenue_category":  sme_data.get("current_revenue_category", "Unknown"),
        "composite_growth_score":    round(float(sme_data.get("composite_growth_score", 0)), 1),
        "growth_action":             _growth_action(growth_rate),
    }

    logger.info(f"[growth] sme={result['sme_id']} rate={result['predicted_6m_growth_rate']}%")
    return jsonify(result), 200


# ── 2. Health Prediction ───────────────────────────────────────
@app.route("/api/v1/predict/health", methods=["POST"])
def predict_health():
    """
    POST /api/v1/predict/health
    Body: { SME feature dict }

    Returns health category + risk flags.

    SYNTAX NOTE:
        int(sme_data.get(col, 0))
        ↑ .get(key, default) is safer than sme_data[key] because
          it won't raise KeyError if the field is missing.
    """
    sme_data = get_body()

    df       = pd.DataFrame([sme_data])
    X        = _build_health_features(df, health_bundle)
    category = health_bundle["model"].predict(X)[0]

    risk_flags          = {col: int(sme_data.get(col, 0)) for col in RISK_FLAG_COLS}
    risk_flags["total"] = sum(risk_flags.values())

    result = {
        "sme_id":          sme_data.get("sme_id", "unknown"),
        "health_category": category,
        "risk_flags":      risk_flags,
    }

    logger.info(f"[health] sme={result['sme_id']} category={category}")
    return jsonify(result), 200


# ── 3. Funding Prediction ──────────────────────────────────────
@app.route("/api/v1/predict/funding", methods=["POST"])
def predict_funding():
    """
    POST /api/v1/predict/funding
    Body: { SME feature dict }

    Returns eligibility, default risk, and business health score.

    SYNTAX NOTE:
        model.predict_proba(X)[:, 1]
        ↑ For binary classifiers, predict_proba returns shape
          (n_samples, 2) — column 0 = P(class 0), column 1 = P(class 1).
          [:, 1] slices out just the positive-class probability.
    """
    sme_data = get_body()

    df      = pd.DataFrame([sme_data])
    X       = _build_funding_features(df, funding_bundle)
    models  = funding_bundle["models"]

    eligible = bool(models["eligibility_model"].predict(X)[0])
    risk     = bool(models["risk_model"].predict(X)[0])
    health   = float(models["health_model"].predict(X)[0])

    result = {
        "sme_id":                sme_data.get("sme_id", "unknown"),
        "eligible_for_funding":  eligible,
        "default_risk":          risk,
        "business_health_score": round(health, 2),
    }

    logger.info(f"[funding] sme={result['sme_id']} eligible={eligible} risk={risk}")
    return jsonify(result), 200


# ── 4. Compliance Scoring ──────────────────────────────────────
@app.route("/api/v1/predict/compliance", methods=["POST"])
def predict_compliance():
    """
    POST /api/v1/predict/compliance
    Body: { SME feature dict }

    Returns compliance risk level + reason codes.

    SYNTAX NOTE:
        "|".join(reasons)
        ↑ Joins a list into a pipe-delimited string — handy for
          storing multi-value fields in a single CSV column.
          Split back with reasons.split("|").
    """
    sme_data = get_body()

    df             = pd.DataFrame([sme_data])
    X              = _build_funding_features(df, funding_bundle)   # compliance reuses funding bundle
    models         = funding_bundle["models"]

    eligible    = bool(models["eligibility_model"].predict(X)[0])
    p_default   = float(models["risk_model"].predict_proba(X)[0, 1])   # probability, not binary
    health      = float(models["health_model"].predict(X)[0])

    risk_level, reasons = _compliance_decision(p_default, eligible, health)

    result = {
        "sme_id":                    sme_data.get("sme_id", "unknown"),
        "eligible_for_funding":      eligible,
        "default_risk_probability":  round(p_default, 3),
        "business_health_score":     round(health, 1),
        "compliance_risk":           risk_level,
        "reasons":                   reasons,       # list — cleaner than pipe string in JSON
    }

    logger.info(f"[compliance] sme={result['sme_id']} risk={risk_level}")
    return jsonify(result), 200


# ── 5. All-in-one Prediction ───────────────────────────────────
@app.route("/api/v1/predict/all", methods=["POST"])
def predict_all():
    """
    POST /api/v1/predict/all
    Body: { SME feature dict }

    Runs all four models in one request.
    Useful for dashboards that need the full picture at once.

    SYNTAX NOTE:
        with app.test_request_context():  ← (testing trick)
        You can call route functions directly in unit tests by
        pushing a fake request context — no HTTP server needed.
    """
    sme_data = get_body()
    sme_id   = sme_data.get("sme_id", "unknown")

    # ── growth ──────────────────────────────────
    df_g        = pd.DataFrame([sme_data])
    X_g         = _build_growth_features(df_g, growth_bundle)
    g_models    = growth_bundle["models"]
    growth_rate = float(g_models["growth_rate_model"].predict(X_g)[0])
    current_rev = float(sme_data.get("revenue", 0))

    growth_result = {
        "predicted_6m_growth_rate": round(growth_rate * 100, 1),
        "predicted_6m_revenue":     round(current_rev * (1 + growth_rate), 0),
        "growth_stage":             g_models["growth_stage_model"].predict(X_g)[0],
        "will_jump_category":       bool(g_models["category_jump_model"].predict(X_g)[0]),
        "current_revenue_category": sme_data.get("current_revenue_category", "Unknown"),
        "composite_growth_score":   round(float(sme_data.get("composite_growth_score", 0)), 1),
        "growth_action":            _growth_action(growth_rate),
    }

    # ── health ──────────────────────────────────
    df_h     = pd.DataFrame([sme_data])
    X_h      = _build_health_features(df_h, health_bundle)
    category = health_bundle["model"].predict(X_h)[0]
    flags    = {col: int(sme_data.get(col, 0)) for col in RISK_FLAG_COLS}
    flags["total"] = sum(flags.values())

    health_result = {"health_category": category, "risk_flags": flags}

    # ── funding ─────────────────────────────────
    df_f    = pd.DataFrame([sme_data])
    X_f     = _build_funding_features(df_f, funding_bundle)
    f_m     = funding_bundle["models"]
    eligible = bool(f_m["eligibility_model"].predict(X_f)[0])
    risk     = bool(f_m["risk_model"].predict(X_f)[0])
    health_s = float(f_m["health_model"].predict(X_f)[0])

    funding_result = {
        "eligible_for_funding":  eligible,
        "default_risk":          risk,
        "business_health_score": round(health_s, 2),
    }

    # ── compliance ──────────────────────────────
    p_default = float(f_m["risk_model"].predict_proba(X_f)[0, 1])
    risk_level, reasons = _compliance_decision(p_default, eligible, health_s)

    compliance_result = {
        "default_risk_probability": round(p_default, 3),
        "compliance_risk":          risk_level,
        "reasons":                  reasons,
    }

    result = {
        "sme_id":     sme_id,
        "growth":     growth_result,
        "health":     health_result,
        "funding":    funding_result,
        "compliance": compliance_result,
    }

    logger.info(f"[all] sme={sme_id} done")
    return jsonify(result), 200


# ═══════════════════════════════════════════════════════════════
#  GLOBAL ERROR HANDLERS
#  SYNTAX NOTE:
#    @app.errorhandler(code) catches that HTTP error ANYWHERE
#    in the app. Without this, Flask returns HTML error pages —
#    not great for a JSON API.
# ═══════════════════════════════════════════════════════════════

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad Request", "message": str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found", "message": str(e)}), 404

@app.errorhandler(500)
def server_error(e):
    logger.exception(e)
    return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
#  SYNTAX NOTE:
#    debug=True   → auto-reloads on file save + shows traceback
#                   in browser. NEVER use in production.
#    host="0.0.0.0" → listens on all network interfaces,
#                   needed if running inside Docker or a VM.
#    port=5000    → default Flask port. Change if taken.
#
#  Run from terminal:
#    python flask_api.py
#  Or with Flask CLI (preferred):
#    flask --app flask_api run --debug
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


# ─────────────────────────────────────────────────────────────
#  HOW TO CALL THE API  (curl cheat-sheet)
# ─────────────────────────────────────────────────────────────
# Health check:
#   curl http://localhost:5000/
#
# Growth prediction:
#   curl -X POST http://localhost:5000/api/v1/predict/growth \
#        -H "Content-Type: application/json" \
#        -d '{"sme_id":"SME_001","revenue":100000,"profit_margin":0.20,...}'
#
# All predictions at once:
#   curl -X POST http://localhost:5000/api/v1/predict/all \
#        -H "Content-Type: application/json" \
#        -d @sample_sme.json          ← reads body from a file
#
# Python requests equivalent:
#   import requests
#   r = requests.post(
#       "http://localhost:5000/api/v1/predict/growth",
#       json=sme_dict          # 'json=' auto-sets Content-Type header
#   )
#   print(r.json())