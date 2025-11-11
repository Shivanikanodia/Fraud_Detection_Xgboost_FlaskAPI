from flask import Flask, request, jsonify
import pickle, pandas as pd

app = Flask(__name__)

# Load ONE artifact: the full pipeline (preprocessor + model)
pipeline = pickle.load(open('xgb_pipe.pkl', 'rb'))

# Keep your expected raw columns in order
EXPECTED_COLS = [
    'creditLimit', 'currentBalance', 'validity_years', 'validity_days',
    'Trans_Hour', 'Trans_year', 'merchant_avg_amount', 'transactionAmount',
    'acqCountry', 'merchantCountryCode', 'posEntryMode',
    'merchantCategoryCode', 'transactionType', 'merchantName',
    'Trans_month_name', 'cardPresent', 'cvv_match', 'is_night'
]

@app.get("/")
def home():
    return "API is running. Try GET /health or POST /predict"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify(error="Request must be JSON"), 400
    thr = float(request.args.get("threshold", 0.2)) 
    rows = payload if isinstance(payload, list) else [payload]

    X_raw = pd.DataFrame(rows)

    # Ensure correct column order and presence
    missing = [c for c in EXPECTED_COLS if c not in X_raw.columns]
    if missing:
        return jsonify(error=f"Missing columns: {missing}"), 400
    X_raw = X_raw[EXPECTED_COLS]

    # Let the pipeline handle preprocessing internally
    probs = pipeline.predict_proba(X_raw)[:, 1]
    preds = (probs >= thr).astype(int)
    out = {"predictions": preds}

    return jsonify(
        threshold=thr,
        predictions=preds.tolist(),
        probabilities=[float(round(p, 4)) for p in probs]
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)



