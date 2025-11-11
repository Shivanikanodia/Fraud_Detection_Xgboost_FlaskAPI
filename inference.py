import pandas as pd
import matplotlib.pyplot as plt
import pickle

with open('xgb_pipe.pkl', 'rb') as model_file:
 loaded_model = pickle.load(model_file)

with open('preprocessor.pkl', 'rb') as preprocessor_file:
   loaded_preprocessor = pickle.load(preprocessor_file)

import pickle

loaded_model.named_steps['preprocessor'].feature_names_in_

# Make predictions with the loaded model
loaded_xgb_pred = loaded_model.predict(X_test)
loaded_xgb_proba = loaded_model.predict_proba(X_test)[:, 1]

# Evaluate the loaded model
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print("Loaded XGBoost Classifier Evaluation")
print(classification_report(y_test, loaded_xgb_pred))
print("ROC AUC:", roc_auc_score(y_test, loaded_xgb_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, loaded_xgb_pred))

import numpy as np
from sklearn.metrics import (
    average_precision_score, precision_score, recall_score, fbeta_score,
    confusion_matrix
)

# --- PR-AUC ---
pr_auc = average_precision_score(y_test, xgb_proba)
print(f"PR-AUC (Average Precision): {pr_auc:.6f}")

# --- Recall@k (top 500) + Precision@k ---
k = 500
topk_idx = np.argsort(-xgb_proba)[:k]
y_topk = y_test.iloc[topk_idx] if hasattr(y_test, "iloc") else y_test[topk_idx]
precision_at_k = y_topk.mean()                                     # fraction of frauds in top-k
recall_at_k = y_topk.sum() / (y_test.sum() if y_test.sum() > 0 else 1)
print(f"Precision@{k}: {precision_at_k:.4f}")
print(f"Recall@{k}: {recall_at_k:.4f}  (captured {int(y_topk.sum())} of {int(y_test.sum())} frauds)")

# --- Precision at a chosen threshold ---
chosen_thr = 0.50  # <-- set the threshold you want to evaluate (e.g., 0.30, 0.50, etc.)
y_pred_chosen = (xgb_proba >= chosen_thr).astype(int)
prec_chosen = precision_score(y_test, y_pred_chosen, zero_division=0)
rec_chosen  = recall_score(y_test, y_pred_chosen, zero_division=0)
f2_chosen   = fbeta_score(y_test, y_pred_chosen, beta=2, zero_division=0)
print(f"\n@Threshold={chosen_thr:.2f}  Precision={prec_chosen:.3f}  Recall={rec_chosen:.3f}  F2={f2_chosen:.3f}")
print("Confusion Matrix @ chosen threshold:\n", confusion_matrix(y_test, y_pred_chosen))

# --- Tune threshold by maximizing F2 ---
thresholds = np.linspace(0.00, 1.00, 1001)  # fine grid
best_thr, best_f2, best_prec, best_rec, best_pred = 0.0, -1.0, 0.0, 0.0, None

for thr in thresholds:
    y_pred = (xgb_proba >= thr).astype(int)
    f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
    if f2 > best_f2:
        best_thr, best_f2 = thr, f2
        best_prec = precision_score(y_test, y_pred, zero_division=0)
        best_rec  = recall_score(y_test, y_pred, zero_division=0)
        best_pred = y_pred

print(f"\nBest F2 threshold: {best_thr:.4f}  |  Precision={best_prec:.3f}  Recall={best_rec:.3f}  F2={best_f2:.3f}")
print("Confusion Matrix @ best F2 threshold:\n", confusion_matrix(y_test, best_pred))


# Get feature importances
importances = xgb_pipe.feature_importances_
feature_names = xgb_pipe.get_booster().feature_names

# Create DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display top features
print(importance_df.head(10))

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances (XGBoost)")
plt.xlabel("Importance")
plt.show()

import pandas as pd
import pickle


# 2️⃣ Define the columns your model was trained on
expected_cols = [
    'creditLimit', 'currentBalance', 'validity_years', 'validity_days',
    'Trans_Hour', 'Trans_year', 'merchant_avg_amount', 'transactionAmount',
    'acqCountry', 'merchantCountryCode', 'posEntryMode',
    'merchantCategoryCode', 'transactionType', 'merchantName',
    'Trans_month_name', 'cardPresent', 'cvv_match', 'is_night'
]

test_data = pd.DataFrame([
    [5000, 1200, 3, 45, 14, 2025, 250, 300, 'US', 'US', '80', '5411', 'PURCHASE', 'WALMART', 'NOV', 1, 1, 0],
    [10000, 8000, 5, 120, 22, 2025, 400, 850, 'US', 'CA', '80', '5732', 'PURCHASE', 'AMAZON', 'OCT', 1, 1, 1],
    [3000, 200, 2, 10, 9, 2025, 150, 180, 'IN', 'IN', '80', '5812', 'PURCHASE', 'BIGBAZAAR', 'NOV', 1, 1, 0],
    [7000, 3500, 4, 365, 18, 2025, 500, 600, 'GB', 'GB', 'CHIP', '5411', 'PURCHASE', 'TESCO', 'SEP', 1, 1, 0],
    [4000, 3800, 1, 300, 3, 2025, 220, 200, 'US', 'US', 'TAP', '4111', 'PURCHASE', 'UBER', 'AUG', 1, 1, 0],
    [8000, 100, 2, 200, 23, 2025, 600, 700, 'FR', 'FR', 'MANUAL', '5311', 'PURCHASE', 'AMAZON', 'JUL', 0, 0, 1],
    [12000, 9000, 6, 60, 12, 2025, 300, 450, 'US', 'US', 'CHIP', '5814', 'PURCHASE', 'STARBUCKS', 'NOV', 1, 1, 0],
    [6000, 600, 3, 20, 1, 2025, 260, 500, 'CA', 'CA', 'TAP', '5732', 'PURCHASE', 'BESTBUY', 'DEC', 1, 1, 0],
    [9000, 7000, 5, 150, 20, 2025, 420, 1000, 'US', 'US', 'CHIP', '5411', 'PURCHASE', 'TARGET', 'NOV', 1, 1, 0],
    [2000, 1800, 1, 15, 2, 2025, 80, 120, 'IN', 'IN', 'SWIPE', '6011', 'PURCHASE', 'TARGET', 'OCT', 1, 1, 0]
], columns=expected_cols)

# 4️⃣ Ensure correct column order (important if plain model, not pipeline)
test_data = test_data[expected_cols]


# 6️⃣ Make prediction
prediction = loaded_model.predict(test_data)

print("Prediction:", prediction)

"""**Save Model and Trainer as Pickle File**"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, xgb_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC–AUC Curve for XGBoost Classifier")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
