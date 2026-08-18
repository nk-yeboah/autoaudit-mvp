"""
detect_anomalies.py

Train an Isolation Forest anomaly detection model against the synthetic
ERP transaction dataset generated in Step 1.

Input:
    transactions.csv

Output:
    flagged_audit_report.csv

The script:
    1. Loads the transaction data.
    2. Engineers timestamp and vendor-frequency features.
    3. One-hot encodes categorical features.
    4. Trains an IsolationForest with contamination=0.03.
    5. Predicts anomalous transactions.
    6. Evaluates predictions against is_fraud_label.
    7. Saves flagged transactions to flagged_audit_report.csv.
"""

import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_FILE = "transactions.csv"
OUTPUT_FILE = "flagged_audit_report.csv"

RANDOM_STATE = 42
CONTAMINATION = 0.03


# ---------------------------------------------------------------------------
# 1. Load transaction data
# ---------------------------------------------------------------------------

print(f"Loading transaction data from: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df):,} transactions.")


# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------

# Convert timestamp to datetime.
df["timestamp"] = pd.to_datetime(df["timestamp"])


# Extract useful time-based features.
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek


# ---------------------------------------------------------------------------
# Vendor transaction frequency
#
# Count how many transactions each vendor has generated within the same
# calendar day. This helps identify duplicate/high-frequency activity.
# ---------------------------------------------------------------------------

df["transaction_date"] = df["timestamp"].dt.date

df["vendor_daily_transaction_count"] = (
    df.groupby(
        ["vendor_name", "transaction_date"]
    )["transaction_id"]
    .transform("count")
)


# Remove the temporary date column because it is not needed by the model.
df.drop(
    columns=["transaction_date"],
    inplace=True,
)


# ---------------------------------------------------------------------------
# 3. Prepare model features
# ---------------------------------------------------------------------------

# Numeric features.
numeric_features = [
    "amount_ghs",
    "hour",
    "day_of_week",
    "vendor_daily_transaction_count",
]


# Categorical features.
categorical_features = [
    "category",
]


# One-hot encode categorical features.
categorical_encoded = pd.get_dummies(
    df[categorical_features],
    prefix=categorical_features,
    dtype=int,
)


# Combine numeric and encoded categorical features.
X = pd.concat(
    [
        df[numeric_features],
        categorical_encoded,
    ],
    axis=1,
)


# Make sure there are no missing values.
X = X.fillna(0)


# ---------------------------------------------------------------------------
# 4. Train Isolation Forest
# ---------------------------------------------------------------------------

print("\nTraining Isolation Forest...")

model = IsolationForest(
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
)

model.fit(X)


# ---------------------------------------------------------------------------
# 5. Generate anomaly predictions
# ---------------------------------------------------------------------------

# IsolationForest returns:
#     -1 = anomaly
#      1 = normal
df["predicted_anomaly"] = model.predict(X)

# Convert Isolation Forest's prediction into a boolean fraud flag.
df["is_predicted_fraud"] = (
    df["predicted_anomaly"] == -1
)


# ---------------------------------------------------------------------------
# 6. Evaluate predictions
# ---------------------------------------------------------------------------

actual = df["is_fraud_label"]
predicted = df["is_predicted_fraud"]


print("\n" + "=" * 70)
print("ANOMALY DETECTION EVALUATION")
print("=" * 70)

print("\nClassification Report:")
print(
    classification_report(
        actual,
        predicted,
        target_names=["Normal", "Fraud"],
        zero_division=0,
    )
)


print("Confusion Matrix:")
cm = confusion_matrix(
    actual,
    predicted,
)

print(cm)


# ---------------------------------------------------------------------------
# Calculate summary metrics
# ---------------------------------------------------------------------------

report = classification_report(
    actual,
    predicted,
    output_dict=True,
    zero_division=0,
)

precision = report["True"]["precision"]
recall = report["True"]["recall"]

correctly_identified_fraud = (
    (actual == True) & (predicted == True)
).sum()


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Precision:                 {precision:.4f}")
print(f"Recall:                    {recall:.4f}")
print(
    f"Correctly identified fraud: {correctly_identified_fraud}"
)
print(
    f"Total actual fraud rows:    {actual.sum()}"
)
print(
    f"Total predicted anomalies:  {predicted.sum()}"
)


# ---------------------------------------------------------------------------
# 7. Save flagged audit report
# ---------------------------------------------------------------------------

flagged = df[
    df["is_predicted_fraud"]
].copy()

flagged.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(
    f"\nFlagged audit report saved to: {OUTPUT_FILE}"
)
print(
    f"Flagged transactions written: {len(flagged):,}"
)