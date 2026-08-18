"""
Generate a synthetic ERP transaction dataset with injected fraud anomalies.

Output:
    transactions.csv

Final dataset:
    1,000 records
    30 fraud-labelled records

Injected anomaly patterns:
    - 10 duplicate invoice copies
    - 10 split-payment transactions
    - 10 outlier-amount transactions

Note:
    Duplicate invoices require 20 physical rows:
    10 legitimate originals + 10 duplicate copies.
    Only the 10 duplicate copies are labelled as fraud.
"""

import uuid
from datetime import datetime, timedelta

import pandas as pd
from faker import Faker


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

fake = Faker()
Faker.seed(42)

NUM_RECORDS = 1_000

# Number of explicitly injected fraud-labelled rows.
NUM_DUPLICATE_FRAUD = 10
NUM_SPLIT_PAYMENTS = 10
NUM_OUTLIERS = 10

# Duplicate anomalies require an original + duplicate row.
NUM_DUPLICATE_ROWS = NUM_DUPLICATE_FRAUD * 2

TOTAL_INJECTED_ROWS = (
    NUM_DUPLICATE_ROWS
    + NUM_SPLIT_PAYMENTS
    + NUM_OUTLIERS
)

# 20 duplicate rows + 10 split payments + 10 outliers = 40 rows.
BASE_RECORDS = NUM_RECORDS - TOTAL_INJECTED_ROWS

OUTPUT_FILE = "transactions.csv"

CATEGORIES = [
    "Logistics",
    "Office Supplies",
    "IT Equipment",
    "Marketing",
]

APPROVAL_STATUSES = [
    "Approved",
    "Pending",
    "Flagged",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def random_timestamp(days_back=90):
    """Return a random timestamp within the last 90 days."""
    end = datetime.now()
    start = end - timedelta(days=days_back)

    random_seconds = fake.random_int(
        min=0,
        max=int((end - start).total_seconds()),
    )

    return start + timedelta(seconds=random_seconds)


def create_transaction(
    vendor_name=None,
    amount=None,
    timestamp=None,
    approval_status=None,
    category=None,
    is_fraud=False,
):
    """Create a single ERP transaction record."""

    if amount is None:
        amount = fake.pyfloat(
            min_value=100.00,
            max_value=50_000.00,
            right_digits=2,
        )

    return {
        "transaction_id": str(uuid.uuid4()),
        "vendor_name": vendor_name or fake.company(),
        "amount_ghs": round(float(amount), 2),
        "timestamp": (
            timestamp or random_timestamp()
        ).isoformat(timespec="seconds"),
        "approval_status": approval_status or fake.random_element(
            APPROVAL_STATUSES
        ),
        "category": category or fake.random_element(CATEGORIES),
        "is_fraud_label": bool(is_fraud),
    }


# ---------------------------------------------------------------------------
# 1. Generate normal/base transactions
# ---------------------------------------------------------------------------

transactions = []

for _ in range(BASE_RECORDS):
    transactions.append(create_transaction())


# ---------------------------------------------------------------------------
# 2. Inject Duplicate Invoice Anomalies
#
# Create 10 legitimate transactions and then create an exact duplicate
# of each within a 12-hour window.
#
# The original transaction is NOT fraud-labelled.
# The duplicate transaction IS fraud-labelled.
#
# This contributes:
#     10 original rows
#     10 duplicate rows
#     = 20 rows total
#     = 10 fraud labels
# ---------------------------------------------------------------------------

for _ in range(NUM_DUPLICATE_FRAUD):
    vendor = fake.company()

    amount = round(
        fake.pyfloat(
            min_value=100.00,
            max_value=50_000.00,
            right_digits=2,
        ),
        2,
    )

    base_time = random_timestamp()

    duplicate_time = base_time + timedelta(
        hours=fake.random_int(min=1, max=12)
    )

    # Legitimate original invoice.
    transactions.append(
        create_transaction(
            vendor_name=vendor,
            amount=amount,
            timestamp=base_time,
            is_fraud=False,
        )
    )

    # Fraudulent duplicate invoice.
    transactions.append(
        create_transaction(
            vendor_name=vendor,
            amount=amount,
            timestamp=duplicate_time,
            is_fraud=True,
        )
    )


# ---------------------------------------------------------------------------
# 3. Inject Split Payment Anomalies
#
# Ten transactions:
#     - Same vendor
#     - Same calendar day
#     - Each just below GHS 5,000
#
# These contribute 10 fraud-labelled rows.
# ---------------------------------------------------------------------------

split_vendor = fake.company()

split_date = random_timestamp().replace(
    hour=9,
    minute=0,
    second=0,
    microsecond=0,
)

for i in range(NUM_SPLIT_PAYMENTS):
    amount = fake.random_int(
        min=4_900,
        max=4_999,
    )

    transaction_time = split_date + timedelta(
        minutes=i * 15
    )

    transactions.append(
        create_transaction(
            vendor_name=split_vendor,
            amount=amount,
            timestamp=transaction_time,
            is_fraud=True,
        )
    )


# ---------------------------------------------------------------------------
# 4. Inject Outlier Amount Anomalies
#
# Ten transactions have amounts greater than GHS 250,000.
#
# These contribute 10 fraud-labelled rows.
# ---------------------------------------------------------------------------

for _ in range(NUM_OUTLIERS):
    amount = fake.pyfloat(
        min_value=250_001.00,
        max_value=500_000.00,
        right_digits=2,
    )

    transactions.append(
        create_transaction(
            amount=amount,
            is_fraud=True,
        )
    )


# ---------------------------------------------------------------------------
# 5. Create DataFrame
# ---------------------------------------------------------------------------

df = pd.DataFrame(transactions)


# ---------------------------------------------------------------------------
# 6. Validation
# ---------------------------------------------------------------------------

expected_columns = [
    "transaction_id",
    "vendor_name",
    "amount_ghs",
    "timestamp",
    "approval_status",
    "category",
    "is_fraud_label",
]

assert len(df) == NUM_RECORDS, (
    f"Expected {NUM_RECORDS} records, got {len(df)}"
)

assert list(df.columns) == expected_columns

assert df["transaction_id"].is_unique

assert df["amount_ghs"].between(
    100.00,
    500_000.00,
).all()

assert set(df["approval_status"]).issubset(
    set(APPROVAL_STATUSES)
)

assert set(df["category"]).issubset(
    set(CATEGORIES)
)

# Exactly 30 rows should be labelled as fraudulent.
expected_fraud_labels = (
    NUM_DUPLICATE_FRAUD
    + NUM_SPLIT_PAYMENTS
    + NUM_OUTLIERS
)

assert df["is_fraud_label"].sum() == expected_fraud_labels, (
    f"Expected {expected_fraud_labels} fraud labels, "
    f"got {df['is_fraud_label'].sum()}"
)


# ---------------------------------------------------------------------------
# 7. Sort chronologically
# ---------------------------------------------------------------------------

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = (
    df
    .sort_values("timestamp")
    .reset_index(drop=True)
)

# Store timestamps as ISO datetime strings in the CSV.
df["timestamp"] = df["timestamp"].dt.strftime(
    "%Y-%m-%dT%H:%M:%S"
)


# ---------------------------------------------------------------------------
# 8. Write CSV
# ---------------------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ---------------------------------------------------------------------------
# 9. Final summary
# ---------------------------------------------------------------------------

print(f"Successfully generated {len(df):,} transactions.")
print(f"Fraud-labelled transactions: {df['is_fraud_label'].sum():,}")
print(f"Output file: {OUTPUT_FILE}")

print("\nDataset breakdown:")
print(f"  Base transactions:       {BASE_RECORDS}")
print(f"  Duplicate anomaly rows:  {NUM_DUPLICATE_ROWS}")
print(f"  Split-payment rows:      {NUM_SPLIT_PAYMENTS}")
print(f"  Outlier rows:             {NUM_OUTLIERS}")
print(f"  Total rows:               {len(df)}")

print("\nFraud-label breakdown:")
print(f"  Duplicate invoices:      {NUM_DUPLICATE_FRAUD}")
print(f"  Split payments:          {NUM_SPLIT_PAYMENTS}")
print(f"  Outlier amounts:         {NUM_OUTLIERS}")
print(f"  Total fraud labels:      {df['is_fraud_label'].sum()}")