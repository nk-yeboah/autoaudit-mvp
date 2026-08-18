"""
main.py

FastAPI backend for the ERP anomaly detection audit system.

Input:
    flagged_audit_report.csv

Run:
    uvicorn main:app --reload
"""

from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "flagged_audit_report.csv"


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ERP Fraud Audit API",
    description="API for reviewing transactions flagged by the anomaly detection model.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# CORS
#
# Allow a React frontend to communicate with this API.
# For production, replace "*" with the actual frontend origin.
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# In-memory transaction store
# ---------------------------------------------------------------------------

transactions_df: pd.DataFrame


@app.on_event("startup")
def load_transactions() -> None:
    """
    Load the flagged audit report into memory when the server starts.
    """

    global transactions_df

    if not CSV_FILE.exists():
        raise RuntimeError(
            f"Required data file not found: {CSV_FILE}"
        )

    transactions_df = pd.read_csv(CSV_FILE)

    # Ensure amount is numeric so filtering works correctly.
    if "amount_ghs" in transactions_df.columns:
        transactions_df["amount_ghs"] = pd.to_numeric(
            transactions_df["amount_ghs"],
            errors="coerce",
        )

    print(
        f"Loaded {len(transactions_df):,} flagged transactions "
        f"from {CSV_FILE.name}"
    )


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class AuditAction(BaseModel):
    transaction_id: str
    action: Literal["approve", "flag"]


# ---------------------------------------------------------------------------
# Endpoint 1: Health/status check
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """
    Return API welcome message and status.
    """

    return {
        "message": "Welcome to the ERP Fraud Audit API",
        "status": "healthy",
        "records_loaded": len(transactions_df),
    }


# ---------------------------------------------------------------------------
# Endpoint 2: Get flagged anomalies
# ---------------------------------------------------------------------------

@app.get("/api/anomalies")
def get_anomalies(
    category: Optional[str] = Query(
        default=None,
        description="Filter anomalies by transaction category.",
    ),
    min_amount: Optional[float] = Query(
        default=None,
        description="Return only transactions with amount_ghs >= min_amount.",
        ge=0,
    ),
):
    """
    Return flagged fraud transactions.

    Optional filters:
        ?category=Logistics
        ?min_amount=10000
        ?category=IT%20Equipment&min_amount=5000
    """

    filtered_df = transactions_df.copy()

    if category:
        filtered_df = filtered_df[
            filtered_df["category"].str.lower()
            == category.lower()
        ]

    if min_amount is not None:
        filtered_df = filtered_df[
            filtered_df["amount_ghs"] >= min_amount
        ]

    # Convert NaN values to None-compatible JSON values.
    records = filtered_df.where(
        pd.notna(filtered_df),
        None,
    ).to_dict(orient="records")

    return records


# ---------------------------------------------------------------------------
# Endpoint 3: Approve / flag an audit transaction
# ---------------------------------------------------------------------------

@app.post("/api/audit/approve")
def audit_transaction(payload: AuditAction):
    """
    Update the audit status of a flagged transaction.

    Supported actions:
        approve -> transaction marked as Approved
        flag    -> transaction marked as Flagged
    """

    global transactions_df

    matches = transactions_df.index[
        transactions_df["transaction_id"].astype(str)
        == payload.transaction_id
    ]

    if len(matches) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction '{payload.transaction_id}' not found.",
        )

    index = matches[0]

    new_status = (
        "Approved"
        if payload.action == "approve"
        else "Flagged"
    )

    # Add/update an audit status column.
    transactions_df.loc[index, "audit_status"] = new_status

    return {
        "success": True,
        "transaction_id": payload.transaction_id,
        "action": payload.action,
        "audit_status": new_status,
        "message": (
            f"Transaction {payload.transaction_id} "
            f"has been marked as {new_status}."
        ),
    }