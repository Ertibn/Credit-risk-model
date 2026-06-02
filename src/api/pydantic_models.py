"""
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """Request model for risk prediction."""
    
    customer_id: str = Field(..., description="Unique customer identifier")
    recency: int = Field(..., description="Days since last transaction", ge=0)
    frequency: int = Field(..., description="Number of transactions", ge=0)
    monetary: float = Field(..., description="Total transaction amount", ge=0)
    avg_transaction_amount: float = Field(
        ..., 
        description="Average transaction amount",
        ge=0
    )
    fraud_count: int = Field(
        default=0,
        description="Number of fraudulent transactions",
        ge=0
    )
    transaction_count: int = Field(
        ...,
        description="Total transaction count",
        ge=1
    )
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "recency": 30,
                "frequency": 10,
                "monetary": 5000.0,
                "avg_transaction_amount": 500.0,
                "fraud_count": 0,
                "transaction_count": 10
            }
        }


class PredictionResponse(BaseModel):
    """Response model for risk prediction."""
    
    customer_id: str = Field(..., description="Customer identifier")
    risk_probability: float = Field(
        ...,
        description="Credit risk probability (0-1)",
        ge=0.0,
        le=1.0
    )
    risk_classification: str = Field(
        ...,
        description="Risk category: LOW, MEDIUM, or HIGH"
    )
    credit_score: int = Field(
        ...,
        description="Credit score (0-100)",
        ge=0,
        le=100
    )
    recommendation: str = Field(
        ...,
        description="Recommended action for loan origination"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "risk_probability": 0.2543,
                "risk_classification": "LOW",
                "credit_score": 75,
                "recommendation": "Recommend approval with standard terms"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_info: dict = Field(..., description="Model information")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    
    model_info: dict = Field(..., description="Model metadata")
    features: list = Field(..., description="List of model input features")
    outputs: list = Field(..., description="List of model outputs")
