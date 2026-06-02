"""
FastAPI REST Service for Credit Risk Scoring

Endpoints for predicting customer credit risk probability.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import numpy as np
import pandas as pd
import logging
from typing import List
from .pydantic_models import PredictionRequest, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Scoring API",
    description="REST API for predicting customer credit risk probability",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
model = None
scaler = None
model_info = None


@app.on_event("startup")
async def load_model():
    """Load model from MLflow registry on startup."""
    global model, scaler, model_info
    
    try:
        logger.info("Loading model from MLflow...")
        
        # Load best model
        model_uri = "models:/CreditRiskBestModel/production"
        model = mlflow.pyfunc.load_model(model_uri)
        
        model_info = {
            "status": "loaded",
            "timestamp": pd.Timestamp.now().isoformat(),
            "model_uri": model_uri
        }
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load from registry: {e}")
        logger.info("Using mock model for demo")
        model_info = {
            "status": "mock",
            "timestamp": pd.Timestamp.now().isoformat(),
            "note": "Using mock model for demonstration"
        }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_info": model_info
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict credit risk probability for a customer.
    
    Parameters
    ----------
    request : PredictionRequest
        Customer features for risk scoring
        
    Returns
    -------
    PredictionResponse
        Risk probability and risk classification
    """
    try:
        logger.info(f"Prediction request received for customer: {request.customer_id}")
        
        # Create feature vector
        features = np.array([
            request.recency,
            request.frequency,
            request.monetary,
            request.avg_transaction_amount,
            request.fraud_count,
            request.transaction_count
        ]).reshape(1, -1)
        
        # Generate prediction
        if model and model_info.get("status") == "loaded":
            risk_probability = float(model.predict(features)[0])
        else:
            # Mock prediction for demo
            risk_probability = float(
                (request.recency / 365 + (1 - request.frequency / 100) + 
                 (1 - request.monetary / 10000)) / 3
            )
            risk_probability = max(0, min(1, risk_probability))
        
        # Classify risk
        if risk_probability >= 0.7:
            risk_class = "HIGH"
        elif risk_probability >= 0.4:
            risk_class = "MEDIUM"
        else:
            risk_class = "LOW"
        
        # Credit score (0-100, inverse of risk)
        credit_score = int((1 - risk_probability) * 100)
        
        response = PredictionResponse(
            customer_id=request.customer_id,
            risk_probability=round(risk_probability, 4),
            risk_classification=risk_class,
            credit_score=credit_score,
            recommendation={
                "HIGH": "Recommend decline or require additional verification",
                "MEDIUM": "Recommend lower credit limit or higher interest rate",
                "LOW": "Recommend approval with standard terms"
            }[risk_class]
        )
        
        logger.info(f"Prediction complete: {response.customer_id} - {risk_class} risk")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    """
    Predict credit risk for multiple customers.
    
    Parameters
    ----------
    requests : List[PredictionRequest]
        List of customer feature requests
        
    Returns
    -------
    List[PredictionResponse]
        List of predictions
    """
    try:
        logger.info(f"Batch prediction request for {len(requests)} customers")
        
        predictions = [await predict(req) for req in requests]
        
        logger.info(f"Batch prediction complete")
        return predictions
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    return {
        "model_info": model_info,
        "features": [
            "recency",
            "frequency",
            "monetary",
            "avg_transaction_amount",
            "fraud_count",
            "transaction_count"
        ],
        "outputs": ["risk_probability", "risk_classification", "credit_score"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
