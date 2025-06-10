"""
Real-time Fraud Detection API
FastAPI endpoint for real-time transaction fraud detection.

Author: Ram Bharat Chowdary Moturi
Date: 2024
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, validator
from typing import Optional, Dict, List
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fraud_detection_model import FraudDetectionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Fraud Detection API",
    description="Real-time fraud detection system for financial transactions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
fraud_model = None

# Pydantic models for request/response
class TransactionRequest(BaseModel):
    """Transaction data for fraud detection."""
    customer_id: int
    amount: float
    merchant_category: str
    payment_method: str
    merchant_location: str
    hour: Optional[int] = None
    day_of_week: Optional[int] = None
    is_weekend: Optional[int] = None
    is_international: Optional[int] = None
    
    @validator('amount')
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v
    
    @validator('hour')
    def hour_must_be_valid(cls, v):
        if v is not None and (v < 0 or v > 23):
            raise ValueError('Hour must be between 0 and 23')
        return v

class FraudDetectionResponse(BaseModel):
    """Fraud detection result."""
    transaction_id: str
    customer_id: int
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    fraud_score: int
    alert_message: str
    model_used: str
    timestamp: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    model_loaded: bool
    version: str

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load fraud detection model on startup."""
    global fraud_model
    try:
        # Try to load pre-trained model
        model_path = "models/"
        if os.path.exists(f"{model_path}fraud_xgboost_model.pkl"):
            logger.info("Loading pre-trained fraud detection model...")
            fraud_model = FraudDetectionModel()
            
            # Load saved components
            fraud_model.models['xgboost'] = joblib.load(f"{model_path}fraud_xgboost_model.pkl")
            fraud_model.scalers['standard'] = joblib.load(f"{model_path}fraud_standard_scaler.pkl")
            fraud_model.feature_names = joblib.load(f"{model_path}fraud_feature_names.pkl")
            
            logger.info("✅ Fraud detection model loaded successfully")
        else:
            logger.warning("⚠️ No pre-trained model found. Please train models first.")
            fraud_model = None
            
    except Exception as e:
        logger.error(f"❌ Error loading fraud detection model: {e}")
        fraud_model = None

# Helper functions
def encode_categorical_features(transaction_data: Dict) -> Dict:
    """Encode categorical features (simplified for demo)."""
    
    # Merchant category encoding
    merchant_categories = {
        'grocery': 0, 'gas_station': 1, 'restaurant': 2, 'retail': 3,
        'online': 4, 'atm': 5, 'pharmacy': 6, 'entertainment': 7,
        'travel': 8, 'other': 9
    }
    
    # Payment method encoding
    payment_methods = {
        'credit_card': 0, 'debit_card': 1, 'online': 2
    }
    
    # Merchant location encoding
    merchant_locations = {
        'same_city': 0, 'same_state': 1, 'different_state': 2, 'international': 3
    }
    
    # Apply encodings
    transaction_data['merchant_category_encoded'] = merchant_categories.get(
        transaction_data.get('merchant_category', 'other'), 9
    )
    transaction_data['payment_method_encoded'] = payment_methods.get(
        transaction_data.get('payment_method', 'credit_card'), 0
    )
    transaction_data['merchant_location_encoded'] = merchant_locations.get(
        transaction_data.get('merchant_location', 'same_city'), 0
    )
    
    return transaction_data

def engineer_transaction_features(transaction_data: Dict) -> Dict:
    """Engineer features for a single transaction."""
    
    # Current time if not provided
    if 'hour' not in transaction_data or transaction_data['hour'] is None:
        transaction_data['hour'] = datetime.now().hour
    
    if 'day_of_week' not in transaction_data or transaction_data['day_of_week'] is None:
        transaction_data['day_of_week'] = datetime.now().weekday()
    
    # Basic feature engineering
    transaction_data['amount_log'] = np.log1p(transaction_data['amount'])
    transaction_data['is_weekend'] = 1 if transaction_data['day_of_week'] in [5, 6] else 0
    transaction_data['is_night'] = 1 if transaction_data['hour'] in [22, 23, 0, 1, 2, 3, 4, 5] else 0
    
    # Simplified features (in production, these would come from customer history)
    transaction_data['time_diff'] = 2.0  # Hours since last transaction
    transaction_data['transactions_last_1h'] = 1
    transaction_data['transactions_last_24h'] = 5
    transaction_data['amount_zscore'] = 0.5  # Standardized amount
    transaction_data['merchant_risk_score'] = 0.1  # Merchant risk score
    
    # International flag
    if 'is_international' not in transaction_data:
        transaction_data['is_international'] = 1 if transaction_data.get('merchant_location') == 'international' else 0
    
    transaction_data['is_different_state'] = 1 if transaction_data.get('merchant_location') == 'different_state' else 0
    
    return transaction_data

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """API documentation landing page."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            .endpoint { background: #ecf0f1; padding: 15px; margin: 15px 0; border-radius: 5px; }
            .method { display: inline-block; padding: 5px 10px; border-radius: 3px; color: white; font-weight: bold; }
            .post { background-color: #e74c3c; }
            .get { background-color: #27ae60; }
            code { background: #34495e; color: white; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="header">🔍 Financial Fraud Detection API</h1>
            <p>Real-time fraud detection system for financial transactions using machine learning.</p>
            
            <h2>📋 Available Endpoints</h2>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/detect-fraud</strong> - Analyze transaction for fraud
                <p>Submit transaction data and get real-time fraud risk assessment.</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/health</strong> - Check API health status
                <p>Verify API and model availability.</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/docs</strong> - Interactive API documentation
                <p>Full Swagger/OpenAPI documentation with request/response examples.</p>
            </div>
            
            <h2>🚀 Quick Start</h2>
            <p>Example fraud detection request:</p>
            <pre><code>curl -X POST "http://localhost:8000/detect-fraud" \\
     -H "Content-Type: application/json" \\
     -d '{
       "customer_id": 1234,
       "amount": 5000.00,
       "merchant_category": "online",
       "payment_method": "credit_card",
       "merchant_location": "international",
       "hour": 2
     }'</code></pre>
            
            <h2>📊 Features</h2>
            <ul>
                <li>Real-time fraud scoring using XGBoost</li>
                <li>Risk level classification (Low/Medium/High)</li>
                <li>Automated alert generation</li>
                <li>RESTful API with JSON responses</li>
                <li>Comprehensive logging and monitoring</li>
            </ul>
            
            <p><strong>Author:</strong> Ram Bharat Chowdary Moturi | <strong>Version:</strong> 1.0.0</p>
        </div>
    </body>
    </html>
    """
    return html_content

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if fraud_model is not None else "degraded",
        timestamp=datetime.now().isoformat(),
        model_loaded=fraud_model is not None,
        version="1.0.0"
    )

@app.post("/detect-fraud", response_model=FraudDetectionResponse)
async def detect_fraud(transaction: TransactionRequest):
    """Detect fraud in a transaction."""
    
    start_time = datetime.now()
    
    if fraud_model is None:
        raise HTTPException(status_code=503, detail="Fraud detection model not available")
    
    try:
        # Convert request to dictionary
        transaction_data = transaction.dict()
        
        # Generate unique transaction ID
        transaction_id = f"TXN_{int(datetime.now().timestamp() * 1000)}"
        
        # Encode categorical features
        transaction_data = encode_categorical_features(transaction_data)
        
        # Engineer features
        transaction_data = engineer_transaction_features(transaction_data)
        
        # Perform fraud detection
        fraud_result = fraud_model.detect_fraud_realtime(transaction_data, model_name='xgboost')
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Log the transaction
        logger.info(f"Fraud detection completed for transaction {transaction_id}: "
                   f"Risk={fraud_result['risk_level']}, Score={fraud_result['fraud_score']}, "
                   f"Time={processing_time:.2f}ms")
        
        return FraudDetectionResponse(
            transaction_id=transaction_id,
            customer_id=transaction.customer_id,
            is_fraud=fraud_result['is_fraud'],
            fraud_probability=fraud_result['fraud_probability'],
            risk_level=fraud_result['risk_level'],
            fraud_score=fraud_result['fraud_score'],
            alert_message=fraud_result['alert_message'],
            model_used=fraud_result['model_used'],
            timestamp=fraud_result['timestamp'],
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing fraud detection request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    if fraud_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "XGBoost Classifier",
        "features_count": len(fraud_model.feature_names) if fraud_model.feature_names else 0,
        "feature_names": fraud_model.feature_names,
        "fraud_threshold": fraud_model.fraud_threshold,
        "model_status": "loaded"
    }

@app.get("/stats")
async def get_api_stats():
    """Get basic API statistics."""
    # In a production environment, these would come from a database or cache
    return {
        "api_version": "1.0.0",
        "uptime_seconds": "N/A",
        "total_requests": "N/A",
        "fraud_detected": "N/A",
        "avg_response_time_ms": "N/A",
        "model_accuracy": "85.2%",
        "last_model_update": "2024-01-01"
    }

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Fraud Detection API Server...")
    print("📊 Interactive docs available at: http://localhost:8000/docs")
    print("🔍 API endpoint: http://localhost:8000/detect-fraud")
    print("❤️ Health check: http://localhost:8000/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )