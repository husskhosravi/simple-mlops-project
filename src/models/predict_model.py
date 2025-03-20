#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module handles model prediction and serving as an API.
"""

import os
import pickle
import logging
import argparse
import yaml
import json
from datetime import datetime

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictionInput(BaseModel):
    """Model input schema."""
    features: List[Dict[str, Any]] = Field(..., example=[{"feature1": 0.5, "feature2": 0.7, "feature3": 0.2}])

class PredictionOutput(BaseModel):
    """Model output schema."""
    predictions: List[int] = Field(..., example=[1, 0, 1])
    probabilities: Optional[List[Dict[str, float]]] = Field(None, example=[{"0": 0.2, "1": 0.8}])
    model_version: str = Field(..., example="v1.0.0")
    timestamp: str = Field(..., example="2023-06-01T12:00:00")

def load_config():
    """Load configuration from config file."""
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_model(config):
    """Load the trained model."""
    try:
        model_path = os.getenv('MODEL_PATH', os.path.join(config['model']['model_dir'], 'model_latest.pkl'))
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def predict(model, features, config):
    """Make predictions using the trained model."""
    try:
        # Convert input to DataFrame if it's not already
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)
        
        # Ensure features are in the correct order
        expected_features = config['data']['features']
        if set(features.columns) != set(expected_features):
            logger.warning(f"Input features do not match expected features. Expected: {expected_features}, Got: {list(features.columns)}")
            # Reorder or fill missing columns
            for col in expected_features:
                if col not in features.columns:
                    features[col] = 0  # Default value for missing features
            features = features[expected_features]
        
        # Make prediction
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        # Format probabilities
        prob_dict = []
        for prob in probabilities:
            prob_dict.append({str(i): float(p) for i, p in enumerate(prob)})
        
        return predictions.tolist(), prob_dict
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise

def create_app(model, config):
    """Create FastAPI app for serving predictions."""
    app = FastAPI(
        title="ML Model API",
        description="API for making predictions with a trained machine learning model",
        version="1.0.0"
    )
    
    @app.get("/")
    def root():
        return {"message": "Welcome to the ML Model API. Use /predict for predictions."}
    
    @app.get("/health")
    def health():
        return {"status": "healthy"}
    
    @app.post("/predict", response_model=PredictionOutput)
    def predict_endpoint(input_data: PredictionInput):
        try:
            # Convert input to DataFrame
            features_df = pd.DataFrame(input_data.features)
            
            # Make prediction
            predictions, probabilities = predict(model, features_df, config)
            
            # Return response
            return {
                "predictions": predictions,
                "probabilities": probabilities,
                "model_version": os.getenv('MODEL_VERSION', 'latest'),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in prediction endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app

def main():
    """Main function for model prediction and serving."""
    parser = argparse.ArgumentParser(description='Serve machine learning model')
    parser.add_argument('--serve', action='store_true', help='Run as API server')
    parser.add_argument('--port', type=int, default=8000, help='Port for API server')
    parser.add_argument('--input', type=str, help='Path to input data for batch prediction')
    parser.add_argument('--output', type=str, help='Path to save batch prediction results')
    args = parser.parse_args()
    
    # Load config and model
    config = load_config()
    model = load_model(config)
    
    if args.serve:
        # Run as API server
        logger.info(f"Starting API server on port {args.port}")
        app = create_app(model, config)
        uvicorn.run(app, host=config['api']['host'], port=args.port)
    elif args.input:
        # Run batch prediction
        logger.info(f"Running batch prediction on {args.input}")
        input_data = pd.read_csv(args.input)
        predictions, probabilities = predict(model, input_data, config)
        
        # Save results
        output_path = args.output or f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results = pd.DataFrame({
            'prediction': predictions,
            'probability': [json.dumps(p) for p in probabilities]
        })
        if 'id' in input_data.columns:
            results['id'] = input_data['id']
        
        results.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
    else:
        logger.error("Either --serve or --input must be specified")
        parser.print_help()

if __name__ == "__main__":
    main()
