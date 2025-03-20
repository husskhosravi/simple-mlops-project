#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module handles model training and registration.
"""

import os
import pickle
import logging
import argparse
import yaml
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config file."""
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_data(config):
    """Load processed data for model training."""
    try:
        data_path = os.path.join(config['data']['processed_dir'], 'processed_data.csv')
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        X = data.drop(columns=[config['data']['target_column']])
        y = data[config['data']['target_column']]
        return X, y
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def train_model(X, y, config):
    """Train the model with the given parameters."""
    logger.info("Starting model training")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['model']['test_size'], 
        random_state=config['model']['random_state']
    )
    
    # Start MLflow run
    mlflow.start_run()
    
    # Log parameters
    mlflow.log_params({
        "model_type": "RandomForestClassifier",
        "n_estimators": config['model']['n_estimators'],
        "max_depth": config['model']['max_depth'],
        "random_state": config['model']['random_state']
    })
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=config['model']['n_estimators'],
        max_depth=config['model']['max_depth'],
        random_state=config['model']['random_state']
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Log metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    mlflow.log_metrics(metrics)
    
    # Log feature importance
    feature_importance = pd.DataFrame(
        model.feature_importances_,
        index=X.columns,
        columns=['importance']
    ).sort_values('importance', ascending=False)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Save model locally
    model_dir = config['model']['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save latest version for quick access
    latest_model_path = os.path.join(model_dir, "model_latest.pkl")
    with open(latest_model_path, 'wb') as f:
        pickle.dump(model, f)
    
    mlflow.end_run()
    
    logger.info(f"Model training completed. Model saved to {model_path}")
    logger.info(f"Model performance: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    
    return model, metrics, model_path

def main():
    """Main function for model training."""
    parser = argparse.ArgumentParser(description='Train machine learning model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Setup MLflow
    config = load_config()
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    # Load data
    X, y = load_data(config)
    
    # Train model
    model, metrics, model_path = train_model(X, y, config)
    
    logger.info("Model training pipeline completed successfully")
    return model_path

if __name__ == "__main__":
    main()
