import os
import pickle
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report

def main():
    """Evaluate the trained model."""
    # Load config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Load test data
    data_path = os.path.join(config['data']['processed_dir'], 'processed_data.csv')
    data = pd.read_csv(data_path)
    X = data.drop(columns=[config['data']['target_column']])
    y = data[config['data']['target_column']]
    
    # Load model
    model_path = os.path.join(config['model']['model_dir'], 'model_latest.pkl')
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"Model file not found at {model_path}. Run train_model.py first.")
        return
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Evaluate
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    
    # Print results
    print(f"Model Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    main()
