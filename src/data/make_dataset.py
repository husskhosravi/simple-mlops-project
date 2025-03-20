import os
import pandas as pd
import yaml

def main():
    """Main function to make dataset."""
    # Load config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Create directories if they don't exist
    os.makedirs(config['data']['processed_dir'], exist_ok=True)
    
    # For demo purposes, create a simple dataset
    data = {
        'feature1': [0.5, 0.7, 0.2, 0.1, 0.8],
        'feature2': [0.1, 0.2, 0.5, 0.8, 0.9],
        'feature3': [0.8, 0.7, 0.6, 0.2, 0.3],
        'target': [1, 1, 0, 0, 1]
    }
    df = pd.DataFrame(data)
    
    # Save processed data
    output_path = os.path.join(config['data']['processed_dir'], 'processed_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset created and saved to {output_path}")

if __name__ == "__main__":
    main()
