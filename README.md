# Simple MLOps Project

This repo demonstrates core MLOps principles with a simple machine learning model pipeline.

## Project Structure

```
simple-mlops-project/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # CI workflow for testing
│       └── cd.yml                  # CD workflow for model deployment
├── data/
│   ├── raw/                        # Raw data directory
│   └── processed/                  # Processed data directory
├── models/                         # Directory to store trained models
├── notebooks/
│   └── exploration.ipynb           # Data exploration notebook
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── make_dataset.py         # Data processing scripts
│   │   └── validate_data.py        # Data validation
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py       # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py          # Model training
│   │   ├── predict_model.py        # Model prediction
│   │   └── evaluate_model.py       # Model evaluation
│   └── utils/
│       ├── __init__.py
│       └── logger.py               # Logging utilities
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Test configuration
│   ├── test_data.py                # Data tests
│   └── test_models.py              # Model tests
├── .gitignore                      # Git ignore file
├── Dockerfile                      # Docker configuration
├── requirements.txt                # Project dependencies
├── Makefile                        # Automation commands
├── config.yaml                     # Configuration parameters
└── README.md                       # Project documentation
```

## MLOps Components

This project showcases the following MLOps practices:

1. **Version Control**: Git for code and data versioning (using DVC)
2. **CI/CD Pipeline**: Automated testing and deployment using GitHub Actions
3. **Containerisation**: Docker for creating reproducible environments
4. **Model Versioning**: Tracking model versions with timestamps
5. **Testing**: Unit and integration tests for data and models
6. **Logging & Metrics**: Comprehensive logging and performance metrics
7. **Config Management**: Centralised configuration in YAML files
8. **Documentation**: Complete project documentation

## Getting Started

### Prerequisites
- Python 3.8+
- Docker
- Git
- DVC

### Installation

```bash
# Clone the repository
git clone https://github.com/husskhosravi/simple-mlops-project.git
cd simple-mlops-project

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

1. **Data Processing**:
   ```bash
   python -m src.data.make_dataset
   ```

2. **Model Training**:
   ```bash
   python -m src.models.train_model
   ```

3. **Model Evaluation**:
   ```bash
   python -m src.models.evaluate_model
   ```

4. **Running the Full Pipeline**:
   ```bash
   make all
   ```

## Project Details

### Machine Learning Pipeline

This project implements a complete machine learning pipeline:

1. **Data Processing**: Cleaning and preparing the data
2. **Feature Engineering**: Creating and selecting relevant features
3. **Model Training**: Training a Random Forest classifier
4. **Model Evaluation**: Evaluating performance with accuracy, precision, recall, and F1 score
5. **Model Deployment**: Serving the model via FastAPI (optional)

### Model Versioning

Each trained model is saved with a timestamp, allowing for easy tracking of model versions:
- `models/model_YYYYMMDD_HHMMSS.pkl`: Versioned models
- `models/model_latest.pkl`: The most recent model for easy access

### Containerisation

The project includes a Dockerfile for containerising the model and its dependencies, ensuring consistent behaviour across different environments.

### CI/CD Pipeline

GitHub Actions workflows are set up to:
- **CI**: Run tests on every push and pull request
- **CD**: Build and deploy the model on releases
