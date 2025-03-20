# Simple MLOps Project

This repository demonstrates core MLOps principles with a simple machine learning model pipeline.

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
├── setup.py                        # Package setup
├── Makefile                        # Automation commands
├── config.yaml                     # Configuration parameters
├── README.md                       # Project documentation
└── mlflow.yaml                     # MLflow configuration
```

## MLOps Components

This project demonstrates the following MLOps practices:

1. **Version Control**: Git for code, data (using DVC), and model versioning
2. **CI/CD Pipeline**: Automated testing and deployment using GitHub Actions
3. **Containerization**: Docker for creating reproducible environments
4. **Experiment Tracking**: MLflow for tracking experiments and model versions
5. **Model Registry**: Storing and versioning models
6. **Testing**: Unit and integration tests for data and models
7. **Monitoring**: Basic monitoring setup for model performance
8. **Documentation**: Comprehensive documentation of the project

## Getting Started

### Prerequisites
- Python 3.8+
- Docker
- Git
- DVC

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/simple-mlops-project.git
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

## License
This project is licensed under the MIT License - see the LICENSE file for details.# simple-mlops-project
MLOps practices
