.PHONY: clean data lint requirements test train evaluate serve all

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = python3
PIP = pip

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements:
	$(PIP) install -r requirements.txt

## Make Dataset
data:
	$(PYTHON_INTERPRETER) -m src.data.make_dataset

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Run tests
test:
	pytest tests/

## Train model
train:
	$(PYTHON_INTERPRETER) -m src.models.train_model

## Evaluate model
evaluate:
	$(PYTHON_INTERPRETER) -m src.models.evaluate_model

## Run model as a service
serve:
	$(PYTHON_INTERPRETER) -m src.models.predict_model --serve

## Build docker image
docker-build:
	docker build -t mlops-model:latest .

## Run docker container
docker-run:
	docker run -p 8000:8000 mlops-model:latest

## Run the entire pipeline
all: requirements data lint test train evaluate

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Set up python interpreter environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv venv
	@echo ">>> New virtualenv created. Activate with:\nsource venv/bin/activate"

## Initialize DVC
init-dvc:
	dvc init
	dvc add data/raw
	git add data/.gitignore data/raw.dvc

## Register model to MLflow
register-model:
	$(PYTHON_INTERPRETER) -m src.models.register_model

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^## (.*)$$', line)
	if match:
		target = next(sys.stdin).split(':')[0].strip()
		print(f"make {target:20s} # {match.groups()[0]}")
endef

export PRINT_HELP_PYSCRIPT
