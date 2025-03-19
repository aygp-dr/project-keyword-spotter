.PHONY: help clean test lint format run detect test-inference analyze status docker-build docker-run quickstart

# Project variables
PROJECT_NAME = project-keyword-spotter
PYTHON = python3
POETRY = poetry
VENV_PATH = $(shell $(POETRY) env info --path 2>/dev/null)
PYTHON_VERSION = 3.9
SHELL := /bin/bash

# Directory and file paths
MODELS_DIR = models
CONFIG_DIR = config/log_analyzer
DATA_DIR = data/log_analyzer
VENV_MARKER = .venv/.initialized
POETRY_INSTALLED = .poetry_installed
EDGETPU_INSTALLED = .edgetpu_installed
VECTORIZER_FILE = $(MODELS_DIR)/log_analyzer/log_vectorizer.pkl
CONFIG_FILE = $(CONFIG_DIR)/config.json
LOG_ANALYZER_INIT = $(DATA_DIR)/.initialized
DOWNLOADS_MARKER = $(MODELS_DIR)/.downloaded

# Default target
help:  ## Show this help message
	@echo "EdgeTPU Machine Learning Project Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# Setup commands
$(POETRY_INSTALLED):  ## Install Poetry
	@echo "Installing Poetry..."
	@if ! command -v $(POETRY) &> /dev/null; then \
		curl -sSL https://install.python-poetry.org | $(PYTHON) -; \
	fi
	@touch $(POETRY_INSTALLED)

$(VENV_MARKER): $(POETRY_INSTALLED) pyproject.toml  ## Create virtual environment
	@echo "Setting up virtual environment..."
	@$(POETRY) env use $(PYTHON_VERSION) || $(POETRY) env use python$(PYTHON_VERSION)
	@$(POETRY) install
	@$(POETRY) run pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl
	@mkdir -p .venv
	@touch $(VENV_MARKER)
	@echo "Virtual environment set up!"

setup: $(VENV_MARKER)  ## Set up the project (install poetry, create venv)

$(EDGETPU_INSTALLED):  ## Install Edge TPU system dependencies
	@echo "Installing Edge TPU system dependencies..."
	@echo 'deb https://packages.cloud.google.com/apt coral-edgetpu-stable main' | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
	@curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
	@sudo apt-get update
	@sudo apt-get install libedgetpu1-std
	@touch $(EDGETPU_INSTALLED)
	@echo "Edge TPU system dependencies installed."

install-edgetpu: $(EDGETPU_INSTALLED)

# Create directory structure
$(CONFIG_DIR) $(MODELS_DIR) $(DATA_DIR):
	@mkdir -p $@

# Download models
$(DOWNLOADS_MARKER): | $(MODELS_DIR)  ## Download test models for Edge TPU
	@echo "Downloading test models for Edge TPU..."
	@wget -nc -P $(MODELS_DIR)/ https://github.com/google-coral/test_data/raw/master/mobilenet_v2_1.0_224_quant_edgetpu.tflite
	@wget -nc -P $(MODELS_DIR)/ https://github.com/google-coral/test_data/raw/master/imagenet_labels.txt
	@wget -nc -P $(MODELS_DIR)/ https://github.com/google-coral/test_data/raw/master/parrot.jpg
	@touch $(DOWNLOADS_MARKER)
	@echo "Test models downloaded."

download-models: $(DOWNLOADS_MARKER)

# Dev tools
install-dev-tools: $(VENV_MARKER)  ## Install development tools (pytest, flake8, mypy, black)
	@echo "Installing development tools..."
	@$(POETRY) add --dev pytest flake8 mypy black pytest-cov

clean:  ## Remove Python artifacts and generated files
	@echo "Cleaning project..."
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.pyd" -delete
	@find . -type f -name ".coverage" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@find . -type d -name "*.egg" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name ".coverage" -exec rm -rf {} +
	@find . -type d -name "htmlcov" -exec rm -rf {} +
	@find . -type d -name ".tox" -exec rm -rf {} +
	@find . -type f -name "*.so" -delete
	@find . -type f -name "*.c" -delete
	@rm -rf build/
	@rm -rf dist/
	@rm -rf .eggs/

# Testing and code quality
test: $(VENV_MARKER)  ## Run tests with pytest
	@echo "Running tests..."
	@$(POETRY) run pytest

lint: $(VENV_MARKER)  ## Run linting checks
	@echo "Running linters..."
	@$(POETRY) run flake8 project_keyword_spotter
	@$(POETRY) run mypy project_keyword_spotter

format: $(VENV_MARKER)  ## Format code with black
	@echo "Formatting code..."
	@$(POETRY) run black project_keyword_spotter

# Run commands
run: $(VENV_MARKER)  ## Run the main application
	@echo "Running the main application..."
	@$(POETRY) run python -m project_keyword_spotter.run_model

# Coral Edge TPU detection
detect: $(VENV_MARKER) $(EDGETPU_INSTALLED)  ## Detect Coral USB Accelerator
	@echo "Detecting Coral USB Accelerator..."
	@$(POETRY) run python project_keyword_spotter/detect_coral.py

test-inference: $(VENV_MARKER) $(EDGETPU_INSTALLED) $(DOWNLOADS_MARKER)  ## Test inference on Coral USB Accelerator
	@echo "Testing inference on Coral USB Accelerator..."
	@$(POETRY) run python project_keyword_spotter/test_coral_inference.py

# Log analyzer commands
$(CONFIG_FILE): | $(CONFIG_DIR)
	@echo "Creating default configuration..."
	@echo '{"log_paths": ["/var/log/syslog", "/var/log/auth.log"], "anomaly_threshold": 0.8, "max_logs_per_check": 1000}' > $(CONFIG_FILE)

$(VECTORIZER_FILE): | $(MODELS_DIR)
	@echo "Creating initial vectorizer..."
	@mkdir -p $(MODELS_DIR)/log_analyzer
	@$(POETRY) run python -c "import pickle; pickle.dump({'feature_map': {}, 'is_fitted': False}, open('$(VECTORIZER_FILE)', 'wb'))"

$(LOG_ANALYZER_INIT): $(VENV_MARKER) $(CONFIG_FILE) $(VECTORIZER_FILE) | $(DATA_DIR)
	@echo "Setting up log analyzer directories..."
	@mkdir -p $(DATA_DIR)/anomalies
	@touch $(LOG_ANALYZER_INIT)

init: $(LOG_ANALYZER_INIT)  ## Initialize the log analyzer

analyze: $(LOG_ANALYZER_INIT)  ## Analyze logs for anomalies
	@echo "Analyzing logs for anomalies..."
	@$(POETRY) run python project_keyword_spotter/log_analyzer.py analyze --verbose

# Check if we have training data files
TRAINING_FILES := $(wildcard $(DATA_DIR)/training_logs_*.txt)

train: $(LOG_ANALYZER_INIT) $(VECTORIZER_FILE)  ## Train the vectorizer using collected logs
	@echo "Training the vectorizer using collected logs..."
	@if [ -z "$(TRAINING_FILES)" ]; then \
		echo "No training files found! Running analyzer to collect some logs first..."; \
		$(MAKE) analyze; \
		echo "Waiting for log collection (5 seconds)..."; \
		sleep 5; \
	fi
	@$(POETRY) run python project_keyword_spotter/log_analyzer.py train

status: $(LOG_ANALYZER_INIT)  ## Show status of log analyzer components
	@echo "Checking status of log analyzer components..."
	@$(POETRY) run python project_keyword_spotter/log_analyzer.py status

# Create a shell script to activate the virtual environment
activate: $(VENV_MARKER)  ## Create an activation script for the virtual environment
	@echo "Creating activation script..."
	@echo '#!/bin/bash' > activate
	@echo 'source $(VENV_PATH)/bin/activate' >> activate
	@chmod +x activate
	@echo "Created 'activate' script. Run 'source ./activate' to activate the environment."

# Voice command runner
run-voice: $(VENV_MARKER)  ## Run voice command recognition
	@echo "Running voice command recognition..."
	@$(POETRY) run ./run_yt_voice_control.sh

# Snake game runner
run-snake: $(VENV_MARKER)  ## Run hearing snake game
	@echo "Running hearing snake game..."
	@$(POETRY) run ./run_snake.sh

# Create a Docker environment if needed
docker-build:  ## Build Docker image for development
	@echo "Building Docker image for development..."
	@docker build -t $(PROJECT_NAME) .

docker-run:  ## Run Docker container with USB passthrough
	@echo "Running Docker container..."
	@docker run --rm -it --privileged -v /dev/bus/usb:/dev/bus/usb -v $(PWD):/app $(PROJECT_NAME)

quickstart: setup install-edgetpu download-models detect  ## Quickly set up and verify Edge TPU environment
	@echo "Edge TPU environment is set up and ready!"

log-analyzer-flow: init analyze train status  ## Run complete log analyzer workflow
	@echo "Log analyzer workflow completed!"

