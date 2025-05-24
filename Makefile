# Makefile for Project Keyword Spotter

.PHONY: help install test snake youtube clean

help:
	@echo "Available commands:"
	@echo "  make install   - Install system packages and Python dependencies"
	@echo "  make test      - Test keyword detection with console output"
	@echo "  make snake     - Run voice-controlled snake game"
	@echo "  make youtube   - Run YouTube voice control"
	@echo "  make clean     - Clean generated files"

install:
	sh install_requirements.sh

test:
	python3 run_model.py

snake:
	bash run_snake.sh

youtube:
	bash run_yt_voice_control.sh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete