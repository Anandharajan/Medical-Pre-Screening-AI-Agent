#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs

# Set environment variables
export PYTHONUNBUFFERED=TRUE
export PYTHONIOENCODING=UTF-8