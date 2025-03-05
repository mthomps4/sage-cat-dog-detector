#!/bin/bash

# Check if an image path was provided
if [ $# -lt 1 ]; then
    echo "Usage: ./predict.sh <image_path> [model_path]"
    echo "Example: ./predict.sh path/to/image.jpg"
    exit 1
fi

# Get the image path
IMAGE_PATH="$1"

# Check if the image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Image file not found: $IMAGE_PATH"
    exit 1
fi

# Get the model path (optional)
MODEL_PATH="cat_dog_model.pth"
if [ $# -ge 2 ]; then
    MODEL_PATH="$2"
fi

# Check if the model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run prediction
echo "Predicting image: $IMAGE_PATH using model: $MODEL_PATH"
python src/predict.py "$IMAGE_PATH" --model "$MODEL_PATH"

# Deactivate virtual environment
deactivate