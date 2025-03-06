import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import json

# Define the same model architecture used in training
class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Define your model architecture exactly as in train.py
        # ... (we'll need to match your training architecture)

def model_fn(model_dir):
    """Load the model from disk"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CatDogCNN()

    # Load the stored model parameters
    model.load_state_dict(torch.load(f"{model_dir}/model.pth", map_location=device))
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    """Convert the incoming image to a format the model can use"""
    if request_content_type == 'application/x-image':
        image = Image.open(io.BytesIO(request_body))
        # Apply same transforms used during training
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make a prediction using the model"""
    with torch.no_grad():
        output = model(input_data)
        prediction = torch.softmax(output, dim=1)
        return prediction

def output_fn(prediction, accept):
    """Convert the prediction to a format to return to the client"""
    probabilities = prediction.numpy().tolist()[0]
    return json.dumps({
        "dog_probability": probabilities[1],
        "cat_probability": probabilities[0]
    })