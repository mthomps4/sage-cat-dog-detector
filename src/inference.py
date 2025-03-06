import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import json

# Define the EXACT same model architecture used in training
class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.fc1(x)
        return x

def model_fn(model_dir):
    """Load the model from disk"""
    try:
        print("Loading model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model = CatDogCNN()
        print("Model architecture created")
        
        # Load the stored model parameters
        with open(f"{model_dir}/model.pth", 'rb') as f:
            print(f"Loading model state from: {model_dir}/model.pth")
            model.load_state_dict(torch.load(f, map_location=device))
            
        model.to(device)
        model.eval()
        print("Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def input_fn(request_body, request_content_type):
    """Convert the incoming image to a format the model can use"""
    try:
        print(f"Received request with content type: {request_content_type}")
        if request_content_type == 'application/x-image':
            image = Image.open(io.BytesIO(request_body))
            print(f"Loaded image of size: {image.size}")
            
            # Apply same transforms used during training
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            transformed_image = transform(image).unsqueeze(0)
            print("Image transformed successfully")
            return transformed_image
            
    except Exception as e:
        print(f"Error in input_fn: {str(e)}")
        raise
    
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make a prediction using the model"""
    try:
        print("Starting prediction...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_data = input_data.to(device)
        
        with torch.no_grad():
            output = model(input_data)
            prediction = torch.softmax(output, dim=1)
            print("Prediction completed successfully")
            return prediction
            
    except Exception as e:
        print(f"Error in predict_fn: {str(e)}")
        raise

def output_fn(prediction, accept):
    """Convert the prediction to a format to return to the client"""
    try:
        print("Formatting output...")
        probabilities = prediction.cpu().numpy().tolist()[0]
        result = {
            "dog_probability": probabilities[1],
            "cat_probability": probabilities[0]
        }
        print(f"Final prediction: {result}")
        return json.dumps(result)
        
    except Exception as e:
        print(f"Error in output_fn: {str(e)}")
        raise