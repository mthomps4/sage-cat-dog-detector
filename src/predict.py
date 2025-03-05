import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from train import SimpleCNN  # Import the model architecture from train.py

def predict_image(image_path, model_path='cat_dog_model.pth'):
    """
    Predict whether an image contains a cat or dog

    Args:
        image_path: Path to the image file
        model_path: Path to the trained model file

    Returns:
        prediction: 'cat' or 'dog'
        confidence: Confidence score (0-1)
    """
    # Check if files exist
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Process image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Get class name
    class_names = ['cat', 'dog']  # Assuming class order from ImageFolder
    prediction = class_names[predicted.item()]

    return prediction, confidence.item()

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Predict if an image contains a cat or dog')
    parser.add_argument('image', help='Path to the image file')
    parser.add_argument('--model', default='cat_dog_model.pth', help='Path to model file')

    args = parser.parse_args()

    try:
        prediction, confidence = predict_image(args.image, args.model)
        print(f"Prediction: This is a {prediction}")
        print(f"Confidence: {confidence:.2f}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()