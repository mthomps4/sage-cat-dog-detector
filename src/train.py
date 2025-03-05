import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

# 1. Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
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

# 2. Set up data preprocessing
def get_data_loaders(data_dir, batch_size=2):  # Smaller batch size for small dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = ImageFolder(data_dir, transform=transform)

    # Print dataset info
    print(f"Found {len(dataset)} images in {data_dir}")
    print(f"Classes: {dataset.classes}")

    # With small dataset, use 6 for training, 2 for testing
    train_size = 6
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# 3. Training function
def train_model(model, train_loader, test_loader, epochs=10):  # More epochs for small dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f'\nEpoch {epoch+1}/{epochs}:')

        # Training phase
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print(f'Batch {batch_idx+1}: Loss: {loss.item():.3f}')

        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        print(f'Training Loss: {epoch_loss:.3f}')
        print(f'Training Accuracy: {train_accuracy:.2f}%')

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        print(f'Test Loss: {test_loss/len(test_loader):.3f}')
        print(f'Test Accuracy: {test_accuracy:.2f}%')

def main():
    # Initialize model
    model = SimpleCNN()

    # Set data directory
    data_dir = os.path.join(os.getcwd(), 'data/training_data')

    print("Starting training process...")
    print(f"Loading data from: {data_dir}")

    # Get data loaders
    train_loader, test_loader = get_data_loaders(data_dir)

    # Train model
    train_model(model, train_loader, test_loader)

    # Save model
    model_save_path = 'cat_dog_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")

if __name__ == "__main__":
    main()