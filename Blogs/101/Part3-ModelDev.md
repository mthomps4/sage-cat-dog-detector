## Part 3: Model Development

### Local Development

#### PyTorch Model
```python
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

#### Testing Framework
```python
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total
```

### SageMaker Training

#### Training Script
```python
def model_fn(model_dir):
    """Load the model for inference."""
    model = CustomModel()
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model

def train(args):
    """Training function for SageMaker."""
    model = CustomModel()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(args.epochs):
        train_epoch(model, train_loader, optimizer)

    save_model(model, args.model_dir)
```