# Cat vs Dog Image Classifier

This project demonstrates a simple image classification model that distinguishes between cats and dogs, using PyTorch and AWS SageMaker.

## Project Structure

```
cat_dog_classifier/
├── requirements.txt      # Project dependencies
├── setup.sh             # Setup script for environment
├── src/
│   └── train.py         # Training script with CNN model
├── data/                # Dataset directory
│   ├── cats/           # Cat images
│   └── dogs/           # Dog images
├── .gitignore           # Git ignore file
└── README.md           # This file
```

## Local Development

### Setup

1. Clone repository and setup environment:

```bash
git clone <repository-url>
cd cat_dog_classifier
chmod +x setup.sh
./setup.sh
```

### Local Training

```bash
# Activate virtual environment
source venv/bin/activate

# Train model
python src/train.py
```

### Local Inference

```bash
./predict.sh path/to/image.jpg
```

## AWS SageMaker Development

### Prerequisites

1. AWS Account with SageMaker access
2. S3 bucket for training data
3. Appropriate IAM roles and permissions

### Setup Steps Completed

1. ✅ Created SageMaker Domain
   - Name: "Cat Dog Image Predictor"
   - Project profile configured for ML development

2. ✅ Data Preparation
   - Training data uploaded to S3
   - Location: `s3://yolo-sagemaker/training_data/`

3. ✅ Training Configuration
   - Created training notebook
   - Using PyTorch estimator
   - GPU instance (ml.g4dn.xlarge)

4. ✅ Model Training
   - Successfully trained on SageMaker
   - Model artifacts saved to S3
   - Training job: `pytorch-training-2025-03-06-16-44-33-803`

### Next Steps

1. [ ] Create SageMaker endpoint
2. [ ] Test endpoint with new images
3. [ ] Set up monitoring and logging

## Model Details

- Architecture: Simple CNN
- Framework: PyTorch 2.0
- Training Environment: SageMaker GPU instance (ml.g4dn.xlarge)
- Data Split: Training (80%), Testing (20%)

## Model Deployment and Inference

### Deployment Process

The model is deployed using two key components:
1. `deploy.ipynb` - Notebook for model deployment
2. `inference.py` - Script that handles prediction requests

#### Deploy Notebook Structure
```python
# Create and deploy the model endpoint
model = PyTorchModel(
    model_data=s3_model_path,
    role=role,
    framework_version='2.0',
    py_version='py310',
    entry_point='inference.py'
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    endpoint_name='cat-dog-classifier-v1'
)
```

### Inference Script Components

The `inference.py` script contains four essential functions that handle the prediction pipeline:

1. **model_fn(model_dir)**
   - Loads the trained model from disk
   - Initializes model architecture
   - Puts model in evaluation mode
   ```python
   def model_fn(model_dir):
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       model = CatDogCNN()
       model.load_state_dict(torch.load(f"{model_dir}/model.pth"))
       model.eval()
       return model
   ```

2. **input_fn(request_body, request_content_type)**
   - Processes incoming image data
   - Applies necessary transformations
   - Converts image to tensor
   ```python
   def input_fn(request_body, request_content_type):
       image = Image.open(io.BytesIO(request_body))
       transform = transforms.Compose([
           transforms.Resize((224, 224)),
           transforms.ToTensor(),
           transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
       ])
       return transform(image).unsqueeze(0)
   ```

3. **predict_fn(input_data, model)**
   - Runs the actual prediction
   - Returns probability scores
   ```python
   def predict_fn(input_data, model):
       with torch.no_grad():
           output = model(input_data)
           return torch.softmax(output, dim=1)
   ```

4. **output_fn(prediction, accept)**
   - Formats prediction results
   - Returns JSON response
   ```python
   def output_fn(prediction, accept):
       probabilities = prediction.numpy().tolist()[0]
       return json.dumps({
           "dog_probability": probabilities[1],
           "cat_probability": probabilities[0]
       })
   ```

### Making Predictions

When deployed, the endpoint accepts HTTP requests with image data and returns predictions in JSON format:

```json
{
    "cat_probability": 0.1,
    "dog_probability": 0.9
}
```

### Important Notes

1. The endpoint remains active and incurs costs until explicitly deleted
2. Input preprocessing must match training preprocessing exactly
3. GPU acceleration is available depending on the instance type
4. Monitor endpoint metrics in CloudWatch

## Resources

- Training Data: `s3://yolo-sagemaker/training_data/`
- Model Artifacts: `s3://yolo-sagemaker/pytorch-training-2025-03-06-16-44-33-803/output/`
- SageMaker Domain URL: [Your domain URL]

## Troubleshooting

Common issues and solutions:

1. S3 Permissions: Ensure proper S3 access policies and permissions boundaries
2. GPU vs CPU: Training uses GPU instance, local inference uses available hardware
3. Path Issues: Check relative paths when running scripts from different locations
