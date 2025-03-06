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

1. ✅ Create SageMaker endpoint
2. ✅ Test endpoint with new images
3. ✅ Set up monitoring and logging

## Model Details

- Architecture: Simple CNN
- Framework: PyTorch 2.0
- Training Environment: SageMaker GPU instance (ml.g4dn.xlarge)
- Inference Environment: SageMaker CPU instance (ml.t2.large)
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
    entry_point='inference.py',
    source_dir='src'  # Important: points to directory containing inference.py
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.large',  # Important: t2.large needed for stable inference
    endpoint_name='cat-dog-classifier-v1'
)
```

### Instance Type Selection

We found that instance type selection is crucial for stable inference:

- ml.t2.medium: Insufficient for this model (timeout issues)
- ml.t2.large: Minimum recommended for stable inference
- ml.m5.xlarge: Recommended for production use
- ml.p3.2xlarge: When GPU acceleration is needed

### Testing the Endpoint

Created a separate notebook `test_endpoint.ipynb` for testing:

```python
def test_prediction(image_path):
    runtime = boto3.client('sagemaker-runtime')

    with open(image_path, 'rb') as f:
        image_data = f.read()

    response = runtime.invoke_endpoint(
        EndpointName='cat-dog-classifier-v1',
        ContentType='application/x-image',
        Body=image_data
    )

    result = json.loads(response['Body'].read().decode())
    return result
```

Example response:

```json
{
    "cat_probability": 0.92,
    "dog_probability": 0.08
}
```

### Endpoint Management

Important considerations:

1. Endpoints incur costs while running
2. Monitor CloudWatch logs for issues
3. Delete endpoints when not in use
4. Consider auto-scaling for production

To delete an endpoint:

```python
import boto3
sagemaker_client = boto3.client('sagemaker')
sagemaker_client.delete_endpoint(EndpointName='cat-dog-classifier-v1')
```

### Common Endpoint Issues and Solutions

1. **Timeout Errors**
   - Symptom: ReadTimeoutError during prediction
   - Solution: Upgrade instance type (ml.t2.large or higher)
   - Check CloudWatch logs for model loading issues

2. **Model Loading Failures**
   - Symptom: 500 error on /ping health check
   - Solution: Verify inference.py matches training architecture
   - Add logging in model_fn() to debug initialization

3. **Memory Issues**
   - Symptom: Model crashes or endpoint unhealthy
   - Solution: Monitor CloudWatch metrics
   - Consider instance types with more memory

### Development Tips

1. **Local Testing First**

   ```python
   # Test model locally before deployment
   model = CatDogCNN()
   model.load_state_dict(torch.load('model.pth'))
   model.eval()
   ```

2. **Debugging Endpoints**

   ```python
   # Check endpoint status
   def check_endpoint_status(endpoint_name):
       client = boto3.client('sagemaker')
       response = client.describe_endpoint(EndpointName=endpoint_name)
       return response['EndpointStatus']
   ```

3. **Cost Management**
   - Delete endpoints when not in use
   - Use auto-scaling for production
   - Monitor usage with AWS Cost Explorer

### Environment Setup

1. **Required IAM Permissions**

   ```json
   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Effect": "Allow",
               "Action": [
                   "s3:GetObject",
                   "s3:PutObject",
                   "s3:ListBucket"
               ],
               "Resource": [
                   "arn:aws:s3:::your-bucket/*",
                   "arn:aws:s3:::your-bucket"
               ]
           }
       ]
   }
   ```

2. **Dependencies**

   ```bash
   # Additional dependencies for development
   pip install sagemaker-studio-image-build
   pip install pytest
   pip install black  # for code formatting
   ```

### Best Practices

1. **Version Control**
   - Tag model versions in S3
   - Use semantic versioning for endpoints
   - Document model changes

2. **Testing**
   - Unit test inference functions
   - Test with various image sizes/formats
   - Benchmark endpoint performance

3. **Monitoring**
   - Set up CloudWatch alarms
   - Monitor model accuracy drift
   - Track endpoint invocations

4. **Security**
   - Use IAM roles with minimal permissions
   - Encrypt data in transit and at rest
   - Regular security audits

### Useful Commands

```bash
# Check endpoint logs
aws logs get-log-events --log-group-name /aws/sagemaker/Endpoints/cat-dog-classifier-v1

# Update endpoint
aws sagemaker update-endpoint --endpoint-name cat-dog-classifier-v1 --endpoint-config-name new-config

# List all endpoints
aws sagemaker list-endpoints
```

### Future Improvements

1. **Model Enhancements**
   - Implement model A/B testing
   - Add support for multiple model versions
   - Implement batch prediction capabilities

2. **Infrastructure**
   - Set up CI/CD pipeline
   - Add automated testing
   - Implement blue-green deployments

3. **Monitoring**
   - Add custom metrics
   - Set up automated retraining
   - Implement prediction logging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

[Your License Here]

## Resources

- Training Data: `s3://yolo-sagemaker/training_data/`
- Model Artifacts: `s3://yolo-sagemaker/pytorch-training-2025-03-06-16-44-33-803/output/`
- SageMaker Domain URL: [Your domain URL]

## Troubleshooting

Common issues and solutions:

1. S3 Permissions: Ensure proper S3 access policies and permissions boundaries
2. GPU vs CPU: Training uses GPU instance, local inference uses available hardware
3. Path Issues: Check relative paths when running scripts from different locations
