# Complete Guide to AWS SageMaker Setup

## Before You Start

### Account Setup (15-30 minutes)

1. **AWS Account Creation**

   ```bash
   1. Go to aws.amazon.com
   2. Click "Create an AWS Account"
   3. Follow signup process
   4. Add payment method
   ```

2. **Budget Alert Setup** (Important!)

   ```bash
   1. Go to AWS Billing Dashboard
   2. Create budget alert for $50
   3. Add email notification
   ```

3. **Required IAM Permissions**
   - Go to IAM Console
   - Create new user with these permissions:

     ```json
     {
         "Version": "2012-10-17",
         "Statement": [
             {
                 "Effect": "Allow",
                 "Action": [
                     "sagemaker:*",
                     "s3:*",
                     "logs:*",
                     "iam:CreateRole",
                     "iam:AttachRolePolicy"
                 ],
                 "Resource": "*"
             }
         ]
     }
     ```

## Step 1: Environment Setup (20-30 minutes)

### Create a SageMaker Domain

1. **Navigate to SageMaker**

   ```bash
   1. Open AWS Console
   2. Search for "SageMaker"
   3. Select your region (top right) - use us-east-1 if unsure
   ```

2. **Domain Setup**

   ```bash
   1. Click "Domains" → "Create Domain"
   2. Choose "Quick setup"
   3. Domain name: "your-name-domain"
   4. User profile: "your-name"
   5. Execution role: "Create new role"
   6. Click through permission prompts
   ```

3. **Verify Setup**

   ```bash
   # Wait for status to show "InService"
   # This can take 5-10 minutes
   ```

4. Create Project Profile

   ```bash
   1. In the SageMaker Studio, navigate to the "Projects" tab.
   2. Click on "Create project profile".
   3. Enter a name for your project profile (e.g., "my-project-profile").
   4. Select the appropriate execution role that you created earlier.
   5. Configure any additional settings as needed (e.g., VPC settings, tags).
   6. Click "Create" to finalize the project profile setup.
   7. Wait for the confirmation message indicating that the project profile has been created successfully.
   ```

## Step 2: Project Setup (15-20 minutes)

### Launch Studio

1. **Access Studio**

   ```bash
   1. Click "Launch Studio"
   2. First launch takes 5-7 minutes
   3. You'll see JupyterLab interface
   ```

2. **Create Project Structure**

   ```bash
   # In Studio Terminal
   mkdir my_project
   cd my_project
   mkdir data models notebooks src
   ```

3. **Install Required Packages**

   ```bash
   # Create requirements.txt
   pip install ipykernel
   python -m ipykernel install --user
   ```

## Step 3: Data Preparation (30-45 minutes)

### Set Up S3 Storage

```python
# In a new notebook
import boto3
import sagemaker

# Initialize session
session = sagemaker.Session()
bucket = session.default_bucket()
prefix = 'your-project-name'

# Create S3 bucket if needed
s3 = boto3.client('s3')
try:
    s3.head_bucket(Bucket=bucket)
except:
    s3.create_bucket(Bucket=bucket)
```

### Upload Training Data

```python
# Upload data to S3
train_path = session.upload_data(
    path='data/train',
    bucket=bucket,
    key_prefix=f'{prefix}/train'
)

# Verify upload
print(f"Training data location: {train_path}")
```

## Step 4: Model Training (1-2 hours)

### Create Training Script

1. **Create `train.py`**

   ```python
   # src/train.py
   import argparse
   import os
   import torch

   def parse_args():
       parser = argparse.ArgumentParser()
       parser.add_argument('--epochs', type=int, default=10)
       parser.add_argument('--batch-size', type=int, default=32)
       parser.add_argument('--learning-rate', type=float, default=0.001)
       return parser.parse_args()

   def train(args):
       # Your training logic here
       pass

   if __name__ == '__main__':
       args = parse_args()
       train(args)
   ```

2. **Configure Training Job**

   ```python
   from sagemaker.pytorch import PyTorch

   estimator = PyTorch(
       entry_point='train.py',
       source_dir='src',  # Important: point to source directory
       role=role,
       framework_version='2.0',
       py_version='py39',
       instance_count=1,
       instance_type='ml.m5.xlarge',
       hyperparameters={
           'epochs': 10,
           'batch-size': 32,
           'learning-rate': 0.001
       }
   )
   ```

3. **Start Training**

   ```python
   # This will take time depending on data size
   estimator.fit({
       'train': train_path,
       'validation': validation_path
   })
   ```

## Step 5: Model Deployment (30-45 minutes)

### Create Inference Script

1. **Create `inference.py`**

   ```python
   # src/inference.py
   import torch

   def model_fn(model_dir):
       # Load model from model_dir
       model = torch.load(f"{model_dir}/model.pth")
       return model

   def input_fn(request_body, content_type):
       # Transform input data
       return transform_input(request_body)

   def predict_fn(input_data, model):
       # Make prediction
       return model(input_data)

   def output_fn(prediction, accept):
       # Transform output
       return transform_output(prediction)
   ```

2. **Deploy Model**

   ```python
   predictor = estimator.deploy(
       initial_instance_count=1,
       instance_type='ml.t2.medium',
       endpoint_name='your-model-endpoint'
   )
   ```

## Step 6: Testing (15-20 minutes)

### Test Endpoint

```python
# Test with sample data
import json

# Prepare test data
test_data = {"data": your_test_data}

# Make prediction
response = predictor.predict(json.dumps(test_data))
print(f"Prediction: {response}")
```

## Step 7: Cleanup (5-10 minutes)

### Important: Delete Resources

```python
# Delete endpoint (IMPORTANT to avoid charges)
predictor.delete_endpoint()

# Delete notebook instance when done
# Via AWS Console:
# SageMaker → Notebook instances → Actions → Stop/Delete
```

## Troubleshooting Guide

### Common Issues

1. **Permission Errors**

   ```bash
   # Check IAM role
   1. Go to IAM Console
   2. Verify role permissions
   3. Add missing policies
   ```

2. **Resource Limits**

   ```bash
   # If you hit limits
   1. Go to Service Quotas
   2. Request increase
   ```

3. **Training Failures**
   - Check CloudWatch logs
   - Verify data paths
   - Check instance capacity

## Cost Management

### Daily Checklist

1. Stop notebook instances when not in use
2. Delete endpoints after testing
3. Monitor CloudWatch usage
4. Check Billing Dashboard

### Estimated Costs

- Notebook Instance (ml.t3.medium): ~$0.05/hour
- Training (ml.m5.xlarge): ~$0.23/hour
- Endpoint (ml.t2.medium): ~$0.05/hour

## Next Steps

1. Set up automated retraining
2. Implement monitoring
3. Optimize costs
4. Set up CI/CD

## Additional Resources

- [Official SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples)
- [AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/)
- [SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Community Forums](https://forums.aws.amazon.com/)
