# SageMaker Models 101: Complete Development Guide

## Part 1: Getting Started

### Project Overview

#### What We're Building
In this comprehensive guide, we'll build a complete machine learning pipeline using Amazon SageMaker. We'll cover:
- Image classification model development
- Training on GPU instances
- Model deployment and serving
- API integration for predictions

#### Architecture Overview
```python
# High-level architecture diagram (ASCII)
'''
[Data Sources] → [SageMaker Processing] → [Training Job]
                                              ↓
[API Gateway] ← [SageMaker Endpoint] ← [Model Registry]
'''
```

#### Prerequisites
1. AWS Account with appropriate permissions
2. Basic Python knowledge
3. Understanding of ML concepts
4. Local development environment

### Environment Setup

#### AWS Account Setup
1. **Create AWS Account**
   ```bash
   # AWS CLI configuration
   aws configure
   AWS Access Key ID: YOUR_ACCESS_KEY
   AWS Secret Access Key: YOUR_SECRET_KEY
   Default region name: us-east-1
   ```

2. **IAM Roles and Permissions**
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "sagemaker:*",
           "s3:*",
           "logs:*"
         ],
         "Resource": "*"
       }
     ]
   }
   ```

3. **SageMaker Studio Setup**
   ```python
   import boto3

   # Create SageMaker client
   sm = boto3.client('sagemaker')

   # Create Domain
   response = sm.create_domain(
       DomainName='ml-domain',
       AuthMode='IAM',
       DefaultUserSettings={
           'ExecutionRole': 'arn:aws:iam::ACCOUNT:role/service-role/AmazonSageMaker-ExecutionRole'
       }
   )
   ```