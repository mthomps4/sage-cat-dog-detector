## Infrastructure and Setup

### AWS Batch Setup

1. **Compute Environment**
   - Define infrastructure:
     ```json
     {
       "computeEnvironmentName": "ml-training",
       "type": "MANAGED",
       "computeResources": {
         "type": "EC2",
         "instanceTypes": ["p3.2xlarge"],
         "maxvCpus": 256,
         "subnets": ["subnet-xxx"],
         "securityGroupIds": ["sg-xxx"]
       }
     }
     ```
   - Manage scaling policies
   - Configure instance types

2. **Job Queues**
   - Priority-based scheduling
   - Multiple queue support
   - Queue-specific compute environments

3. **GPU Configuration**
   - NVIDIA driver setup
   - CUDA toolkit integration
   - GPU monitoring

### SageMaker Setup

1. **Domain and Studio**
   ```python
   import sagemaker

   # Create session
   session = sagemaker.Session()

   # Configure training
   estimator = sagemaker.PyTorch(
       entry_point='train.py',
       instance_type='ml.p3.2xlarge',
       framework_version='2.0'
   )
   ```

2. **Built-in GPU Support**
   - Pre-configured GPU instances
   - Optimized containers
   - Automatic driver management