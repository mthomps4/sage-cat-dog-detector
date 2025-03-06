## When to Choose Which Service

### Use AWS Batch When:

1. **Custom Infrastructure Needed**
   - Custom ML frameworks
   - Specific container requirements
   - Complex networking needs

2. **Cost Optimization Critical**
   - Large-scale training
   - Spot instance optimization
   - Resource sharing with non-ML workloads

3. **Example Use Case**
   ```python
   # AWS Batch: Custom Training Pipeline
   job_definition = {
       'jobDefinitionName': 'custom-ml',
       'containerProperties': {
           'image': 'custom-ml-image',
           'vcpus': 8,
           'memory': 32000,
           'command': ['train.py'],
           'environment': [
               {'name': 'CUDA_VISIBLE_DEVICES', 'value': 'all'}
           ]
       }
   }
   ```

### Use SageMaker When:

1. **ML-First Development**
   - Quick prototyping
   - Built-in algorithms needed
   - Managed ML workflow

2. **Example Use Case**
   ```python
   # SageMaker: Quick Model Development
   from sagemaker.pytorch import PyTorch

   estimator = PyTorch(
       entry_point='train.py',
       framework_version='2.0',
       instance_type='ml.p3.2xlarge',
       hyperparameters={
           'epochs': 10,
           'batch-size': 32
       }
   )
   ```