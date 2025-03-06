## Development Experience

### AWS Batch Development

1. **Container Requirements**
   ```dockerfile
   FROM nvidia/cuda:11.0-base

   # ML framework setup
   RUN pip install torch torchvision

   # Training code
   COPY train.py /opt/ml/
   ENTRYPOINT ["python", "/opt/ml/train.py"]
   ```

2. **Job Submission**
   ```python
   import boto3

   batch = boto3.client('batch')

   response = batch.submit_job(
       jobName='ml-training-job',
       jobQueue='ml-queue',
       jobDefinition='ml-job-def',
       containerOverrides={
           'command': ['train.py', '--epochs', '10']
       }
   )
   ```

### SageMaker Development

1. **Training Script**
   ```python
   def train(args):
       model = create_model()
       train_dataset = load_data()

       for epoch in range(args.epochs):
           train_epoch(model, train_dataset)

       # SageMaker automatically handles model saving
   ```

2. **Experiment Tracking**
   ```python
   from sagemaker.experiments import Experiment

   experiment = Experiment(
       experiment_name='training-exp',
       description='Model training experiment'
   )

   with experiment.run('training-run'):
       estimator.fit({'training': 's3://data/train'})
   ```