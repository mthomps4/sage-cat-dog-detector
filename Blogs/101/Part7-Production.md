## Part 7: Production Considerations

### Monitoring

#### CloudWatch Setup
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Create custom metric
cloudwatch.put_metric_data(
    Namespace='ML/ModelMetrics',
    MetricData=[{
        'MetricName': 'InferenceLatency',
        'Value': latency,
        'Unit': 'Milliseconds'
    }]
)

# Create alarm
cloudwatch.put_metric_alarm(
    AlarmName='HighLatency',
    MetricName='InferenceLatency',
    Namespace='ML/ModelMetrics',
    Threshold=100,
    ComparisonOperator='GreaterThanThreshold',
    Period=300,
    EvaluationPeriods=2
)
```

### Cost Optimization

#### Auto Scaling
```python
app_scaling = boto3.client('application-autoscaling')

# Register scalable target
app_scaling.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=4
)

# Configure scaling policy
app_scaling.put_scaling_policy(
    PolicyName='ScaleOnInvocations',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        }
    }
)
```

Would you like me to:
1. Add more code examples?
2. Expand any specific section?
3. Add troubleshooting sections?
4. Add best practices for each section?