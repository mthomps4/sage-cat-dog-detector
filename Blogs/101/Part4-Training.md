## Part 4: Training and Optimization

### Training Process

#### Job Configuration
```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    source_dir='src',
    framework_version='2.0',
    py_version='py39',
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    hyperparameters={
        'epochs': 10,
        'batch-size': 32,
        'learning-rate': 0.001
    },
    role=role
)

estimator.fit({
    'training': 's3://bucket/training',
    'validation': 's3://bucket/validation'
})
```

#### Monitoring
```python
from sagemaker.debugger import Rule, rule_configs

debugger_hook_config = DebuggerHookConfig(
    s3_output_path='s3://bucket/debug',
    hook_parameters={
        "save_interval": "100"
    }
)

rules = [
    Rule.sagemaker(
        rule_configs.loss_not_decreasing(),
        rule_parameters={
            "base_trial": "training-job-name"
        }
    )
]
```

### Model Optimization

#### Hyperparameter Tuning
```python
from sagemaker.tuner import HyperparameterTuner

hyperparameter_ranges = {
    'learning-rate': ContinuousParameter(0.001, 0.1),
    'batch-size': CategoricalParameter([16, 32, 64, 128]),
    'optimizer': CategoricalParameter(['adam', 'sgd'])
}

tuner = HyperparameterTuner(
    estimator,
    objective_metric_name='validation:accuracy',
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=[
        {'Name': 'validation:accuracy',
         'Regex': 'validation accuracy: ([0-9\\.]+)'}
    ],
    max_jobs=20,
    max_parallel_jobs=2
)