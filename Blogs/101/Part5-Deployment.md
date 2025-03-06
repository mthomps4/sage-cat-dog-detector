## Part 5: Deployment

### Endpoint Creation

#### Instance Selection
```python
from sagemaker.model import Model

model = Model(
    model_data='s3://bucket/model.tar.gz',
    image_uri=image_uri,
    role=role,
    predictor_cls=PyTorchPredictor
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.c5.xlarge',
    endpoint_name='model-endpoint'
)
```

#### Configuration
```python
endpoint_config = {
    'EndpointConfigName': 'model-endpoint-config',
    'ProductionVariants': [{
        'VariantName': 'AllTraffic',
        'ModelName': 'model-name',
        'InitialInstanceCount': 1,
        'InstanceType': 'ml.c5.xlarge',
        'InitialVariantWeight': 1
    }]
}
```

### Testing

#### Endpoint Testing
```python
import json
import numpy as np

def test_endpoint(image_path):
    # Preprocess image
    image = preprocess_image(image_path)

    # Create payload
    payload = json.dumps({
        'instances': image.tolist()
    })

    # Get prediction
    response = predictor.predict(payload)
    return json.loads(response)