## Part 6: API Integration

### API Gateway Setup

#### Direct Integration
```python
import boto3

api_client = boto3.client('apigateway')

# Create REST API
api = api_client.create_rest_api(
    name='ModelAPI',
    description='ML Model API'
)

# Create resource
resource = api_client.create_resource(
    restApiId=api['id'],
    parentId=api['rootResourceId'],
    pathPart='predict'
)

# Create POST method
api_client.put_method(
    restApiId=api['id'],
    resourceId=resource['id'],
    httpMethod='POST',
    authorizationType='NONE'
)
```

#### Lambda Integration
```python
def lambda_handler(event, context):
    """
    Lambda function for model inference
    """
    try:
        # Parse input
        body = json.loads(event['body'])
        image_data = body['image']

        # Call SageMaker endpoint
        response = runtime.invoke_endpoint(
            EndpointName='model-endpoint',
            ContentType='application/json',
            Body=json.dumps(image_data)
        )

        # Parse response
        result = json.loads(response['Body'].read())

        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': str(e)
        }
```