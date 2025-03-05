# Cat vs Dog Image Classifier

This project demonstrates a simple image classification model that distinguishes between cats and dogs. It's built as a learning exercise, starting with local development and eventually moving to AWS SageMaker deployment.

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

1. Clone the repository:

```bash
git clone <repository-url>
cd cat_dog_classifier
```

2. Run the setup script:

```bash
chmod +x setup.sh
./setup.sh
```

### Training

1. Ensure you have images in the data directory:
   - Place cat images in `data/cats/`
   - Place dog images in `data/dogs/`

2. Run the training script:

```bash
# Activate your virtual environment if not already activated
source venv/bin/activate

# Train the model
python src/train.py
```

3. The trained model will be saved as `cat_dog_model.pth`

### Inference

Use the trained model to classify new images:

```bash
chmod +x predict.sh

# Basic usage
./predict.sh path/to/image.jpg

# Specify a different model
./predict.py path/to/image.jpg path/to/other/model.pth
```

## AWS SageMaker (Future)

This section will be expanded as we implement SageMaker integration.

### Training on SageMaker

Coming soon!

### Deploying an Endpoint

Coming soon!

### Making Predictions

Coming soon!

## Model Architecture

The current model is a simple CNN with:

- 2 convolutional layers
- Max pooling
- ReLU activation
- Final fully connected layer

## Dataset

- Input: RGB images (resized to 224x224)
- Classes: Cat, Dog
- Split: 80% training, 20% testing

## Troubleshooting

### Common Issues

- If you get an error about missing modules, make sure your virtual environment is activated:

  ```bash
  source venv/bin/activate  # On Mac/Linux
  venv\Scripts\activate     # On Windows
  ```

- If you have issues with image loading, check that your images are valid JPEG/PNG files and are in the correct directories.

- If the model file isn't found during prediction, make sure you're running the script from the project root directory or specify the full path to the model.

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)

## Results

(To be updated with model performance metrics)

## Future Improvements

- [ ] Data augmentation
- [ ] More sophisticated model architecture
- [ ] SageMaker training implementation
- [ ] Model deployment
- [ ] Performance monitoring
