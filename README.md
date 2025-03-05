# Cat vs Dog Image Classifier

This project demonstrates a simple image classification model that distinguishes between cats and dogs. It's built as a learning exercise for AWS SageMaker, starting with local development and gradually moving to cloud deployment.

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

## Quick Start

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

3. Run the training script:

```bash
python src/train.py
```

## Development Phases

### Phase 1: Local Development ✅

- Basic CNN implementation
- Local training and testing
- Simple data pipeline

### Phase 2: SageMaker Development 🔄

- Moving to SageMaker training
- Creating SageMaker endpoint
- Testing predictions

### Phase 3: Production 🔄

- Model optimization
- Deployment pipeline
- Monitoring and logging

## Model Architecture

The current model is a simple CNN with:

- 2 convolutional layers
- Max pooling
- ReLU activation
- Final fully connected layer

## Dataset

- Input: RGB images (resized to 224x224)
- Classes: Cat, Dog
- Split: 6 training, 2 testing (for our sample dataset)

## Next Steps

- [ ] Test local training with sample images
- [ ] Adapt code for SageMaker training
- [ ] Create SageMaker endpoint
- [ ] Build inference pipeline

## Troubleshooting

### Common Issues

- If you get an error about missing modules, make sure your virtual environment is activated:

  ```bash
  source venv/bin/activate  # On Mac/Linux
  venv\Scripts\activate     # On Windows
  ```

- If you have issues with image loading, check that your images are valid JPEG/PNG files and are in the correct directories.

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
