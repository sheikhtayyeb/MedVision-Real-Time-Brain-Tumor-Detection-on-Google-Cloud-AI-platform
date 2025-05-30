# MedVision-Real-Time-Brain-Tumor-Detection-on-Google-Cloud-AI-platform

This project implements a CNN-based deep learning model for classifying brain tumor MRI images into four different categories. The model achieves over 90% accuracy on the validation dataset and is deployed on Google Cloud AI Platform for real-time inference.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Installation Requirements](#installation-requirements)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Data Augmentation](#data-augmentation)
- [Training Process](#training-process)
- [Results](#results)
- [Deployment](#deployment)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [License](#license)


## Project Overview

This project aims to develop a robust deep learning model capable of accurately classifying brain MRI scans into different tumor categories. Early and accurate tumor classification is critical for treatment planning and patient management in clinical settings. The CNN model in this project was trained on a dataset of brain MRI images and deployed as a service using Google Cloud AI Platform for potential clinical applications.

## Dataset Description

The dataset consists of brain MRI images divided into four classes:

- Glioma
- Meningioma
- No tumor (healthy)
- Pituitary tumor

The dataset is organized in the following structure:

```
brain_tumor/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```
Dataset link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?select=Training
The model was trained on 4,003 images and validated on 1,311 images.

## Installation Requirements

To run this project, you'll need the following dependencies:

```
tensorflow>=2.8.0
matplotlib
numpy
pillow
google-cloud-aiplatform
```

If you're using Google Cloud, you'll also need to set up the Google Cloud SDK and authenticate your account.

## Project Structure

The project follows a standard machine learning workflow:

1. **Data Loading and Preparation**: Loading images from Google Cloud Storage
2. **Data Augmentation**: Applying transformations to training images
3. **Model Building**: Constructing the CNN architecture
4. **Training**: Training the model with callbacks for checkpointing
5. **Evaluation**: Assessing model performance
6. **Deployment**: Exporting and deploying the model to Google Cloud

## Model Architecture

The CNN model architecture consists of:

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(4, activation='softmax')
])
```

The model includes:

- 3 convolutional layers with ReLU activation and max pooling
- Fully connected layers with 128 and 64 neurons
- Dropout (0.2) for regularization
- Output layer with softmax activation for 4-class classification


## Data Augmentation

To improve model robustness and prevent overfitting, we implemented data augmentation using Keras' ImageDataGenerator:

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

These transformations create variations of the training images, effectively expanding the dataset and helping the model generalize better to unseen data.

## Training Process

The model was trained with the following parameters:

- Optimizer: Adam
- Loss function: Categorical Cross-Entropy
- Batch size: 32
- Image dimensions: 200×200 pixels
- Number of epochs: 50

We implemented several callbacks for training:

- ModelCheckpoint to save the best model
- EarlyStopping to prevent overfitting
- TensorBoard for monitoring training progress


## Results

The model achieved impressive performance metrics:

- Final validation accuracy: 90.92%
- Final validation loss: 0.2362

Training progression shows steady improvement in both accuracy and loss metrics over time:

## Deployment

The trained model was deployed on Google Cloud AI Platform for inference:

1. The model was exported in TensorFlow SavedModel format
2. Uploaded to Google Cloud Storage
3. Registered as a model in Vertex AI
4. Deployed to an endpoint for real-time predictions

## Usage

To use the deployed model for inference:

```python
from google.cloud import aiplatform
import numpy as np
from PIL import Image

# Load and preprocess image
image = Image.open("path_to_image.jpg")
image_resized = image.resize((200, 200))
image_array = np.array(image_resized)/255.

# Convert to list format for API
test_image = image_array.tolist()

# Set up endpoint
endpoint = aiplatform.Endpoint(
    endpoint_name="projects/$PROJECT/locations/us-central1/endpoints/5810194374633455616"
)

# Get prediction
response = endpoint.predict(instances=[test_image])

# Process results
class_labels = ["glioma", "meningioma", "notumor", "pituitary"]
predictions = response.predictions[^1_0]
predicted_class_idx = np.argmax(predictions)
predicted_label = class_labels[predicted_class_idx]
confidence = np.max(predictions) * 100

print(f"Predicted class: {predicted_label} (confidence: {confidence:.2f}%)")
```


## Future Improvements

Potential enhancements for this project include:

- Experimenting with more advanced architectures (ResNet, EfficientNet)
- Implementing explainable AI techniques to visualize model decisions
- Adding more data or applying transfer learning for improved performance
- Creating a web application interface for easier access to the model


## License

This project is licensed under the MIT License 


