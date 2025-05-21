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
- [Acknowledgments](#acknowledgments)


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

The model was trained on 4,003 images and validated on 1,311 images[^1_9].

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

To improve model robustness and prevent overfitting, we implemented data augmentation using Keras' ImageDataGenerator[^1_10]:

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

These transformations create variations of the training images, effectively expanding the dataset and helping the model generalize better to unseen data[^1_5].

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
    endpoint_name="projects/rhic-innovation/locations/us-central1/endpoints/5810194374633455616"
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

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to TensorFlow and Keras teams for providing the deep learning framework
- Google Cloud for providing the infrastructure for model deployment
- Contributors to the brain tumor dataset

<div style="text-align: center">⁂</div>

[^1_1]: Brain_tumor_classification_CNN.pdf

[^1_2]: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

[^1_3]: https://studymachinelearning.com/keras-imagedatagenerator-with-flow_from_directory/

[^1_4]: https://docs2.w3cub.com/tensorflow~python/tf/keras/preprocessing/image/imagedatagenerator/

[^1_5]: https://www.datacamp.com/tutorial/complete-guide-data-augmentation

[^1_6]: https://github.com/catiaspsilva/README-template

[^1_7]: https://www.youtube.com/watch?v=Rv6UFGNmNZg

[^1_8]: https://deepdatascience.wordpress.com/2016/11/10/documentation-best-practices/

[^1_9]: https://www.archbee.com/blog/readme-document-elements

[^1_10]: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

[^1_11]: https://github.com/keras-team/keras/issues/3946

[^1_12]: https://github.com/sfbrigade/data-science-wg/blob/master/dswg_project_resources/Project-README-template.md

[^1_13]: https://pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/

[^1_14]: https://deepsense.ai/blog/standard-template-for-machine-learning-projects-deepsense-ais-approach/

[^1_15]: https://keras.io/api/data_loading/image/

[^1_16]: https://hackernoon.com/how-to-create-an-engaging-readme-for-your-data-science-project-on-github

[^1_17]: https://www.cloudthat.com/resources/blog/image-data-augmentation-using-keras-api-imagedatagenerator

[^1_18]: https://medium.datadriveninvestor.com/how-to-write-a-good-readme-for-your-data-science-project-on-github-ebb023d4a50e

[^1_19]: https://stackoverflow.com/questions/70080062/how-to-correctly-use-imagedatagenerator-in-keras

[^1_20]: https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/

[^1_21]: https://github.com/ethen8181/machine-learning/blob/master/README.md

[^1_22]: https://www.youtube.com/watch?v=4ATucrptdYA

[^1_23]: https://data.research.cornell.edu/data-management/sharing/readme/

[^1_24]: https://github.com/mbadry1/Top-Deep-Learning/blob/master/readme.md

