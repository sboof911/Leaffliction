![Plant Disease Classification](Convolutional-Neural-Network.jpg)
# Leaffliction - Plant Disease Classification
This project uses a deep learning model to classify plant diseases from leaf images. The code utilizes a pre-trained ResNet-18 model and is fine-tuned for the task of classifying different plant diseases. The project is divided into two parts: training the model (`train.py`) and making predictions on new images (`predict.py`).

## Features
- **Model Training**: Train a ResNet-based model on a custom dataset of plant disease images.
- **Image Augmentation**: Random transformations such as resizing, blurring, and grayscale are applied to the training images for better generalization.
- **Dataset Handling**: Includes custom dataset class `PlantDiseaseDataset` that loads and preprocesses images, balances data, and saves intermediate results.
- **Prediction**: Uses the trained model to classify new images and visualize results.

## Prerequisites

Before you begin, make sure you have the following installed:

- Python 3.8+
- Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Dataset

This project requires a dataset of plant disease images. The dataset should be organized in subfolders where each subfolder contains images of a specific disease class. An example folder structure would look like this:

```
data_train/
│
├── Disease1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Disease2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

## Training

To train the model, run the following command:

```
python train.py --data_train <path_to_dataset> --total_images <number_of_images>
```

- `<path_to_dataset>`: Path to the folder containing the dataset.
- `<number_of_images>`: Total number of images to be used for training. If the total number is not specified, all available images will be used.

## Example:
```
python train.py --data_train ./data_train --total_images 1000
```

## Model Saving:
The model will be saved as `plant_disease_model.pth` after training is completed. The model includes the state dictionary and class labels.

## Prediction:

To make predictions on new images, use the following command:

```
python predict.py <image_path1> <image_path2> ...
```

The script will load the trained model and output the predicted class for each image along with a visual display.

## Example:
```
python predict.py ./test_images/leaf1.jpg ./test_images/leaf2.jpg
```
