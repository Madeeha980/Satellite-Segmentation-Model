Satellite Imagery Segmentation
This repository contains Jupyter notebooks for a satellite imagery segmentation project. The project utilizes a U-Net model built with TensorFlow and Keras to perform semantic segmentation on satellite images, classifying pixels into different categories such as buildings, roads, water, and vegetation.

Project Overview
The core of this project is a deep learning model that takes satellite images and generates a segmentation mask. The model is trained on a custom dataset (dubai_dataset) that includes both original images and their corresponding mask files. The process involves image patching, normalization, and training with custom loss functions to achieve accurate segmentation.

Dependencies
The following Python libraries are required to run this project:

numpy

scikit-learn

matplotlib

keras

tensorflow

wandb

patchify

opencv-python

A requirements.txt file with these dependencies is included in the repository.

Installation
Clone this repository:

Bash

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
Install the required packages using pip:

Bash

pip install -r requirements.txt
The notebooks are designed to run in a Google Colab environment. You will need to mount your Google Drive to access the dataset:

Python

from google.colab import drive
drive.mount('/content/drive')
Ensure your dataset is correctly placed in the specified root folder (/content/drive/MyDrive/Colab Notebooks/datasets/).

Usage
The primary notebook (Satellite_Imagery_Segmentation.ipynb) demonstrates the full workflow:

Data Loading and Preprocessing: Images and their corresponding masks are loaded from the /dubai_dataset/ folder. The images are read using OpenCV (cv2).

Image Patching: Images are split into smaller patches of size 256x256 pixels using the patchify library. This is essential for handling large satellite images and creating a sufficient number of training samples.

Normalization: The pixel values of the image patches are normalized using MinMaxScaler from scikit-learn.

Labeling: Mask images are converted from RGB colors to integer labels (0-5) representing different classes.

[60, 16, 152] -> Class 3 (Building)

[132, 41, 246] -> Class 1 (Land)

[110, 193, 228] -> Class 2 (Road)

[254, 221, 58] -> Class 4 (Vegetation)

[226, 169, 41] -> Class 0 (Water)

[155, 155, 155] -> Class 5 (Unlabeled)

Model Training: The U-Net model is trained on the preprocessed data.

Prediction: The trained model is used to predict segmentation masks on new images, which are then visualized using matplotlib.

Model Details
Architecture: U-Net

Custom Loss Functions: The model uses a combination of two custom loss functions for training:

jaccard_coef: A custom Jaccard coefficient for evaluation.

dice_loss: A custom Dice Loss function.

categorical_focal_loss: A custom Categorical Focal Loss function.

total_loss: A combination of Dice Loss and Categorical Focal Loss.

