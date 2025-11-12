# Gender and Age Detection System
## Project Objective

This project aims to automatically predict a person’s gender and age from facial images using Convolutional Neural Networks (CNNs).
It demonstrates how deep learning can be applied to visual recognition tasks to support applications like targeted marketing, demographic analytics, and access control systems.

## Problem Statement

-> Businesses and digital systems often need to understand who their users are — not just in name, but by demographic traits like age and gender.
-> Manual data collection is inefficient, error-prone, and privacy-sensitive.
-> This project solves that by providing an automated, scalable, and accurate computer vision solution for age and gender prediction using facial images.

## Key Features

-> Detects gender (Male/Female) and age from facial images.

-> Built using deep learning (CNN) on the UTKFace dataset.

-> Achieves over 96% accuracy in gender prediction.

-> Trains a multi-output model — one branch predicts gender, another predicts age.

-> Includes data preprocessing, EDA, model training, and visualizations.

-> Visual performance tracking via accuracy/loss graphs.

## Tech Stack

-> Languages: Python
-> Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn, PIL
-> Dataset: UTKFace Dataset
-> Tools: Jupyter Notebook, GitHub

## How It Works

-> Data Extraction & Labeling: Images are parsed from filenames to extract age and gender.

-> Preprocessing: Each image is resized to 128×128 grayscale and normalized.

-> Model Architecture: A CNN with multiple Conv2D + MaxPooling layers learns facial features.

-> Two output heads:
gender_out: Binary classification (Male/Female)
age_out: Regression (predicts numerical age)

-> Training: The model is trained on 23,000+ images with 30 epochs and validation split.

-> Evaluation: Accuracy/loss visualizations track training and validation performance.

## Results

-> Gender Prediction Accuracy: ~96%

-> Age Prediction: Approximate MAE-based continuous prediction.

-> Model performs consistently across diverse age groups and facial features.

## Use Cases

-> Personalized recommendations and ads.

-> Demographic-based customer analytics.

-> Access control or attendance systems.

-> Social media content moderation and tagging.

## Future Improvements

-> Introduce transfer learning with pre-trained CNNs (e.g., VGGFace, ResNet).

-> Enhance age prediction with regression optimization.

-> Build a web-based interface for real-time predictions.
