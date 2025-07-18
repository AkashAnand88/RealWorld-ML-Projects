# American Sign Language Detection (ASL Detector)

*ASL Detector* is a real-time hand gesture recognition system designed to detect and classify *American Sign Language (ASL)* alphabets using computer vision and machine learning. It empowers silent communication by recognizing hand gestures through a webcam and mapping them to ASL letters.

## Overview

This project uses *MediaPipe* for real-time hand landmark detection and an *MLP Classifier* to recognize ASL letters (A–Z). It consists of three main phases:

1. *Data Collection* – Capture hand landmarks and save as CSV files
2. *Model Training* – Train a multi-layer perceptron classifier
3. *Real-Time Prediction* – Predict ASL letters live from webcam

## Features

-  Real-time gesture detection via webcam  
-  Hand landmark extraction (21 points)  
-  Alphabet classification (A–Z)  
-  Fast and lightweight (runs on CPU)  
-  Easily customizable and extensible

## Tech Stack

- *Python* – Core language  
- *OpenCV* – Webcam video capture and display  
- *MediaPipe* – Hand tracking and landmark detection  
- *Scikit-learn* – Machine learning model (MLP Classifier)  
- *NumPy / Pandas* – Data manipulation and feature engineering  


## ASL-Detection/
├── data_collection.py # Script to collect gesture data
├── model_training.py # Script to train the MLP model
├── asl_detector.py # Real-time detection script
└── README.md # Project documentation

###  Clone the Repository

```bash
git clone https://github.com/AkashAnand88/ASL Detection.git
cd ASL Detection

## Collect Hand Gesture Data
python collector.py

## Train the Model
python train.py

## Run Real-Time Detector
python detect.py
