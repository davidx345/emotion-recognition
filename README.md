# Emotion Recognition Web Application

<p align="center">
  <img src="images/Intro.gif" style="width: 300px;"/>
</p>

**Table of Contents**:

<!--ts-->
- [Overview](#overview)
- [Facial Emotion Recognition](#facial-emotion-recognition)
- [Project Objective](#project-objective)
- [Project Summary](#project-summary)
  - [1. Building and Training a CNN](#1-building-and-training-a-cnn)
  - [2. Face Detection](#2-face-detection)
  - [3. Hosting the App](#3-hosting-the-app)
- [Data](#data)
- [Running the App](#running-the-app)
- [References](#references)
<!--te-->

<br>

## Overview

This project creates a web app that detects human faces in images or video and classifies emotions using deep learning. The project includes:

- Building and training a Convolutional Neural Network (CNN) with Keras
- Implementing face detection using OpenCV
- Hosting the app in a browser using Flask with database integration

**→ Skills**: *Image Classification with Convolutional Neural Networks, Computer Vision (Face Detection), Model Deployment, Data Visualization, Data Augmentation* <br>
**→ Technologies**: *Python, Jupyter Notebook, Flask* <br>
**→ Libraries**: *Keras, TensorFlow, Flask, OpenCV, Scikit-learn, NumPy, Matplotlib, Seaborn, SQLAlchemy* <br>

<br>

## Facial Emotion Recognition

Facial Emotion Recognition (FER) detects human emotions using AI to analyze non-verbal cues. It enables systems to respond naturally to user emotions, with applications in tutoring, marketing, healthcare, and more.

<br>

## Project Objective

Develop a deep learning model for emotion recognition integrated with face detection, delivered as a web app that accepts image uploads and live video. The app stores user data and results in a database.

<br>

## Project Summary

### 1. Building and Training a CNN

This section is in the [Jupyter notebook](Emotion_Recognition_Notebook.ipynb). Steps include:

- Loading and augmenting data
- Creating, compiling, and training the model
- Making predictions

The FER2013 dataset is used. The model achieves ~69.5% validation accuracy.

### 2. Face Detection

Uses Haar Cascades and OpenCV for face detection.

### 3. Hosting the App

Flask hosts the app, loading the CNN model and Haar cascade. The app detects faces, predicts emotions, and saves to a database. HTML templates are in the `templates/` folder.

<br>

## Data

Uses the FER2013 dataset from Kaggle, with 35,887 grayscale images of 7 emotions: angry, disgust, fear, happy, neutral, sad, surprise.

<br>

## Running the App

To run locally:

1. Install dependencies: `pip install -r requirements.txt`
2. Set environment: `export FLASK_APP=app.py` (or `set FLASK_APP=app.py` on Windows)
3. Run: `flask run`
4. Open `http://localhost:5000/`

For deployment, use Render or similar.

<br>

## References

- Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*.
- Chollet, F. *Deep Learning with Python*.
- OpenCV documentation for Haar cascades.
- Flask documentation.
- Various online tutorials on emotion detection.