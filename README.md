# ASL Sign Recognition Project

This project aims to develop a system for recognizing American Sign Language (ASL) signs using machine learning techniques and MediaPipe for landmark extraction.

## Overview

The system is designed to capture ASL signs from video input, process the data using MediaPipe Holistic, and classify the signs using trained machine learning models. The following steps outline the workflow:

1. **Capturing ASL Signs**: Use `capture_signs.py` to capture video input and extract hand, pose, and face landmarks.
2. **Training a Model**: Use `train_new_AI.ipynb` to train a new AI model based on the captured landmarks.
3. **Testing the Model**: Use `TESTER_SUR_DATASET.ipynb` and `Test_On_Example.ipynb` to evaluate the model on datasets and examples.

---

## Components

### 1. `capture_signs.py`

This script is used to capture landmarks from real-time video input.

- **Description**:
  - Uses MediaPipe Holistic to extract facial, hand, and body landmarks.
  - Saves the extracted data as `.parquet` files for further processing.
- **How to Run**:
  ```bash
  python capture_signs.py
- **Output**:
  - A `.parquet` file containing spatial coordinates of the landmarks.

---

### 2. `train_new_AI.ipynb`

This Jupyter notebook is for training a machine learning model.

- **Description**:
  - Implements a stacking algorithm to improve model performance. 
  - Multiple base models are trained independently on the same landmark data, generating predictions that are used as features for a meta-model.
  - The meta-model combines the outputs of the base models to make a final prediction, leveraging their collective strengths while compensating for individual weaknesses.
  - The stacking approach enhances accuracy and robustness by reducing variance and bias in predictions.
---

### 3. `TESTER_SUR_DATASET.ipynb`

This notebook evaluates the trained model on a test dataset.

- **Description**:
  - Loads a pre-trained TensorFlow Lite model.
  - Tests the model on new, unseen data for accuracy and robustness.
- **How to Use**:
  - Provide the path to the test dataset.
  - Execute the cells to generate evaluation metrics.

---

### 4. `Test_On_Example.ipynb`

This notebook is for testing the model on individual ASL sign examples.

- **Description**:
  - Loads a specific `.parquet` file.
  - Predicts the ASL sign based on extracted landmarks.
- **How to Use**:
  - Supply a `.parquet` file of landmarks for a single sign.
  - Run the notebook to see the model's prediction.

---

## Prerequisites

Before running the project, ensure the following are set up:

### Software Requirements
- Python 3.8 or higher
- Required Libraries:
  - TensorFlow
  - MediaPipe
  - OpenCV
  - Pandas

### Hardware Requirements
- A webcam for real-time video input.

---

## Instructions

1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install tensorflow mediapipe opencv-python pandas
3. Use `capture_signs.py` to collect landmark data for ASL signs.
4. Train the model using `train_new_AI.ipynb`.
5. Test the model using the provided test notebooks.

---


## References and Links

- [Kaggle ASL Signs Competition Overview](https://www.kaggle.com/competitions/asl-signs/overview)
- [MediaPipe Documentation](https://google.github.io/mediapipe/solutions/holistic.html)


