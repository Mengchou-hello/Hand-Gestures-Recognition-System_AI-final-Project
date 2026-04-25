# 🖐️ Hand Gesture Recognition System

## 📌 Overview

This project is a real-time **Hand Gesture Recognition System** built using **Computer Vision and Deep Learning**. It detects hand landmarks and classifies gestures using a trained neural network model.

The system can recognize gestures through a webcam and display predictions instantly.

## 🎯 Features

* Real-time hand detection using webcam
* Gesture classification using a trained CNN model
* Image preprocessing and dataset preparation pipeline
* Model evaluation with confusion matrix
* Clean and modular Python scripts

## 🛠️ Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* MediaPipe
* NumPy & Matplotlib


## 📂 Project Structure

```bash
Hand-Gesture-Recognition-System/
│
├── models/                     # Trained model files
├── Python scripts/            # All project scripts
│   ├── 0_split_dataset.py
│   ├── 1_crop_hands.py
│   ├── 2_train_model.py
│   ├── 3_evaluate.py
│   ├── 4_realtime_app.py
│
├── confusion_matrix.png       # Model evaluation result
├── hand_landmarker.task       # MediaPipe model
├── requirements.txt           # Dependencies
├── README.md
```
## ⚠️ Dataset Notice

The dataset is **not included** in this repository due to size limitations.

### 👉 To run this project:

1. Prepare your dataset and place it in:

```bash
/dataset
```

2. Run preprocessing:

```bash
python "Python scripts/0_split_dataset.py"
python "Python scripts/1_crop_hands.py"
```

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python "Python scripts/2_train_model.py"
```

### 3. Evaluate the model

```bash
python "Python scripts/3_evaluate.py"
```

### 4. Run real-time detection

```bash
python "Python scripts/4_realtime_app.py"
```

## 📊 Output Example

* Real-time gesture prediction via webcam
* Confusion matrix saved as:

```bash
confusion_matrix.png
```

## ⚠️ Notes

* Ensure your webcam is connected before running the real-time script
* Press **Q** to exit the camera window
* Large files such as datasets and virtual environments are excluded

## 📌 Future Improvements

* Improve model accuracy (target: 80–90%)
* Add more gesture classes
* Deploy as a web or mobile application
* Optimize performance for low-end devices

## 📜 License

This project is for educational purposes.

