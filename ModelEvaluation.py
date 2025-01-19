import os
import numpy as np
import cv2
import tensorflow as tf
from tf_keras.models import load_model
from tf_keras.preprocessing.image import img_to_array
from tf_keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import tkinter as tk
from tkinter import Scrollbar, Canvas
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tqdm import tqdm


# Define emotion labels based on folder names
emotion_labels = ["Angry", "Fearful", "Happy", "Sad", "Surprised", "Neutral"]
emotion_map = {'angry': 0, 'fearful': 1, 'happy': 2, 'sad': 3, 'surprised': 4, 'neutral': 5}

# Load the pre-trained model
try:
    model = load_model("EmotionsModel.h5")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model not found. Please train the model first.")
    exit()

# Preprocess the image for prediction
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48)) / 255.0
        reshaped = np.reshape(resized, (1, 48, 48, 1))
        return reshaped
    return None

# Predict emotion of a single image
def predict_image(image_path):
    processed_image = preprocess_image(image_path)
    if processed_image is not None:
        prediction = model.predict(processed_image)
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        true_emotion = image_path.split(os.sep)[-2].lower()
        print(f"Predicted Emotion: {predicted_emotion}, True Emotion: {true_emotion.capitalize()}")
        return np.argmax(prediction), emotion_map.get(true_emotion, -1)
    else:
        print(f"Error processing image: {image_path}")
        return None, None

# Evaluate model on the test dataset
def evaluate_model(test_folder, max_images_per_class=1000):
    y_true, y_pred = [], []
    emotion_counts_predicted = {i: 0 for i in range(len(emotion_labels))}
    emotion_counts_true = {i: 0 for i in range(len(emotion_labels))}
    
    for emotion in os.listdir(test_folder):
        emotion_folder = os.path.join(test_folder, emotion)
        if os.path.isdir(emotion_folder):
            images_processed = 0
            for image_name in tqdm(os.listdir(emotion_folder), desc=f"Processing {emotion}"):
                if images_processed >= max_images_per_class:
                    break
                image_path = os.path.join(emotion_folder, image_name)
                if os.path.isfile(image_path):
                    predicted, true = predict_image(image_path)
                    if predicted is not None:
                        y_true.append(true)
                        y_pred.append(predicted)
                        emotion_counts_predicted[predicted] += 1
                        emotion_counts_true[true] += 1
                    images_processed += 1
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(emotion_map.values()))
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotion_labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax1)
    ax1.set_title("Confusion Matrix")
    plt.show()
    
    # Pie Chart: Emotion Distribution
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 6))
    ax2.pie(list(emotion_counts_predicted.values()), labels=emotion_labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title("Emotion Distribution (Predicted)")
    ax3.pie(list(emotion_counts_true.values()), labels=emotion_labels, autopct='%1.1f%%', startangle=90)
    ax3.set_title("Emotion Distribution (True)")
    plt.show()
    
    # Histogram: True vs Predicted
    fig3, ax4 = plt.subplots(figsize=(6, 6))
    ax4.hist([y_true, y_pred], bins=np.arange(len(emotion_labels) + 1) - 0.5, 
             label=["True", "Predicted"], edgecolor='black', color=['blue', 'orange'])
    ax4.set_xticks(np.arange(len(emotion_labels)))
    ax4.set_xticklabels(emotion_labels, rotation=45)
    ax4.set_title("Histogram of True vs Predicted Labels")
    ax4.legend()
    plt.show()

# Specify the test folder path
test_folder = r"C:\Users\Fatima Naveed\Documents\GitHub\Emotions-and-Speech-Recognition-ML-Model\EmotionsDataset\test"

# Run the evaluation
evaluate_model(test_folder, max_images_per_class=6)