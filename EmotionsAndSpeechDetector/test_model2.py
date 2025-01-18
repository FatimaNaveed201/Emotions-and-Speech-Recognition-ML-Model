import os
import numpy as np
import cv2
import tensorflow as tf
from tf_keras.models import load_model
from tf_keras.preprocessing.image import img_to_array
from tf_keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Define Emotion labels based on the folder names
emotion_labels = ["Angry",  "Fearful", "Happy", "Sad", "Surprised", "Neutral"]
emotion_map = {'angry': 0, 'fearful': 1, 'happy': 2, 'sad': 3, 'surprised': 4, 'neutral': 5}

# Load the pre-trained model
try:
    #model = load_model("test_model3.h5")
    model = load_model("detect_model3.h5")
    print("Model loaded successfully.")
except:
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

# Function to predict emotion of a random image from the test folder
def predict_random_image(image_path):
    # Preprocess the image and predict the emotion
    processed_image = preprocess_image(image_path)
    if processed_image is not None:
        prediction = model.predict(processed_image)
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        
        # Extract emotion folder name as true label and convert to lowercase to match emotion_map
        true_emotion = image_path.split(os.sep)[-2].lower()

        # Display prediction and true emotion
        print(f"Predicted Emotion: {predicted_emotion}, True Emotion: {true_emotion.capitalize()}")
        return np.argmax(prediction), emotion_map.get(true_emotion, -1)  # Use .get() to avoid KeyError
    else:
        print("Error reading image.")
        return None, None

# Function to evaluate model accuracy on a test set
def evaluate_model(test_folder, max_images_per_class=100):
    y_true = []
    y_pred = []
    
    # Loop through each emotion folder
    for emotion in os.listdir(test_folder):
        emotion_folder = os.path.join(test_folder, emotion)
        if os.path.isdir(emotion_folder):
            images_processed = 0  # Counter for images processed in each folder
            for image_name in tqdm(os.listdir(emotion_folder), desc=f"Processing {emotion}"):
                if images_processed >= max_images_per_class:
                    break  # Stop after processing max_images_per_class images
                image_path = os.path.join(emotion_folder, image_name)
                if os.path.isfile(image_path):  # Ensure we're working with a file
                    predicted, true = predict_random_image(image_path)
                    if predicted is not None:
                        y_true.append(true)
                        y_pred.append(predicted)
                    images_processed += 1 

    # Calculate and display accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy of model: {accuracy * 100:.2f}%")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(emotion_map.values()))
    print("Confusion Matrix:\n", cm)

    # Display confusion matrix as a graph
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotion_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

# Specify the path to the test folder
test_folder = "EmotionsAndSpeechDetector/test"  # Path to the test folder

# Evaluate model accuracy with a limit of 100 images per class
evaluate_model(test_folder, max_images_per_class=5)