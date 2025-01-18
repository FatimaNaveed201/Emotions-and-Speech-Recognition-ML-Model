import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cv2
import tensorflow as tf
from tf_keras.models import Sequential, load_model
from tf_keras.callbacks import EarlyStopping
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tf_keras.utils import to_categorical
from tf_keras.preprocessing.image import img_to_array
from tf_keras import models
from tf_keras.optimizers import Adam
from tf_keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from collections import deque
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm.keras import TqdmCallback  # Added tqdm for progress bar
import tkinter as tk
from PIL import Image, ImageTk  # For GUI image display
import threading
import speech_recognition as sr
from googletrans import Translator
from collections import deque
from sklearn.utils import resample 

# Define Emotion labels based on the folder names
emotion_labels = ["Angry",  "Fear", "Happy", "Sad", "Surprise", "Neutral"]
emotion_map = {'angry': 0, 'fearful': 1, 'happy': 2, 'sad': 3, 'surprised': 4, 'neutral': 5}

# Define available languages for speech recognition
SUPPORTED_LANGUAGES = ['en', 'ar', 'hi', 'ur', 'tl', 'es', 'ta']

# Initialize webcam
cap = cv2.VideoCapture(0)

def load_data_from_folders(dataset_path):
    X = []
    y = []

    emotion_data = {}  # To store data for each emotion

    # Loop through each emotion folder
    for emotion in os.listdir(dataset_path):
        emotion_folder = os.path.join(dataset_path, emotion)
        if os.path.isdir(emotion_folder):
            emotion_samples = []
            # Loop through each image in the folder
            for image_name in os.listdir(emotion_folder):
                image_path = os.path.join(emotion_folder, image_name)
                img = cv2.imread(image_path)
                if img is not None:
                    # Convert image to grayscale, resize to 48x48, and normalize
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (48, 48)) / 255.0
                    emotion_samples.append(resized)
            
            # Store the emotion samples
            emotion_data[emotion] = emotion_samples

    # Get the size of the "Neutral" class
    neutral_size = len(emotion_data.get('neutral', []))

    # Define target sizes for each class
    target_sizes = {
        'happy': neutral_size,  # Keep "Happy" as it is
        'surprised': neutral_size,  # Keep "Surprise" as it is
        'neutral': neutral_size,  # Keep "Neutral" as it is
        'sad': int(neutral_size * 0.85),  # Reduce "Sad" by 15%
        'angry': int(neutral_size * 0.85),  # Reduce "Angry" by 15%
        'fearful': int(neutral_size * 0.6)  # Reduce "Fear" by 30% compared to "Neutral"
    }

    # Balance the classes based on target sizes
    for emotion, target_size in target_sizes.items():
        if emotion in emotion_data:
            emotion_data[emotion] = resample(emotion_data[emotion], n_samples=target_size, random_state=42)

    # Combine all the emotion data back into X and y
    for emotion, samples in emotion_data.items():
        X.extend(samples)
        y.extend([emotion_map[emotion]] * len(samples))

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Add channel dimension (for grayscale images)
    X = X[..., np.newaxis]

    # One-hot encode the labels
    y = to_categorical(y, num_classes=len(emotion_labels))

    return X, y

# Build the CNN Model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(6, activation='softmax')  # Number of emotions
    ])

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(X_train, y_train):
    model = build_model()
    # Initialize the TensorBoard callback
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
    
    # Using TQDM in model training for progress bar
    history = model.fit(X_train, y_train, epochs=35, batch_size=32, verbose=1, 
                        callbacks=[tensorboard_callback, TqdmCallback()])  # Add TQDMCallback
    
    # Display final training accuracy
    final_accuracy = history.history['accuracy'][-1]
    print(f"Training completed. Final Training Accuracy: {final_accuracy * 100:.2f}%")
    
    model.save("EmotionsModel.h5")  # Save model to file
    return model

# Preprocess frame for emotion prediction
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (48, 48))
    normalized_frame = resized_frame / 255.0
    reshaped_frame = np.reshape(normalized_frame, (1, 48, 48, 1))
    return reshaped_frame

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_emotion():
    emotion_history = deque(maxlen=10)  # Smoothing emotions over time
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the face region for emotion detection
            face_roi = frame[y:y+h, x:x+w]
            processed_frame = preprocess_frame(face_roi)
            
            # Predict emotion on the cropped face
            predictions = model.predict(processed_frame)
            emotion_index = np.argmax(predictions)
            detected_emotion = emotion_labels[emotion_index]

            # Update emotion history for smoothing
            emotion_history.append(detected_emotion)

        # Ensure most_common_emotion has a value before being used
        if emotion_history:
            most_common_emotion = max(set(emotion_history), key=emotion_history.count)
        else:
            most_common_emotion = "Unknown"  # Default value if history is empty

        # Convert the frame for tkinter compatibility
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)

        # Update the label with the new image and emotion
        label_img.config(image=image_tk)
        label_img.image = image_tk
        emotion_label.config(text=most_common_emotion)

        # Update the window
        window.update_idletasks()
        window.update()

# Speech recognition and translation
translator = Translator()

# Global Variables for Speech Recognition
is_listening = False  # To control the listening state
last_result = ""
listening_event = threading.Event()  # This event will control the listening state

# Start and stop listening properly
def toggle_listening():
    global is_listening, last_result
    if is_listening:
        # Stop listening
        is_listening = False
        listening_event.set()  # Signal the thread to stop listening
        last_result = ""  # Clear the last result
        subtitle_label.config(text="Press the Start button to initiate speech recognition")
        start_stop_button.config(image=start_img)  # Switch to start button
    else:
        # Start listening in a new thread
        is_listening = True
        listening_event.clear()  # Reset the event to allow listening
        subtitle_label.config(text="Listening...")
        start_stop_button.config(image=stop_img)  # Switch to stop button
        threading.Thread(target=recognize_and_translate_speech, daemon=True).start()  # Run speech recognition in background

def recognize_and_translate_speech():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    global is_listening, last_result

    while is_listening:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                # Recognize speech
                audio = recognizer.listen(source)
                subtitle_label.config(text="Processing...")

                speech_text = recognizer.recognize_google(audio, show_all=False)
                detected_lang = translator.detect(speech_text).lang

                if detected_lang in SUPPORTED_LANGUAGES:
                    if detected_lang != 'en':
                        translation = translator.translate(speech_text, src=detected_lang, dest='en')
                        translated_text = translation.text
                        display_text = f"Original: {speech_text}\nTranslated: {translated_text}"
                    else:
                        display_text = f"English: {speech_text}"
                else:
                    display_text = f"Unsupported Language Detected: {detected_lang}"

                last_result = display_text
                # Update the subtitle_label with translated text in the main thread
                window.after(0, lambda: subtitle_label.config(text=display_text))  # Update GUI in the main thread
            
            except sr.UnknownValueError:
                last_result = "Could not understand audio"
                window.after(0, lambda: subtitle_label.config(text=last_result))
            except sr.RequestError as e:
                last_result = f"Request Error: {e}"
                window.after(0, lambda: subtitle_label.config(text=last_result))

    # If listening is stopped, reset the status
    if not is_listening:
        window.after(0, lambda: subtitle_label.config(text="Press the Start button to initiate speech recognition"))

# Create tkinter window with aesthetic improvements
window = tk.Tk()
window.title("Emotion Detection and Speech Recognition")
window.geometry("1550x780")  # Adjusted window size
window.configure(bg="#2e2e2e")

# Fonts and Styles for aesthetic look
font_large = ("Helvetica", 18, "bold")
font_medium = ("Helvetica", 14)
font_small = ("Helvetica", 12)

# Frames for different screens
frame_main = tk.Frame(window, bg="#2e2e2e")
frame_main.pack(fill="both", expand=True)

frame_emotion = tk.Frame(window, bg="#2e2e2e")
frame_speech = tk.Frame(window, bg="#2e2e2e")

# Functions for changing screens
def show_emotion_screen():
    frame_main.pack_forget()
    frame_emotion.pack(fill="both", expand=True)

def show_speech_screen():
    frame_main.pack_forget()
    frame_speech.pack(fill="both", expand=True)

def show_main_screen():
    frame_emotion.pack_forget()
    frame_speech.pack_forget()
    frame_main.pack(fill="both", expand=True)

# Load background images
bg_main = ImageTk.PhotoImage(Image.open(r"C:\Users\Fatima Naveed\Documents\GitHub\Emotions-and-Speech-Recognition-ML-Model\GUI Images\MainPage.png"))
bg_emotion = ImageTk.PhotoImage(Image.open(r"C:\Users\Fatima Naveed\Documents\GitHub\Emotions-and-Speech-Recognition-ML-Model\GUI Images\EmotionsPage.png"))
bg_speech = ImageTk.PhotoImage(Image.open(r"C:\Users\Fatima Naveed\Documents\GitHub\Emotions-and-Speech-Recognition-ML-Model\GUI Images\SpeechPage.png"))
emotion_img = tk.PhotoImage(file=r"C:\Users\Fatima Naveed\Documents\GitHub\Emotions-and-Speech-Recognition-ML-Model\GUI Images\EmotionButton.png") 
speech_img = tk.PhotoImage(file=r"C:\Users\Fatima Naveed\Documents\GitHub\Emotions-and-Speech-Recognition-ML-Model\GUI Images\SpeechButton.png") 
back_img = tk.PhotoImage(file=r"C:\Users\Fatima Naveed\Documents\GitHub\Emotions-and-Speech-Recognition-ML-Model\GUI Images\BackButton.png") 
start_img = ImageTk.PhotoImage(Image.open(r"C:\Users\Fatima Naveed\Documents\GitHub\Emotions-and-Speech-Recognition-ML-Model\GUI Images\StartButton.png").resize((200, 70)))
stop_img = ImageTk.PhotoImage(Image.open(r"C:\Users\Fatima Naveed\Documents\GitHub\Emotions-and-Speech-Recognition-ML-Model\GUI Images\StopButton.png").resize((200, 70)))

# Main Screen with Background
bg_label_main = tk.Label(frame_main, image=bg_main)
bg_label_main.place(relwidth=1, relheight=1)  # Set background to fill frame

# Button for Emotion Detection
btn_emotion = tk.Button(frame_main, image=emotion_img, command=show_emotion_screen, bg="#e0ccfc",
                         borderwidth=0, highlightthickness=0, relief="flat")  # Remove 3D effect and borders
btn_emotion.place(relx=0.5, rely=0.7, anchor="center")  # Centered placement

# Button for Speech Recognition
btn_speech = tk.Button(frame_main, image=speech_img, command=show_speech_screen, bg="#e0ccfc",
                        borderwidth=0, highlightthickness=0, relief="flat")  # Remove 3D effect and borders
btn_speech.place(relx=0.5, rely=0.8, anchor="center")  # Centered placement

# Ensure images don't get garbage collected
btn_emotion.image = emotion_img
btn_speech.image = speech_img

# Emotion Detection Screen with Background
bg_label_emotion = tk.Label(frame_emotion, image=bg_emotion)
bg_label_emotion.place(relwidth=1, relheight=1)  # Set background to fill frame

label_img = tk.Label(frame_emotion)
label_img.pack(pady=(145, 6))  # Padding 50 from the top, 10 from the bottom

emotion_label = tk.Label(frame_emotion, font=("Times New Roman", 20, "bold"), fg="#350960", bg="#e0ccfc")
emotion_label.pack(pady=1)

# Resize the image before placing it on the button
back_img_resized = back_img.subsample(2, 2)  # Adjust the scaling factor (2, 2) to make the image smaller

btn_back = tk.Button(frame_emotion, image=back_img_resized, command=show_main_screen, bg="#e0ccfc",
                         borderwidth=0, highlightthickness=0, relief="flat")
btn_back.place(relx=0.5, rely=0.9, anchor="center")  # Centered placement

# Speech Recognition Screen with Background
bg_label_speech = tk.Label(frame_speech, image=bg_speech)
bg_label_speech.place(relwidth=1, relheight=1)  # Set background to fill frame

subtitle_label = tk.Label(frame_speech, font=("Times New Roman", 20, "bold"), fg="#350960", bg="#e0ccfc", text="Press the Start button to initiate speech recognition")
subtitle_label.place(relx=0.5, rely=0.57, anchor="center")  # Centered placement

# Button for speech recognition start/stop
start_stop_button = tk.Button(frame_speech, image=start_img, command=toggle_listening, bg="#e0ccfc",
                               borderwidth=0, highlightthickness=0, relief="flat")
start_stop_button.place(relx=0.5, rely=0.7, anchor="center")  # Centered placement

btn_back = tk.Button(frame_speech, image=back_img_resized, command=show_main_screen, bg="#e0ccfc",
                         borderwidth=0, highlightthickness=0, relief="flat")
btn_back.place(relx=0.5, rely=0.88, anchor="center")  # Centered placement

# Load pre-trained model or train new model
try:
    model = load_model("EmotionsModel.h5") # detect1,detect2, model 3, 5, 11 are good
    print("Model loaded successfully.")
except:
    print("Model not found. Training a new model.")
    X, y = load_data_from_folders(r"C:\Users\Fatima Naveed\Documents\GitHub\Emotions-and-Speech-Recognition-ML-Model\EmotionsDataset\train")
    model = train_model(X, y)

# Start threads
emotion_thread = threading.Thread(target=detect_emotion, daemon=True)
emotion_thread.start()

speech_thread = threading.Thread(target=recognize_and_translate_speech, daemon=True)
speech_thread.start()

# Start GUI main loop
window.mainloop()

cap.release()
cv2.destroyAllWindows()