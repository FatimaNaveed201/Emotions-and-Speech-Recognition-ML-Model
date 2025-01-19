import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for the progress bar

# Define Emotion labels based on the folder names
emotion_labels = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
emotion_map = {'angry': 0, 'fearful': 1, 'happy': 2, 'sad': 3, 'surprised': 4, 'neutral': 5}

def load_data_from_folders(dataset_path):
    emotion_data = {}  # To store data for each emotion
    emotion_counts = {}  # To store count of images per emotion

    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Dataset path does not exist: {dataset_path}")
        return None, None, None, None

    # Loop through each emotion folder with tqdm to show progress
    for emotion in tqdm(os.listdir(dataset_path), desc="Processing Emotion Folders"):
        emotion_folder = os.path.join(dataset_path, emotion)
        
        # Check if it's a valid folder
        if os.path.isdir(emotion_folder):
            count = 0
            # Loop through each image in the folder
            for image_name in tqdm(os.listdir(emotion_folder), desc=f"Reading images in {emotion}", leave=False):
                image_path = os.path.join(emotion_folder, image_name)
                img = cv2.imread(image_path)
                if img is not None:
                    count += 1

            # Store emotion counts
            emotion_counts[emotion] = count
            emotion_data[emotion] = count

    return emotion_data, emotion_counts

def calculate_target_sizes(emotion_counts):
    # Get the size of the "Neutral" class
    neutral_size = emotion_counts.get('neutral', 0)

    # Define target sizes for each class
    target_sizes = {
        'happy': neutral_size,  # Keep "Happy" as it is
        'surprised': neutral_size,  # Keep "Surprise" as it is
        'neutral': neutral_size,  # Keep "Neutral" as it is
        'sad': int(neutral_size * 0.85),  # Reduce "Sad" by 15%
        'angry': int(neutral_size * 0.85),  # Reduce "Angry" by 15%
        'fearful': int(neutral_size * 0.6)  # Reduce "Fear" by 30% compared to "Neutral"
    }
    return target_sizes

def plot_target_sizes(target_sizes):
    emotions = list(target_sizes.keys())
    target_counts = list(target_sizes.values())

    # Plotting target sizes
    plt.figure(figsize=(12, 6))

    # Plot target sizes
    plt.bar(emotions, target_counts, alpha=0.6, label="Target Size", color='salmon')

    plt.title("Target Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Specify the path to the dataset
dataset_path = r'C:\Users\Fatima Naveed\Documents\GitHub\Emotions-and-Speech-Recognition-ML-Model\EmotionsDataset\train' 

# Load the dataset
emotion_data, emotion_counts = load_data_from_folders(dataset_path)

# If dataset loaded successfully, calculate target sizes and plot the target size comparison
if emotion_data is not None and emotion_counts is not None:
    target_sizes = calculate_target_sizes(emotion_counts)
    plot_target_sizes(target_sizes)
else:
    print("Failed to load dataset.")