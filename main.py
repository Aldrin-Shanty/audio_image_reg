import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from face_reg import extract_face_features
from audio_reg import audio_reg

def train_face_model(data_dir="Image_Data", save_path="face_svm_model.pkl"):
    X = []
    y = []

    print("📷 Training face recognition model...")
    for person in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir):
            continue

        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            features = extract_face_features(img_path)

            if features is not None:
                X.append(features)
                y.append(person)
            else:
                print(f"⚠️ No face found in {img_path}")

    if not X:
        print("❌ No face data found. Training aborted.")
        return

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = SVC(kernel='linear', probability=True)
    model.fit(X, y_encoded)

    joblib.dump((model, le), save_path)
    print(f"✅ Face model trained and saved to {save_path}")

def train_audio_model(data_dir="data/audio", save_path="audio_svm_model.pkl"):
    X = []
    y = []

    print("🎙️ Training audio recognition model...")
    for person in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir):
            continue

        for audio_file in os.listdir(person_dir):
            audio_path = os.path.join(person_dir, audio_file)
            features = audio_reg(audio_path)
            if features is not None and len(features) > 0:
                X.append(features)
                y.append(person)
            else:
                print(f"⚠️ No features extracted from {audio_path}")

    if not X:
        print("❌ No audio data found. Training aborted.")
        return

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = SVC(kernel='linear', probability=True)
    model.fit(X, y_encoded)

    joblib.dump((model, le), save_path)
    print(f"✅ Audio model trained and saved to {save_path}")

def main():
    train_face_model()
    # train_audio_model()  # Uncomment when you have audio data

if __name__ == "__main__":
    main()
