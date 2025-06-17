import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import joblib
import tempfile
import sounddevice as sd
import soundfile as sf
from face_reg import face_reg
from audio_reg import audio_reg
from model_utils import load_models
import time # Constants
NUM_FRAMES = 10
AUDIO_DURATION = 3  # seconds

# Load models
face_model, audio_model = load_models()

# Global capture object
cap = None

def record_video_frames(num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(0)
    images = []

    if not cap.isOpened():
        messagebox.showerror("Error", "❌ Cannot open webcam")
        return []

    print("📸 Capturing images from webcam...")
    count = 0
    while count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        images.append(frame)
        count += 1
        time.sleep(0.2)  # Sleep for 200ms instead of using cv2.waitKey()

    cap.release()
    return images

def predict_from_image():
    if face_model is None:
        messagebox.showerror("Error", "Face model not found")
        return

    frames = record_video_frames()
    predictions = []

    for idx, frame in enumerate(frames):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
            cv2.imwrite(tmp_img.name, frame)
            encoding_data = face_reg(tmp_img.name)
            os.remove(tmp_img.name)

        if encoding_data:
            encoding = encoding_data[0][0]
            pred = face_model.predict([encoding])[0]
            predictions.append(pred)

    if not predictions:
        messagebox.showwarning("Warning", "😶 No face detected in any frame")
        return

    final_prediction = max(set(predictions), key=predictions.count)
    messagebox.showinfo("Result", f"🧠 Face Prediction: {final_prediction}")

def record_audio(duration=AUDIO_DURATION):
    fs = 22050
    print("🎙️ Recording audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        sf.write(tmp_audio.name, audio, fs)
        return tmp_audio.name

def predict_from_audio():
    if audio_model is None:
        messagebox.showwarning("Missing Model", "Audio model not available (placeholder used)")
        messagebox.showinfo("Result", "🎤 Audio Prediction: Placeholder_User")
        return

    audio_path = record_audio()
    features = audio_reg(audio_path)
    os.remove(audio_path)

    if features is None:
        messagebox.showerror("Error", "Could not process audio")
        return

    pred = audio_model.predict([features])[0]
    messagebox.showinfo("Result", f"🎤 Audio Prediction: {pred}")

# Optional: Show webcam preview in Tkinter
def start_camera_preview():
    def update_frame():
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            cam_label.imgtk = imgtk
            cam_label.config(image=imgtk)
        cam_label.after(10, update_frame)

    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "❌ Cannot open webcam")
        return

    top = tk.Toplevel(root)
    top.title("📷 Webcam Preview")
    cam_label = tk.Label(top)
    cam_label.pack()
    update_frame()

# GUI
root = tk.Tk()
root.title("Multimodal Recognition")
root.geometry("300x230")

tk.Label(root, text="Select Recognition Mode:", font=("Helvetica", 14)).pack(pady=20)

tk.Button(root, text="📷 Show Webcam Preview", command=start_camera_preview, width=25).pack(pady=5)
tk.Button(root, text="🧠 Predict from Image", command=predict_from_image, width=25).pack(pady=10)
tk.Button(root, text="🎤 Predict from Audio", command=predict_from_audio, width=25).pack(pady=10)

root.mainloop()
