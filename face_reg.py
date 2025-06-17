import cv2
import numpy as np
import dlib
import os
from pathlib import Path

def find_model_file(filename):
    """Try to find dlib model files in common locations"""
    search_paths = [
        Path.cwd(),
        Path.cwd() / "models",
        Path.cwd() / "face_reg",
        Path("/usr/local/share/dlib/models"),
        Path.home() / "dlib_models",
    ]

    for path in search_paths:
        full_path = path / filename
        if full_path.exists():
            return str(full_path)

    raise FileNotFoundError(f"Could not find {filename}. Please download it from http://dlib.net/files/")

# Load Dlib models globally
try:
    shape_predictor_path = find_model_file("shape_predictor_68_face_landmarks.dat")
    face_rec_model_path = find_model_file("dlib_face_recognition_resnet_model_v1.dat")

    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(shape_predictor_path)
    face_encoder = dlib.face_recognition_model_v1(face_rec_model_path)

    print("✅ Dlib models loaded successfully")
except Exception as e:
    print(f"❌ Error loading dlib models: {e}")
    raise

def face_reg(image_path):
    """
    Detects faces in an image and returns list of (face_encoding, bounding_box).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray)
    if len(faces) == 0:
        return []

    encodings = []
    for face in faces:
        shape = shape_predictor(gray, face)
        encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))
        box = (face.left(), face.top(), face.width(), face.height())
        encodings.append((encoding, box))

    return encodings

def extract_face_features(image_path):
    """
    Extracts a single face encoding from an image.
    Returns: 128D encoding or None
    """
    results = face_reg(image_path)
    return results[0][0] if results else None
