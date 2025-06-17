# model_utils.py
import joblib

def load_models():
    """Load the trained SVM models"""
    try:
        face_model = joblib.load("face_svm_model.pkl")
    except FileNotFoundError:
        face_model = None
        print("⚠️ Face model not found")

    try:
        audio_model = joblib.load("audio_svm_model.pkl")
    except FileNotFoundError:
        audio_model = None
        print("⚠️ Audio model not found")

    return face_model, audio_model
