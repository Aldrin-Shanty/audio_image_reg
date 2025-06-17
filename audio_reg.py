import librosa
import numpy as np
import os

def audio_reg(audio_path=None, sr=22050, duration=3):
    """
    Extracts MFCC features from a .wav file.
    If no path is provided, it can be extended to record from mic in future.
    Returns: 1D feature vector or None
    """
    if audio_path is None:
        print("❌ No audio path provided.")
        return None

    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return None

    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"⚠️ Error processing {audio_path}: {e}")
        return None
