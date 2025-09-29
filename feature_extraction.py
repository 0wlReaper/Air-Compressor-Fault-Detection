import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm  # progress bar

def extract_features(file_path):
    """
    Extracts audio features from a given file.
    Returns a 1D numpy array of features.
    """
    y, sr = librosa.load(file_path, mono=True, sr=None)

    # --- Time domain features ---
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

    # --- Frequency domain features ---
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    # --- MFCCs ---
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

    # Combine all features
    features = [rms, zcr, centroid, bandwidth, rolloff, flatness] + list(mfccs)
    return np.array(features)


def build_feature_table(dataset_path):
    """
    Loops through dataset folders and builds a feature table.
    Each subfolder = class label.
    Returns a pandas DataFrame.
    """
    data = []
    labels = []

    # Collect all audio file paths first
    all_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                all_files.append(os.path.join(root, file))

    # Loop with progress bar
    for file_path in tqdm(all_files, desc="Extracting features", unit="file"):
        label = os.path.basename(os.path.dirname(file_path))  # folder name as label
        try:
            feats = extract_features(file_path)
            data.append(feats)
            labels.append(label)
        except Exception as e:
            print(f"⚠️ Error processing {file_path}: {e}")

    # Create dataframe
    feature_names = (
        ["RMS", "ZCR", "Centroid", "Bandwidth", "Rolloff", "Flatness"]
        + [f"MFCC{i}" for i in range(1, 14)]
    )
    df = pd.DataFrame(data, columns=feature_names)
    df["Label"] = labels

    return df


if __name__ == "__main__":
    dataset_path = r"C:\Users\sivad\Desktop\Self tried projects\Fault detection\data\AirCompressorDataset"
    df = build_feature_table(dataset_path)

    os.makedirs("features", exist_ok=True)
    df.to_csv(os.path.join("features", "features.csv"), index=False)
    print("✅ Features saved to features/features.csv")
