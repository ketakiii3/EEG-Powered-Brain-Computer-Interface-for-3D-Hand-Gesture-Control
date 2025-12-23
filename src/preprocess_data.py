import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.impute import SimpleImputer

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def preprocess_eeg(csv_path, fs=128):
    """
    Loads raw EEG, interpolates missing values, and applies bandpass filter.
    """
    # Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return None

    # Select only EEG channels (adjust column names based on Emotiv export)
    eeg_cols = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    
    # Check if columns exist, otherwise try lowercase
    if not set(eeg_cols).issubset(df.columns):
        eeg_cols = [c.lower() for c in eeg_cols]
    
    raw_data = df[eeg_cols].values

    # 1. Linear Interpolation for missing data [cite: 535]
    imputer = SimpleImputer(strategy='mean') # or interpolation
    raw_data = imputer.fit_transform(raw_data)

    # 2. Bandpass Filter (1-45 Hz) [cite: 534]
    filtered_data = np.zeros_like(raw_data)
    for i in range(raw_data.shape[1]):
        filtered_data[:, i] = butter_bandpass_filter(raw_data[:, i], 1.0, 45.0, fs, order=5)
        
    return pd.DataFrame(filtered_data, columns=eeg_cols)