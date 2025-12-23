import numpy as np
import pandas as pd
from scipy.stats import entropy, skew, kurtosis, iqr
from scipy.signal import welch
from nolds import sampen
import os

# Constants based on Report [cite: 663]
WINDOW_SIZE = 1.5  # seconds
STEP_SIZE = 0.5    # seconds
FS = 128           # Sampling frequency

def hjorth_params(data):
    """Calculate Hjorth Mobility and Complexity [cite: 538]"""
    first_deriv = np.diff(data)
    second_deriv = np.diff(data, 2)
    
    var_zero = np.var(data)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)
    
    mobility = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / mobility
    return mobility, complexity

def extract_features_from_window(window_data):
    features = []
    
    # Iterate over each channel
    for channel_data in window_data.T:
        # 1. Statistical Features [cite: 537]
        features.append(np.mean(channel_data))
        features.append(np.var(channel_data))
        features.append(skew(channel_data))
        features.append(kurtosis(channel_data))
        features.append(np.ptp(channel_data)) # Range/Peak-to-Peak
        features.append(iqr(channel_data))
        
        # 2. Complexity Features [cite: 538]
        features.append(entropy(np.abs(channel_data))) # Shannon Entropy (approx)
        # Note: Sampen is computationally expensive; using simpler logic for demo or library
        features.append(sampen(channel_data, emb_dim=2)) 
        h_mob, h_comp = hjorth_params(channel_data)
        features.append(h_mob)
        features.append(h_comp)
        
        # 3. Frequency Domain (Power Bands) [cite: 539]
        freqs, psd = welch(channel_data, fs=FS)
        # Delta (1-4), Theta (4-7), Alpha (8-12), Beta (13-30), Gamma (30-45)
        bands = [(1, 4), (4, 7), (8, 12), (13, 30), (30, 45)]
        for low, high in bands:
            # Trapz integration for power in band
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            features.append(np.trapz(psd[idx_band], freqs[idx_band]))

    return np.array(features)

def process_dataset(data_dir):
    """
    Iterates through gesture folders: 'open', 'close', 'index', 'victory'
    """
    X = []
    y = []
    labels = {'open': 0, 'close': 1, 'index': 2, 'victory': 3}
    
    for label_name, label_idx in labels.items():
        folder_path = os.path.join(data_dir, label_name)
        if not os.path.exists(folder_path): continue
        
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                # Load and preprocess
                processed_df = preprocess_eeg(os.path.join(folder_path, file), FS)
                data = processed_df.values
                
                # Sliding Window Segmentation [cite: 662]
                samples_per_window = int(WINDOW_SIZE * FS)
                step = int(STEP_SIZE * FS)
                
                for start in range(0, len(data) - samples_per_window, step):
                    window = data[start:start + samples_per_window]
                    # Data Augmentation (Gaussian Noise) [cite: 555]
                    noise = np.random.normal(0, 0.02 * np.std(window), window.shape)
                    aug_window = window + noise
                    
                    feats = extract_features_from_window(aug_window)
                    X.append(feats)
                    y.append(label_idx)
                    
    return np.array(X), np.array(y)