"""
Heart Rate Synchronization - Feature Extraction Module

This module contains functions for extracting features from heart rate and audio data:
- Heart rate features (mean, std, trend, etc.)
- Audio features (tempo, dominant frequency, spectral contrast, etc.)
"""

import numpy as np
import pandas as pd
import librosa
from scipy import stats, signal
from sklearn.linear_model import LinearRegression


def extract_heart_rate_features(hr_window):
    """
    Extract heart rate features from a window of heart rate data.
    
    Parameters:
    -----------
    hr_window : pd.DataFrame
        DataFrame with 'timestamp' and 'heart_rate' columns
        
    Returns:
    --------
    dict
        Dictionary of extracted features
    """
    if len(hr_window) < 2:
        return None
    
    # Get heart rate values
    hr_values = hr_window['heart_rate'].values
    timestamps = hr_window['timestamp'].values
    
    # Feature 1: Mean Heart Rate
    hr_mean = np.mean(hr_values)
    
    # Feature 2: Heart Rate Standard Deviation
    hr_std = np.std(hr_values)
    
    # Feature 3: Heart Rate Trend (slope of linear regression)
    # Reshape for sklearn
    X = timestamps.reshape(-1, 1)
    y = hr_values
    model = LinearRegression().fit(X, y)
    hr_trend = model.coef_[0]
    
    # Feature 4: Maximum Heart Rate Change
    hr_diff = np.abs(np.diff(hr_values))
    hr_max_delta = np.max(hr_diff) if len(hr_diff) > 0 else 0
    
    # Feature 5: Approximation of RMSSD (HRV)
    # Convert heart rate (BPM) to RR intervals (ms)
    rr_intervals = 60000 / hr_values
    rr_diff = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(rr_diff**2)) if len(rr_diff) > 0 else 0
    
    features = {
        'hr_mean': hr_mean,
        'hr_std': hr_std,
        'hr_trend': hr_trend,
        'hr_max_delta': hr_max_delta,
        'rmssd': rmssd
    }
    
    return features


def extract_audio_features(audio_window, sr):
    """
    Extract audio features from a window of audio data.
    
    Parameters:
    -----------
    audio_window : np.ndarray
        Array of audio samples
    sr : int
        Sample rate
        
    Returns:
    --------
    dict
        Dictionary of extracted features
    """
    if len(audio_window) < sr // 10:  # Minimum 100ms of audio
        return None
    
    # Feature 1: Tempo/BPM
    # Use librosa's beat tracking to estimate tempo
    tempo, _ = librosa.beat.beat_track(y=audio_window, sr=sr)
    audio_tempo = tempo
    
    # Feature 2: Dominant Frequency
    # Compute the magnitude spectrum
    spec = np.abs(librosa.stft(audio_window))
    # Average over time
    mean_spec = np.mean(spec, axis=1)
    # Find the bin with the maximum energy
    max_bin = np.argmax(mean_spec)
    # Convert bin to frequency
    freqs = librosa.fft_frequencies(sr=sr)
    audio_dom_freq = freqs[max_bin]
    
    # Feature 3: Spectral Contrast
    # Compute the spectral contrast
    contrast = librosa.feature.spectral_contrast(y=audio_window, sr=sr)
    # Average over time and bands
    audio_contrast = np.mean(contrast)
    
    # Feature 4: Timeseries Information Complexity
    # We'll use a simplified estimate based on spectral entropy
    # Compute the power spectrum
    power_spec = np.abs(librosa.stft(audio_window))**2
    # Normalize it
    power_spec_norm = power_spec / np.sum(power_spec, axis=0, keepdims=True)
    # Compute entropy along frequency axis
    entropy = -np.sum(power_spec_norm * np.log2(power_spec_norm + 1e-10), axis=0)
    # Average over time
    audio_complexity = np.mean(entropy)
    
    # Feature 5: Dynamic Range
    # Compute the RMS energy in short frames
    frame_length = 1024
    hop_length = 512
    frames = librosa.util.frame(audio_window, frame_length=frame_length, hop_length=hop_length)
    frame_energies = np.sqrt(np.mean(frames**2, axis=0))
    # Get dynamic range as ratio between max and min energy (in dB)
    if len(frame_energies) > 0 and np.min(frame_energies) > 0:
        audio_dynamic_range = 20 * np.log10(np.max(frame_energies) / np.min(frame_energies))
    else:
        audio_dynamic_range = 0
    
    features = {
        'audio_tempo': audio_tempo,
        'audio_dom_freq': audio_dom_freq,
        'audio_contrast': audio_contrast,
        'audio_complexity': audio_complexity,
        'audio_dynamic_range': audio_dynamic_range
    }
    
    return features


def extract_all_features(hr_windows1, hr_windows2, audio_windows, audio_sr, window_timestamps):
    """
    Extract features from all data windows.
    
    Parameters:
    -----------
    hr_windows1 : list
        List of heart rate DataFrames for participant 1
    hr_windows2 : list
        List of heart rate DataFrames for participant 2
    audio_windows : list
        List of audio data arrays
    audio_sr : int
        Audio sample rate
    window_timestamps : list
        List of window start timestamps
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with all extracted features
    """
    all_features = []
    
    for i in range(len(hr_windows1)):
        # Extract features for current window
        hr_features1 = extract_heart_rate_features(hr_windows1[i])
        hr_features2 = extract_heart_rate_features(hr_windows2[i])
        audio_feats = extract_audio_features(audio_windows[i], audio_sr)
        
        # Skip if any feature extraction failed
        if hr_features1 is None or hr_features2 is None or audio_feats is None:
            continue
        
        # Combine all features
        window_features = {
            'window_start': window_timestamps[i],
            'window_end': window_timestamps[i] + 30  # Assuming 30s windows
        }
        
        # Add participant 1 features with prefix
        for key, value in hr_features1.items():
            window_features[f'p1_{key}'] = value
            
        # Add participant 2 features with prefix
        for key, value in hr_features2.items():
            window_features[f'p2_{key}'] = value
            
        # Add audio features
        window_features.update(audio_feats)
        
        all_features.append(window_features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    return features_df 