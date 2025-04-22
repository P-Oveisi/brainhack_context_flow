"""
Heart Rate Synchronization - Feature Extraction Module

This module contains functions for extracting features from heart rate and audio data:
- Heart rate features (mean, std, trend, etc.)
- Audio features (energy, ZCR, spectral centroid, etc.)
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
    
    # Feature 1: RMS Energy
    audio_energy = np.sqrt(np.mean(audio_window**2))
    
    # Feature 2: Zero-Crossing Rate
    audio_zcr = librosa.feature.zero_crossing_rate(audio_window, frame_length=2048, hop_length=512).mean()
    
    # Feature 3: Spectral Centroid
    audio_centroid = librosa.feature.spectral_centroid(y=audio_window, sr=sr).mean()
    
    # Feature 4: Spectral Flux
    # Compute the spectrogram
    stft = np.abs(librosa.stft(audio_window))
    # Compute the spectral flux as the sum of squared differences between consecutive frames
    diff = np.diff(stft, axis=1)
    audio_flux = np.mean(diff**2)
    
    # Feature 5: Voice Activity Ratio
    # Simple energy-based voice activity detection
    # Split into frames
    frame_length = 1024
    hop_length = 512
    frames = librosa.util.frame(audio_window, frame_length=frame_length, hop_length=hop_length)
    
    # Compute energy for each frame
    energies = np.sum(frames**2, axis=0)
    
    # Determine an adaptive energy threshold (could be refined)
    energy_threshold = 0.2 * np.max(energies)
    
    # Count frames above threshold
    voice_frames = np.sum(energies > energy_threshold)
    total_frames = len(energies)
    
    audio_voice_ratio = voice_frames / total_frames if total_frames > 0 else 0
    
    features = {
        'audio_energy': audio_energy,
        'audio_zcr': audio_zcr,
        'audio_centroid': audio_centroid,
        'audio_flux': audio_flux,
        'audio_voice_ratio': audio_voice_ratio
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