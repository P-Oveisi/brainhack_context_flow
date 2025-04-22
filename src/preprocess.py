"""
Heart Rate Synchronization - Preprocessing Module

This module contains functions for preprocessing heart rate and audio data:
- Heart rate data cleaning and interpolation
- Audio data normalization and filtering
- Segmentation into analysis windows
"""

import numpy as np
import pandas as pd
import librosa
from scipy import signal, interpolate


def preprocess_heart_rate(hr_data, participant_id, sample_rate=1.0):
    """
    Preprocess heart rate data for a single participant.
    
    Parameters:
    -----------
    hr_data : pd.DataFrame
        DataFrame with 'timestamp' and 'heart_rate' columns
    participant_id : str
        ID of the participant
    sample_rate : float
        Expected samples per second
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed heart rate data
    """
    # Sort by timestamp
    hr_data = hr_data.sort_values('timestamp').reset_index(drop=True)
    
    # Remove outliers (outside physiological range)
    hr_data = hr_data[(hr_data['heart_rate'] >= 40) & (hr_data['heart_rate'] <= 200)]
    
    # Create evenly spaced time grid
    t_start = hr_data['timestamp'].min()
    t_end = hr_data['timestamp'].max()
    t_grid = np.arange(t_start, t_end, 1/sample_rate)
    
    # Interpolate missing values
    f_interp = interpolate.interp1d(
        hr_data['timestamp'].values,
        hr_data['heart_rate'].values,
        bounds_error=False,
        fill_value="extrapolate"
    )
    hr_interp = f_interp(t_grid)
    
    # Create new DataFrame with interpolated values
    hr_processed = pd.DataFrame({
        'timestamp': t_grid,
        'heart_rate': hr_interp,
        'participant_id': participant_id
    })
    
    return hr_processed


def preprocess_audio(audio_path, sr=44100):
    """
    Preprocess audio data.
    
    Parameters:
    -----------
    audio_path : str
        Path to audio file
    sr : int
        Target sample rate
        
    Returns:
    --------
    tuple
        (audio_data, sample_rate)
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    
    # Normalize amplitude
    y = librosa.util.normalize(y)
    
    # Apply minimal noise filtering (optional)
    # Simple high-pass filter to remove low-frequency noise
    b, a = signal.butter(3, 80/(sr/2), 'highpass')
    y_filtered = signal.filtfilt(b, a, y)
    
    return y_filtered, sr


def segment_data(hr_data1, hr_data2, audio_data, audio_sr, window_size=30, hop_size=30):
    """
    Segment heart rate and audio data into non-overlapping windows.
    
    Parameters:
    -----------
    hr_data1 : pd.DataFrame
        Preprocessed heart rate data for participant 1
    hr_data2 : pd.DataFrame
        Preprocessed heart rate data for participant 2
    audio_data : np.ndarray
        Audio samples
    audio_sr : int
        Audio sample rate
    window_size : int
        Window size in seconds
    hop_size : int
        Hop size in seconds (distance between consecutive windows)
        
    Returns:
    --------
    tuple
        (hr_windows1, hr_windows2, audio_windows, window_timestamps)
    """
    # Align participants data on the same time grid
    start_time = max(hr_data1['timestamp'].min(), hr_data2['timestamp'].min())
    end_time = min(hr_data1['timestamp'].max(), hr_data2['timestamp'].max())
    
    # Calculate number of windows
    total_duration = end_time - start_time
    n_windows = int(total_duration // hop_size)
    
    hr_windows1 = []
    hr_windows2 = []
    audio_windows = []
    window_timestamps = []
    
    # Extract windows
    for i in range(n_windows):
        window_start = start_time + i * hop_size
        window_end = window_start + window_size
        window_timestamps.append(window_start)
        
        # Extract heart rate data for the window
        hr_win1 = hr_data1[(hr_data1['timestamp'] >= window_start) & 
                           (hr_data1['timestamp'] < window_end)]
        hr_win2 = hr_data2[(hr_data2['timestamp'] >= window_start) & 
                           (hr_data2['timestamp'] < window_end)]
        
        # Extract audio data for the window
        audio_start_idx = int((window_start - start_time) * audio_sr)
        audio_end_idx = int((window_end - start_time) * audio_sr)
        
        # Check bounds
        if audio_end_idx <= len(audio_data):
            audio_win = audio_data[audio_start_idx:audio_end_idx]
            
            # Only add window if both heart rate series have enough data points
            if len(hr_win1) >= 10 and len(hr_win2) >= 10:
                hr_windows1.append(hr_win1)
                hr_windows2.append(hr_win2)
                audio_windows.append(audio_win)
    
    return hr_windows1, hr_windows2, audio_windows, window_timestamps


def align_multimodal_data(hr_files, audio_file, participant_ids):
    """
    Align heart rate data from multiple participants with audio data.
    
    Parameters:
    -----------
    hr_files : list
        List of heart rate data file paths
    audio_file : str
        Path to audio file
    participant_ids : list
        List of participant IDs
        
    Returns:
    --------
    tuple
        (aligned_hr_data, audio_data, audio_sr)
    """
    # Load heart rate data
    hr_data = []
    for i, (hr_file, participant_id) in enumerate(zip(hr_files, participant_ids)):
        df = pd.read_csv(hr_file)
        processed_df = preprocess_heart_rate(df, participant_id)
        hr_data.append(processed_df)
    
    # Load and process audio data
    audio_data, audio_sr = preprocess_audio(audio_file)
    
    return hr_data, audio_data, audio_sr 