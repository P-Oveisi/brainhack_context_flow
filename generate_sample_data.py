"""
Generate Sample Data for Heart Rate Synchronization Analysis

This script generates synthetic heart rate data for two participants and a sample audio file
to allow testing of the analysis pipeline without real data.
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
import argparse
from datetime import datetime, timedelta


def generate_heart_rate_data(duration, sample_rate=1.0, base_hr=70, sync_windows=None):
    """
    Generate synthetic heart rate data.
    
    Parameters:
    -----------
    duration : int
        Duration in seconds
    sample_rate : float
        Number of samples per second
    base_hr : int
        Base heart rate in BPM
    sync_windows : list
        List of (start, end, strength) tuples defining synchronization windows
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with timestamp and heart rate columns
    """
    # Generate timestamps
    timestamps = np.arange(0, duration, 1/sample_rate)
    
    # Generate base heart rate with natural variability
    heart_rate = base_hr + 5 * np.sin(2 * np.pi * 0.005 * timestamps)  # Slow oscillation
    
    # Add random noise
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, 1, len(timestamps))
    smoothed_noise = np.convolve(noise, np.ones(5)/5, mode='same')  # Smooth noise
    heart_rate += smoothed_noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'heart_rate': heart_rate
    })
    
    return df


def generate_synchronized_heart_rates(duration, sample_rate=1.0, sync_windows=None):
    """
    Generate two heart rate time series with synchronization in specified windows.
    
    Parameters:
    -----------
    duration : int
        Duration in seconds
    sample_rate : float
        Number of samples per second
    sync_windows : list
        List of (start, end, strength) tuples defining synchronization windows
        
    Returns:
    --------
    tuple
        (hr_data1, hr_data2) - DataFrames for two participants
    """
    # Generate base data for participant 1
    hr_data1 = generate_heart_rate_data(duration, sample_rate, base_hr=70)
    
    # Generate base data for participant 2 (slightly different base rate)
    hr_data2 = generate_heart_rate_data(duration, sample_rate, base_hr=75)
    
    # Apply synchronization in specified windows
    if sync_windows:
        for start, end, strength in sync_windows:
            # Get indices for window
            idx_start = int(start * sample_rate)
            idx_end = int(end * sample_rate)
            
            if idx_end > len(hr_data1):
                idx_end = len(hr_data1)
            
            # Calculate weights based on how far into the window (for smooth transition)
            window_len = idx_end - idx_start
            transition = np.ones(window_len)
            
            # Apply transition at start and end (10% of window length)
            transition_len = max(int(window_len * 0.1), 1)
            transition[:transition_len] = np.linspace(0, 1, transition_len)
            transition[-transition_len:] = np.linspace(1, 0, transition_len)
            
            # Get slice of data
            p1_hr = hr_data1['heart_rate'].values[idx_start:idx_end].copy()
            p2_hr = hr_data2['heart_rate'].values[idx_start:idx_end].copy()
            
            # Calculate target (weighted average of both heart rates)
            target = (p1_hr + p2_hr) / 2
            
            # Apply synchronization with strength factor
            p1_hr_new = p1_hr * (1 - strength * transition) + target * strength * transition
            p2_hr_new = p2_hr * (1 - strength * transition) + target * strength * transition
            
            # Update heart rates
            hr_data1.loc[idx_start:idx_end-1, 'heart_rate'] = p1_hr_new
            hr_data2.loc[idx_start:idx_end-1, 'heart_rate'] = p2_hr_new
    
    return hr_data1, hr_data2


def generate_audio_data(duration, sr=44100, sync_windows=None):
    """
    Generate synthetic audio data with different contexts.
    
    Parameters:
    -----------
    duration : int
        Duration in seconds
    sr : int
        Sample rate in Hz
    sync_windows : list
        List of (start, end, strength) tuples defining synchronization windows
        
    Returns:
    --------
    tuple
        (audio_data, sample_rate)
    """
    # Calculate total number of samples
    n_samples = int(duration * sr)
    
    # Generate base background noise (brown noise)
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, 1, n_samples)
    # Convert to brown noise
    brown_noise = np.cumsum(noise)
    # Normalize
    brown_noise = 0.15 * brown_noise / np.max(np.abs(brown_noise))
    
    # Initialize audio data with background noise
    audio_data = brown_noise
    
    # Define different audio contexts to apply in different segments
    contexts = []
    
    # If sync_windows provided, align contexts with sync windows
    if sync_windows:
        for start, end, strength in sync_windows:
            # Randomly choose context type for synchronization window
            context_type = np.random.choice(['voice', 'music', 'silence'])
            contexts.append((start, end, context_type))
            
            # Add some non-sync windows with random contexts
            if end < duration - 30:
                non_sync_start = end + 10  # 10s after sync window
                non_sync_end = min(non_sync_start + 30, duration)
                non_sync_type = np.random.choice(['noise', 'beeps'])
                contexts.append((non_sync_start, non_sync_end, non_sync_type))
    else:
        # If no sync_windows, create random contexts
        segment_length = 30  # 30-second segments
        for start in range(0, duration, segment_length):
            end = min(start + segment_length, duration)
            context_type = np.random.choice(['voice', 'music', 'silence', 'noise', 'beeps'])
            contexts.append((start, end, context_type))
    
    # Apply contexts
    for start, end, context_type in contexts:
        # Calculate sample indices
        idx_start = int(start * sr)
        idx_end = int(end * sr)
        
        # Apply context
        segment_len = idx_end - idx_start
        
        if context_type == 'voice':
            # Simulate voice with formants
            formants = np.zeros(segment_len)
            # Add multiple frequencies for formants
            for formant_freq in [150, 500, 1500, 3000]:
                t = np.arange(segment_len) / sr
                formant = 0.1 * np.sin(2 * np.pi * formant_freq * t)
                # Add some amplitude modulation
                am = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # 3 Hz modulation
                formants += formant * am
            
            # Apply smoothing for more natural sound
            formants = np.convolve(formants, np.ones(1000)/1000, mode='same')
            
            # Add to audio data (mix with background)
            audio_data[idx_start:idx_end] = 0.7 * formants + 0.3 * audio_data[idx_start:idx_end]
            
        elif context_type == 'music':
            # Simulate music with multiple frequencies
            music = np.zeros(segment_len)
            # Add multiple notes
            for note_freq in [261.63, 329.63, 392.00]:  # C4, E4, G4 (C major chord)
                t = np.arange(segment_len) / sr
                note = 0.1 * np.sin(2 * np.pi * note_freq * t)
                music += note
            
            # Add rhythm
            rhythm = np.zeros(segment_len)
            beat_len = int(sr * 0.5)  # 0.5s beat
            for i in range(0, segment_len, beat_len):
                if i + beat_len <= segment_len:
                    # Exponentially decaying beat
                    decay = np.exp(-5 * np.arange(beat_len) / beat_len)
                    rhythm[i:i+beat_len] = 0.2 * decay
            
            # Combine rhythm and notes
            music = music + rhythm
            
            # Add to audio data (mix with background)
            audio_data[idx_start:idx_end] = 0.8 * music + 0.2 * audio_data[idx_start:idx_end]
            
        elif context_type == 'silence':
            # Reduce volume significantly
            audio_data[idx_start:idx_end] *= 0.1
            
        elif context_type == 'noise':
            # White noise
            white_noise = 0.3 * np.random.normal(0, 1, segment_len)
            audio_data[idx_start:idx_end] = white_noise
            
        elif context_type == 'beeps':
            # Generate beeps
            beeps = np.zeros(segment_len)
            beep_interval = int(sr * 2)  # 2-second interval
            beep_len = int(sr * 0.2)  # 0.2-second beep
            
            for i in range(0, segment_len, beep_interval):
                if i + beep_len <= segment_len:
                    t = np.arange(beep_len) / sr
                    beep = 0.3 * np.sin(2 * np.pi * 1000 * t)  # 1000 Hz tone
                    beeps[i:i+beep_len] = beep
            
            # Add to audio data (replace background)
            audio_data[idx_start:idx_end] = beeps
    
    # Normalize final audio
    audio_data = librosa.util.normalize(audio_data)
    
    return audio_data, sr


def save_data(hr_data1, hr_data2, audio_data, sr, output_dir):
    """
    Save generated data to files.
    
    Parameters:
    -----------
    hr_data1 : pd.DataFrame
        Heart rate data for participant 1
    hr_data2 : pd.DataFrame
        Heart rate data for participant 2
    audio_data : np.ndarray
        Audio data
    sr : int
        Audio sample rate
    output_dir : str
        Output directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save heart rate data
    hr_data1.to_csv(os.path.join(output_dir, 'participant1_hr.csv'), index=False)
    hr_data2.to_csv(os.path.join(output_dir, 'participant2_hr.csv'), index=False)
    
    # Save audio data
    sf.write(os.path.join(output_dir, 'session_audio.wav'), audio_data, sr)
    
    print(f"Data saved to {output_dir}")


def visualize_data(hr_data1, hr_data2, audio_data, sr, sync_windows, output_dir):
    """
    Visualize generated data.
    
    Parameters:
    -----------
    hr_data1 : pd.DataFrame
        Heart rate data for participant 1
    hr_data2 : pd.DataFrame
        Heart rate data for participant 2
    audio_data : np.ndarray
        Audio data
    sr : int
        Audio sample rate
    sync_windows : list
        List of synchronization windows
    output_dir : str
        Output directory
    """
    plt.figure(figsize=(12, 8))
    
    # Plot heart rates
    plt.subplot(2, 1, 1)
    plt.plot(hr_data1['timestamp'], hr_data1['heart_rate'], label='Participant 1')
    plt.plot(hr_data2['timestamp'], hr_data2['heart_rate'], label='Participant 2')
    
    # Highlight synchronization windows
    if sync_windows:
        for start, end, strength in sync_windows:
            plt.axvspan(start, end, alpha=0.2, color='green', label=f'Sync ({strength:.1f})' if not plt.gca().get_legend_handles_labels()[1].count(f'Sync ({strength:.1f})') else "")
    
    plt.xlabel('Time (s)')
    plt.ylabel('Heart Rate (BPM)')
    plt.title('Synthetic Heart Rate Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot audio waveform
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(audio_data, sr=sr)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Synthetic Audio Data')
    
    # Highlight synchronization windows
    if sync_windows:
        for start, end, strength in sync_windows:
            plt.axvspan(start, end, alpha=0.2, color='green')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'sample_data_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate sample data for heart rate synchronization analysis.')
    
    parser.add_argument('--duration', type=int, default=300, help='Duration of data in seconds')
    parser.add_argument('--hr-sample-rate', type=float, default=1.0, help='Heart rate sample rate in Hz')
    parser.add_argument('--audio-sample-rate', type=int, default=44100, help='Audio sample rate in Hz')
    parser.add_argument('--output-dir', default='data/raw', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize generated data')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print(f"Generating sample data ({args.duration} seconds)...")
    
    # Define synchronization windows (start, end, strength)
    # Strength is between 0 (no sync) and 1 (perfect sync)
    sync_windows = [
        (30, 60, 0.7),    # Strong sync from 30s to 60s
        (120, 150, 0.5),  # Medium sync from 120s to 150s
        (210, 240, 0.9),  # Very strong sync from 210s to 240s
    ]
    
    # Generate heart rate data
    hr_data1, hr_data2 = generate_synchronized_heart_rates(
        args.duration, 
        sample_rate=args.hr_sample_rate,
        sync_windows=sync_windows
    )
    
    # Generate audio data
    audio_data, sr = generate_audio_data(
        args.duration, 
        sr=args.audio_sample_rate,
        sync_windows=sync_windows
    )
    
    # Save data
    save_data(hr_data1, hr_data2, audio_data, sr, args.output_dir)
    
    # Visualize data if requested
    if args.visualize:
        visualize_data(hr_data1, hr_data2, audio_data, sr, sync_windows, args.output_dir)
        print(f"Visualization saved to {os.path.join(args.output_dir, 'sample_data_plot.png')}")


if __name__ == "__main__":
    main() 