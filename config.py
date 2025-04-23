"""
Heart Rate Synchronization - Configuration File

This file contains configuration parameters for the Heart Rate Synchronization project.
"""

# Data paths
DATA_CONFIG = {
    'raw_data_dir': 'data/raw/',
    'processed_data_dir': 'data/processed/',
    'features_dir': 'data/features/',
    'results_dir': 'data/results/'
}

# Data processing parameters
PROCESSING_CONFIG = {
    'heart_rate': {
        'sample_rate': 1.0,  # Hz (samples per second)
        'min_hr': 40,        # Minimum valid heart rate (BPM)
        'max_hr': 200        # Maximum valid heart rate (BPM)
    },
    'audio': {
        'sample_rate': 44100,  # Hz
        'normalize': True,
        'filter_lowfreq': True,
        'lowfreq_cutoff': 80  # Hz
    },
    'segmentation': {
        'window_size': 30,  # seconds
        'hop_size': 30,     # seconds (no overlap for 30)
        'min_samples_per_window': 10  # Minimum heart rate samples per window
    }
}

# Feature extraction parameters
FEATURE_CONFIG = {
    'heart_rate_features': [
        'hr_mean',
        'hr_std',
        'hr_trend',
        'hr_max_delta',
        'rmssd'
    ],
    'audio_features': [
        'audio_tempo',
        'audio_dom_freq',
        'audio_contrast', 
        'audio_complexity',
        'audio_dynamic_range'
    ],
    'audio': {
        'frame_length': 1024,
        'hop_length': 512
    }
}

# Synchronization analysis parameters
SYNC_CONFIG = {
    'correlation_window_size': 5,  # Number of windows for moving correlation
    'sync_threshold': 0.6,         # Threshold for binary synchronization (optional)
    'max_lag': 3                   # Maximum time lag for lag analysis (windows)
}

# Statistical analysis parameters
STATS_CONFIG = {
    'multiple_testing_correction': 'fdr_bh',  # Benjamini-Hochberg FDR correction
    'permutation_tests': 1000,               # Number of permutations
    'cross_validation_test_size': 0.3,       # Proportion of data for testing
    'correlation_thresholds': {              # Thresholds for interpreting correlations
        'weak': 0.3,
        'moderate': 0.5,
        'strong': 0.7
    }
}

# Visualization parameters
VIZ_CONFIG = {
    'timeline_figsize': (15, 12),
    'correlation_heatmap_figsize': (10, 8),
    'correlation_plot_figsize': (12, 8),
    'feature_profile_figsize': (15, 10),
    'lag_analysis_figsize': (10, 6),
    'sensitivity_figsize': (10, 8),
    'save_plots': True,
    'plot_formats': ['png', 'pdf']
}

# Sensitivity analysis parameters
SENSITIVITY_CONFIG = {
    'window_sizes': [3, 5, 7, 10],
    'correlation_thresholds': [0.4, 0.5, 0.6, 0.7, 0.8]
} 