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
        'audio_energy',
        'audio_zcr',
        'audio_centroid', 
        'audio_flux',
        'audio_voice_ratio'
    ],
    'audio': {
        'frame_length': 1024,
        'hop_length': 512,
        'voice_energy_threshold': 0.2  # Relative threshold for voice detection
    }
}

# Synchronization analysis parameters
SYNC_CONFIG = {
    'correlation_window_size': 5,  # Number of windows for moving correlation
    'sync_threshold': 0.6,         # Threshold for binary synchronization
    'max_lag': 3                   # Maximum time lag for lag analysis (windows)
}

# Context classification parameters
CONTEXT_CONFIG = {
    'voice_threshold': 0.3,  # Threshold for voice activity detection
    'binary_contexts': [
        'context_loud',
        'context_high_freq',
        'context_changing',
        'context_voice',
        'context_complex'
    ],
    'multi_contexts': [
        'context_quiet_voice_steady',
        'context_loud_novoice_changing',
        'context_quiet_simple',
        'context_loud_complex',
        'context_voice_highfreq'
    ]
}

# Statistical analysis parameters
STATS_CONFIG = {
    'multiple_testing_correction': 'fdr_bh',  # Benjamini-Hochberg FDR correction
    'permutation_tests': 1000,               # Number of permutations
    'cross_validation_test_size': 0.3,       # Proportion of data for testing
    'effect_size_thresholds': {
        'negligible': 0.2,
        'small': 0.5,
        'medium': 0.8
    }
}

# Visualization parameters
VIZ_CONFIG = {
    'timeline_figsize': (15, 12),
    'heatmap_figsize': (10, 8),
    'effect_sizes_figsize': (12, 8),
    'context_profile_figsize': (15, 10),
    'lag_analysis_figsize': (10, 6),
    'sensitivity_figsize': (10, 8),
    'save_plots': True,
    'plot_formats': ['png', 'pdf']
}

# Sensitivity analysis parameters
SENSITIVITY_CONFIG = {
    'window_sizes': [3, 5, 7, 10],
    'sync_thresholds': [0.4, 0.5, 0.6, 0.7, 0.8]
} 