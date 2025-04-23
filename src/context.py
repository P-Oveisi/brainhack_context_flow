"""
Heart Rate Synchronization - Audio Feature Analysis Module

This module contains functions for analyzing audio features:
- Preparing audio features for analysis
- Correlating audio features with synchronization
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_audio_features(features_df):
    """
    Prepare audio features for analysis. This function simply passes through
    the continuous audio features without binarization.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with audio features
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with prepared audio features
    """
    # Make a copy to avoid modifying original
    audio_features_df = features_df.copy()
    
    # Standardize/normalize features if needed
    audio_cols = ['audio_tempo', 'audio_dom_freq', 'audio_contrast', 'audio_complexity', 'audio_dynamic_range']
    
    # Calculate z-scores for each feature (optional)
    for col in audio_cols:
        if col in audio_features_df.columns:
            mean = audio_features_df[col].mean()
            std = audio_features_df[col].std()
            if std > 0:  # Avoid division by zero
                audio_features_df[f'{col}_z'] = (audio_features_df[col] - mean) / std
    
    return audio_features_df


def calculate_feature_correlation_with_sync(features_df, sync_series, feature_col):
    """
    Calculate the correlation between an audio feature and synchronization.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with audio features
    sync_series : pd.Series
        Synchronization series (continuous measure)
    feature_col : str
        Name of the feature column
        
    Returns:
    --------
    tuple
        (correlation, p_value)
    """
    if len(sync_series) == 0 or feature_col not in features_df.columns:
        return None, None
    
    # Calculate Pearson correlation
    correlation, p_value = stats.pearsonr(
        features_df[feature_col].reset_index(drop=True),
        sync_series.reset_index(drop=True)
    )
    
    return correlation, p_value


def analyze_feature_synchronization(features_df, sync_results):
    """
    Analyze the relationship between audio features and heart rate synchronization.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with audio features
    sync_results : dict
        Dictionary with synchronization results
        
    Returns:
    --------
    dict
        Dictionary with feature-synchronization analysis results
    """
    # Get synchronization data for each heart rate feature
    correlations = sync_results.get('window_correlations', {})
    
    # Identify audio feature columns
    audio_cols = ['audio_tempo', 'audio_dom_freq', 'audio_contrast', 'audio_complexity', 'audio_dynamic_range']
    
    # Analyze each combination of heart rate feature and audio feature
    results = {}
    
    for hr_feature, sync_series in correlations.items():
        feature_results = {}
        
        for audio_col in audio_cols:
            corr, p = calculate_feature_correlation_with_sync(features_df, sync_series, audio_col)
            
            if corr is not None:
                feature_results[audio_col] = {
                    'correlation': corr,
                    'p_value': p
                }
        
        results[hr_feature] = feature_results
    
    return results


def correlation_analysis(features_df, sync_df):
    """
    Calculate correlation between continuous audio features and synchronization measures.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with continuous audio features
    sync_df : pd.DataFrame
        DataFrame with synchronization measures
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with correlation results
    """
    # Identify synchronization columns
    sync_cols = [col for col in sync_df.columns if col.startswith('corr_')]
    
    # Identify audio feature columns
    audio_cols = ['audio_tempo', 'audio_dom_freq', 'audio_contrast', 'audio_complexity', 'audio_dynamic_range']
    
    # Calculate correlations
    correlations = []
    
    for sync_col in sync_cols:
        for audio_col in audio_cols:
            # Calculate Pearson correlation
            r, p = stats.pearsonr(sync_df[sync_col], features_df[audio_col])
            
            correlations.append({
                'sync_feature': sync_col,
                'audio_feature': audio_col,
                'correlation': r,
                'p_value': p
            })
    
    return pd.DataFrame(correlations) 