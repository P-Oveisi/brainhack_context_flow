"""
Heart Rate Synchronization - Synchronization Analysis Module

This module contains functions for analyzing synchronization between heart rate features:
- Correlation analysis
- Binary and weighted synchronization measures
- Time lag analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import correlate


def calculate_pearson_correlation(features_df, hr_features):
    """
    Calculate Pearson correlation between participants for each heart rate feature.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with features for both participants
    hr_features : list
        List of heart rate feature names (without p1/p2 prefix)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with correlation coefficients for each feature and window
    """
    correlations = {}
    
    for feature in hr_features:
        p1_feature = f'p1_{feature}'
        p2_feature = f'p2_{feature}'
        
        # Check if features exist in DataFrame
        if p1_feature not in features_df.columns or p2_feature not in features_df.columns:
            continue
        
        # Calculate correlation
        r, p = stats.pearsonr(features_df[p1_feature], features_df[p2_feature])
        correlations[f'corr_{feature}'] = r
        correlations[f'p_value_{feature}'] = p
    
    return pd.DataFrame([correlations])


def calculate_windowwise_correlation(features_df, hr_features, window_size=5):
    """
    Calculate Pearson correlation for each window between participants.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with features for both participants
    hr_features : list
        List of heart rate feature names (without p1/p2 prefix)
    window_size : int
        Number of consecutive windows to use for correlation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with window-wise correlation coefficients
    """
    window_corrs = []
    
    # Skip if not enough windows
    if len(features_df) < window_size:
        return pd.DataFrame()
    
    for i in range(len(features_df) - window_size + 1):
        window_df = features_df.iloc[i:i+window_size]
        window_corr = {}
        window_corr['window_start'] = window_df['window_start'].iloc[0]
        window_corr['window_end'] = window_df['window_end'].iloc[-1]
        
        for feature in hr_features:
            p1_feature = f'p1_{feature}'
            p2_feature = f'p2_{feature}'
            
            # Check if features exist in DataFrame
            if p1_feature not in window_df.columns or p2_feature not in window_df.columns:
                continue
                
            # Calculate correlation within this window
            if len(window_df) >= 3:  # Minimum 3 points for meaningful correlation
                r, p = stats.pearsonr(window_df[p1_feature], window_df[p2_feature])
                window_corr[f'corr_{feature}'] = r
                window_corr[f'p_value_{feature}'] = p
            else:
                window_corr[f'corr_{feature}'] = np.nan
                window_corr[f'p_value_{feature}'] = np.nan
                
        window_corrs.append(window_corr)
    
    return pd.DataFrame(window_corrs)


def binary_synchronization(corr_df, feature, threshold=0.6):
    """
    Convert correlation coefficients to binary synchronization indicators.
    
    Parameters:
    -----------
    corr_df : pd.DataFrame
        DataFrame with correlation coefficients
    feature : str
        Heart rate feature name (without corr_ prefix)
    threshold : float
        Correlation threshold for synchronization
        
    Returns:
    --------
    pd.Series
        Binary series (1 = synchronized, 0 = not synchronized)
    """
    corr_col = f'corr_{feature}'
    if corr_col not in corr_df.columns:
        return pd.Series([])
    
    return (corr_df[corr_col] > threshold).astype(int)


def calculate_time_lag_correlation(features_df, hr_feature, max_lag=3):
    """
    Calculate time-lagged correlation for a heart rate feature.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with features for both participants
    hr_feature : str
        Heart rate feature name (without p1/p2 prefix)
    max_lag : int
        Maximum number of windows to lag
        
    Returns:
    --------
    tuple
        (optimum_lag, max_correlation)
    """
    p1_feature = f'p1_{hr_feature}'
    p2_feature = f'p2_{hr_feature}'
    
    # Check if features exist in DataFrame
    if p1_feature not in features_df.columns or p2_feature not in features_df.columns:
        return None, None
    
    # Get the feature series
    p1_series = features_df[p1_feature].values
    p2_series = features_df[p2_feature].values
    
    # Calculate cross-correlation
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    
    for lag in lags:
        if lag < 0:
            # Shift p1 backward relative to p2
            corr = np.corrcoef(p1_series[-lag:], p2_series[:lag])[0, 1]
        elif lag > 0:
            # Shift p1 forward relative to p2
            corr = np.corrcoef(p1_series[:-lag], p2_series[lag:])[0, 1]
        else:
            # No shift
            corr = np.corrcoef(p1_series, p2_series)[0, 1]
        
        correlations.append(corr)
    
    # Find lag with maximum correlation
    max_corr_idx = np.argmax(correlations)
    opt_lag = lags[max_corr_idx]
    max_corr = correlations[max_corr_idx]
    
    return opt_lag, max_corr


def calculate_feature_correlation(feature1, feature2):
    """
    Calculate correlation between two heart rate features.
    
    Parameters:
    -----------
    feature1 : pd.Series
        Heart rate feature values for participant 1
    feature2 : pd.Series
        Heart rate feature values for participant 2
        
    Returns:
    --------
    float
        Correlation coefficient
    """
    # Calculate Pearson correlation
    if len(feature1) != len(feature2):
        raise ValueError("Feature series must have the same length")
        
    if len(feature1) < 2 or len(feature2) < 2:
        return np.nan
        
    correlation = np.corrcoef(feature1, feature2)[0, 1]
    
    return correlation


def calculate_synchronization(features_df):
    """
    Calculate synchronization between heart rate features of two participants.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with heart rate features for both participants
        
    Returns:
    --------
    dict
        Dictionary with synchronization results
    """
    # Get available heart rate features
    p1_features = [col for col in features_df.columns if col.startswith('p1_')]
    p2_features = [col for col in features_df.columns if col.startswith('p2_')]
    
    # Extract features without prefix
    hr_features = [f.replace('p1_', '') for f in p1_features]
    
    # Calculate window-by-window correlation
    window_correlations = pd.DataFrame(index=features_df['window_start'])
    
    for feature in hr_features:
        p1_feature = f'p1_{feature}'
        p2_feature = f'p2_{feature}'
        
        if p1_feature in features_df.columns and p2_feature in features_df.columns:
            window_correlations[feature] = calculate_feature_correlation(
                features_df[p1_feature],
                features_df[p2_feature]
            )
    
    # Calculate time-lag correlations
    lag_results = {}
    for feature in hr_features:
        p1_feature = f'p1_{feature}'
        p2_feature = f'p2_{feature}'
        
        if p1_feature in features_df.columns and p2_feature in features_df.columns:
            lag_results[feature] = calculate_lag_correlation(
                features_df[p1_feature],
                features_df[p2_feature],
                max_lag=10  # Maximum lag in samples
            )
    
    return {
        'window_correlations': window_correlations,
        'lag_results': lag_results
    } 