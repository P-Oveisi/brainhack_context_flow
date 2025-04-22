"""
Heart Rate Synchronization - Context Classification Module

This module contains functions for classifying audio contexts:
- Defining binary context categories
- Creating multi-dimensional contexts
- Context-synchronization analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def classify_audio_contexts(features_df):
    """
    Classify audio contexts based on audio features.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with audio features
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with audio context classifications
    """
    # Make a copy to avoid modifying original
    context_df = features_df.copy()
    
    # 1. Loudness Context
    energy_threshold = features_df['audio_energy'].median()
    context_df['context_loud'] = (features_df['audio_energy'] >= energy_threshold).astype(int)
    
    # 2. Frequency Content Context
    centroid_threshold = features_df['audio_centroid'].median()
    context_df['context_high_freq'] = (features_df['audio_centroid'] >= centroid_threshold).astype(int)
    
    # 3. Stability Context
    flux_threshold = features_df['audio_flux'].median()
    context_df['context_changing'] = (features_df['audio_flux'] >= flux_threshold).astype(int)
    
    # 4. Voice Context
    voice_threshold = 0.3  # Can be adjusted based on data
    context_df['context_voice'] = (features_df['audio_voice_ratio'] >= voice_threshold).astype(int)
    
    # 5. Complexity Context
    zcr_threshold = features_df['audio_zcr'].median()
    context_df['context_complex'] = (features_df['audio_zcr'] >= zcr_threshold).astype(int)
    
    return context_df


def create_multidimensional_contexts(context_df):
    """
    Create compound contexts by combining binary categories.
    
    Parameters:
    -----------
    context_df : pd.DataFrame
        DataFrame with binary context classifications
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with multi-dimensional context classifications
    """
    # Make a copy to avoid modifying original
    multi_df = context_df.copy()
    
    # Define common context combinations
    # Example: "Quiet + Voice Present + Steady"
    multi_df['context_quiet_voice_steady'] = (
        (multi_df['context_loud'] == 0) & 
        (multi_df['context_voice'] == 1) & 
        (multi_df['context_changing'] == 0)
    ).astype(int)
    
    # "Loud + Voice Absent + Changing"
    multi_df['context_loud_novoice_changing'] = (
        (multi_df['context_loud'] == 1) & 
        (multi_df['context_voice'] == 0) & 
        (multi_df['context_changing'] == 1)
    ).astype(int)
    
    # "Quiet + Simple"
    multi_df['context_quiet_simple'] = (
        (multi_df['context_loud'] == 0) & 
        (multi_df['context_complex'] == 0)
    ).astype(int)
    
    # "Loud + Complex"
    multi_df['context_loud_complex'] = (
        (multi_df['context_loud'] == 1) & 
        (multi_df['context_complex'] == 1)
    ).astype(int)
    
    # "Voice + High Frequency"
    multi_df['context_voice_highfreq'] = (
        (multi_df['context_voice'] == 1) & 
        (multi_df['context_high_freq'] == 1)
    ).astype(int)
    
    return multi_df


def calculate_sync_probability_by_context(context_df, sync_series, context_col):
    """
    Calculate the probability of synchronization within each context.
    
    Parameters:
    -----------
    context_df : pd.DataFrame
        DataFrame with context classifications
    sync_series : pd.Series
        Binary synchronization series (1 = synchronized, 0 = not synchronized)
    context_col : str
        Name of the context column
        
    Returns:
    --------
    tuple
        (context_present_prob, context_absent_prob, odds_ratio, p_value)
    """
    if len(sync_series) == 0 or context_col not in context_df.columns:
        return None, None, None, None
    
    # Create a DataFrame with context and synchronization
    df = pd.DataFrame({
        'context': context_df[context_col],
        'sync': sync_series.reset_index(drop=True)
    })
    
    # Calculate probabilities
    # P(sync | context present)
    context_present = df[df['context'] == 1]
    if len(context_present) > 0:
        context_present_prob = context_present['sync'].mean()
    else:
        context_present_prob = np.nan
    
    # P(sync | context absent)
    context_absent = df[df['context'] == 0]
    if len(context_absent) > 0:
        context_absent_prob = context_absent['sync'].mean()
    else:
        context_absent_prob = np.nan
    
    # Create contingency table for chi-square test
    contingency = pd.crosstab(df['context'], df['sync'])
    
    # Perform chi-square test
    if contingency.shape == (2, 2):  # Ensure 2x2 table
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        
        # Calculate odds ratio
        odds_ratio = (contingency.iloc[1, 1] * contingency.iloc[0, 0]) / \
                     (contingency.iloc[1, 0] * contingency.iloc[0, 1]) \
                     if (contingency.iloc[1, 0] * contingency.iloc[0, 1]) > 0 else np.inf
    else:
        p_value = np.nan
        odds_ratio = np.nan
    
    return context_present_prob, context_absent_prob, odds_ratio, p_value


def analyze_context_synchronization(context_df, sync_results):
    """
    Analyze the relationship between audio contexts and heart rate synchronization.
    
    Parameters:
    -----------
    context_df : pd.DataFrame
        DataFrame with context classifications
    sync_results : dict
        Dictionary with synchronization results
        
    Returns:
    --------
    dict
        Dictionary with context-synchronization analysis results
    """
    # Get binary synchronization for each heart rate feature
    binary_sync = sync_results['binary_synchronization']
    
    # Identify context columns
    context_cols = [col for col in context_df.columns if col.startswith('context_')]
    
    # Analyze each combination of heart rate feature and context
    results = {}
    
    for feature, sync_series in binary_sync.items():
        feature_results = {}
        
        for context_col in context_cols:
            probs = calculate_sync_probability_by_context(context_df, sync_series, context_col)
            if probs[0] is not None:
                feature_results[context_col] = {
                    'sync_prob_context_present': probs[0],
                    'sync_prob_context_absent': probs[1],
                    'odds_ratio': probs[2],
                    'p_value': probs[3]
                }
        
        results[feature] = feature_results
    
    return results


def point_biserial_correlation(context_df, features_df):
    """
    Calculate point-biserial correlation between continuous audio features and binary synchronization.
    
    Parameters:
    -----------
    context_df : pd.DataFrame
        DataFrame with binary context and synchronization data
    features_df : pd.DataFrame
        DataFrame with continuous audio features
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with correlation results
    """
    # Identify context columns
    context_cols = [col for col in context_df.columns if col.startswith('context_')]
    
    # Identify synchronization columns
    sync_cols = [col for col in context_df.columns if col.startswith('sync_')]
    
    # Identify audio feature columns
    audio_cols = ['audio_energy', 'audio_zcr', 'audio_centroid', 'audio_flux', 'audio_voice_ratio']
    
    # Calculate correlations
    correlations = []
    
    for sync_col in sync_cols:
        for audio_col in audio_cols:
            # Point-biserial correlation is mathematically equivalent to Pearson correlation
            # when one variable is binary and the other is continuous
            r, p = stats.pearsonr(context_df[sync_col], features_df[audio_col])
            
            correlations.append({
                'sync_feature': sync_col,
                'audio_feature': audio_col,
                'correlation': r,
                'p_value': p
            })
    
    return pd.DataFrame(correlations) 