"""
Heart Rate Synchronization - Analysis Module

This module contains functions for statistical analysis:
- Analyzing context-synchronization relationships
- Statistical testing
- Result summarization
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests


def compute_effect_sizes(context_sync_results):
    """
    Compute effect sizes for context-synchronization relationships.
    
    Parameters:
    -----------
    context_sync_results : dict
        Dictionary with context-synchronization analysis results
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with effect sizes and confidence intervals
    """
    effect_sizes = []
    
    for feature, contexts in context_sync_results.items():
        for context, stats in contexts.items():
            # Extract statistics
            odds_ratio = stats.get('odds_ratio', np.nan)
            p_value = stats.get('p_value', np.nan)
            
            # Calculate Cohen's h (effect size for proportions)
            p1 = stats.get('sync_prob_context_present', np.nan)
            p2 = stats.get('sync_prob_context_absent', np.nan)
            
            if not np.isnan(p1) and not np.isnan(p2):
                # Cohen's h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
                h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
            else:
                h = np.nan
                
            # Categorize effect size
            if np.isnan(h):
                effect_category = 'Unknown'
            elif abs(h) < 0.2:
                effect_category = 'Negligible'
            elif abs(h) < 0.5:
                effect_category = 'Small'
            elif abs(h) < 0.8:
                effect_category = 'Medium'
            else:
                effect_category = 'Large'
                
            effect_sizes.append({
                'feature': feature,
                'context': context,
                'odds_ratio': odds_ratio,
                'p_value': p_value,
                'effect_size': h,
                'effect_category': effect_category
            })
    
    return pd.DataFrame(effect_sizes)


def apply_multiple_testing_correction(effect_sizes_df, method='fdr_bh'):
    """
    Apply multiple testing correction to p-values.
    
    Parameters:
    -----------
    effect_sizes_df : pd.DataFrame
        DataFrame with effect sizes and p-values
    method : str
        Multiple testing correction method
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with corrected p-values
    """
    # Extract p-values
    p_values = effect_sizes_df['p_value'].values
    
    # Apply correction
    reject, p_corrected, _, _ = multipletests(p_values, method=method)
    
    # Add to DataFrame
    df_corrected = effect_sizes_df.copy()
    df_corrected['p_corrected'] = p_corrected
    df_corrected['significant'] = reject
    
    return df_corrected


def create_context_profile(features_df, context_cols):
    """
    Create a profile of audio contexts over time.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with features and context classifications
    context_cols : list
        List of context column names
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with context profiles over time
    """
    # Extract time and context columns
    time_cols = ['window_start', 'window_end']
    profile_df = features_df[time_cols + context_cols].copy()
    
    # Add window index
    profile_df['window_idx'] = range(len(profile_df))
    
    return profile_df


def summarize_significant_contexts(corrected_results):
    """
    Summarize significant context-synchronization relationships.
    
    Parameters:
    -----------
    corrected_results : pd.DataFrame
        DataFrame with corrected p-values and significance flags
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with significant results sorted by effect size
    """
    # Filter for significant results
    sig_results = corrected_results[corrected_results['significant']].copy()
    
    # Sort by absolute effect size
    sig_results['abs_effect'] = np.abs(sig_results['effect_size'])
    sig_results = sig_results.sort_values('abs_effect', ascending=False)
    
    return sig_results


def permutation_test(context_df, sync_series, context_col, n_permutations=1000):
    """
    Perform permutation test to validate context-synchronization relationships.
    
    Parameters:
    -----------
    context_df : pd.DataFrame
        DataFrame with context classifications
    sync_series : pd.Series
        Binary synchronization series
    context_col : str
        Name of the context column
    n_permutations : int
        Number of permutations to perform
        
    Returns:
    --------
    tuple
        (observed_diff, p_value)
    """
    # Get original difference in synchronization probabilities
    df = pd.DataFrame({
        'context': context_df[context_col],
        'sync': sync_series.reset_index(drop=True)
    })
    
    context_present = df[df['context'] == 1]['sync'].mean()
    context_absent = df[df['context'] == 0]['sync'].mean()
    observed_diff = context_present - context_absent
    
    # Perform permutation test
    permuted_diffs = []
    for _ in range(n_permutations):
        # Shuffle synchronization labels
        shuffled_sync = np.random.permutation(df['sync'].values)
        
        # Calculate difference with shuffled data
        present_mean = shuffled_sync[df['context'] == 1].mean()
        absent_mean = shuffled_sync[df['context'] == 0].mean()
        permuted_diffs.append(present_mean - absent_mean)
    
    # Calculate p-value
    p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
    
    return observed_diff, p_value


def perform_cross_validation(features_df, context_col, sync_feature, test_size=0.3):
    """
    Perform cross-validation to verify if context-synchronization relationships generalize.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with features and synchronization
    context_col : str
        Name of the context column
    sync_feature : str
        Name of the synchronization feature
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    tuple
        (train_diff, test_diff, generalizes)
    """
    # Prepare data
    df = features_df[[context_col, sync_feature]].dropna()
    
    # Randomly split data
    np.random.seed(42)
    mask = np.random.rand(len(df)) >= test_size
    train = df[mask]
    test = df[~mask]
    
    # Calculate differences in training set
    train_present = train[train[context_col] == 1][sync_feature].mean()
    train_absent = train[train[context_col] == 0][sync_feature].mean()
    train_diff = train_present - train_absent
    
    # Calculate differences in test set
    test_present = test[test[context_col] == 1][sync_feature].mean()
    test_absent = test[test[context_col] == 0][sync_feature].mean()
    test_diff = test_present - test_absent
    
    # Check if relationship generalizes
    # It generalizes if the sign of the effect is the same
    generalizes = (train_diff * test_diff) > 0
    
    return train_diff, test_diff, generalizes


def sensitivity_analysis(features_df, window_sizes, sync_thresholds, context_col, hr_feature):
    """
    Perform sensitivity analysis by varying window size and synchronization thresholds.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with features
    window_sizes : list
        List of window sizes to test
    sync_thresholds : list
        List of synchronization thresholds to test
    context_col : str
        Name of the context column
    hr_feature : str
        Name of the heart rate feature
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with sensitivity analysis results
    """
    from src.synchronization import calculate_windowwise_correlation, binary_synchronization
    
    results = []
    
    # Define heart rate features
    hr_features = [hr_feature]  # Only analyze one feature for sensitivity
    
    for window_size in window_sizes:
        # Recalculate window-wise correlation with different window size
        window_corr = calculate_windowwise_correlation(features_df, hr_features, window_size)
        
        for threshold in sync_thresholds:
            # Recalculate binary synchronization with different threshold
            binary_sync = binary_synchronization(window_corr, hr_feature, threshold)
            
            # Calculate synchronization probabilities
            probs = calculate_sync_probability_by_context(features_df, binary_sync, context_col)
            
            if probs[0] is not None:
                results.append({
                    'window_size': window_size,
                    'threshold': threshold,
                    'sync_prob_context_present': probs[0],
                    'sync_prob_context_absent': probs[1],
                    'odds_ratio': probs[2],
                    'p_value': probs[3]
                })
    
    return pd.DataFrame(results) 