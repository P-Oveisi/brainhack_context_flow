#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis module for heart rate synchronization studies.

This module provides statistical analysis functions for examining 
relationships between audio features and heart rate synchronization.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings

# Suppress specific warnings that might appear during analysis
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")


def compute_statistics(df, feature_cols, target_cols):
    """
    Compute statistical measures for relationships between features and targets.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features and target variables
    feature_cols : list
        List of feature column names
    target_cols : list
        List of target column names
        
    Returns:
    --------
    stats_dict : dict
        Dictionary with statistical results
    """
    stats_dict = {
        'sample_size': len(df),
        'feature_count': len(feature_cols),
        'target_count': len(target_cols),
        'correlations': {},
        'significance': {},
        'r_squared': {}
    }
    
    # Compute correlations and p-values
    for target in target_cols:
        if target not in df.columns:
            continue
            
        corr_results = {}
        p_values = {}
        r_squared = {}
        
        for feature in feature_cols:
            if feature not in df.columns:
                continue
                
            # Remove NaN values for analysis
            valid_data = df[[feature, target]].dropna()
            
            if len(valid_data) < 5:  # Skip if too few valid data points
                continue
                
            # Pearson correlation
            corr, p_val = stats.pearsonr(valid_data[feature], valid_data[target])
            corr_results[feature] = corr
            p_values[feature] = p_val
            
            # Calculate R-squared using simple linear regression
            X = valid_data[feature].values.reshape(-1, 1)
            y = valid_data[target].values
            
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            r_squared[feature] = r2
            
        stats_dict['correlations'][target] = corr_results
        stats_dict['significance'][target] = p_values
        stats_dict['r_squared'][target] = r_squared
    
    # Add correction for multiple testing
    stats_dict['adjusted_significance'] = {}
    
    for target in target_cols:
        if target not in stats_dict['significance']:
            continue
            
        p_values = stats_dict['significance'][target]
        if not p_values:
            continue
            
        # Benjamini-Hochberg correction
        sorted_p = sorted([(feat, p) for feat, p in p_values.items()], key=lambda x: x[1])
        m = len(sorted_p)
        
        adjusted_p = {}
        for i, (feature, p) in enumerate(sorted_p):
            # BH adjustment formula
            adjusted_p[feature] = min(p * m / (i + 1), 1.0)
            
        stats_dict['adjusted_significance'][target] = adjusted_p
    
    return stats_dict


def run_permutation_test(df, feature, target, n_permutations=1000):
    """
    Perform permutation test to assess significance of correlation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing feature and target variables
    feature : str
        Feature column name
    target : str
        Target column name
    n_permutations : int, default=1000
        Number of permutations
        
    Returns:
    --------
    result : dict
        Dictionary with permutation test results
    """
    # Remove NaN values
    valid_data = df[[feature, target]].dropna()
    
    if len(valid_data) < 10:
        return {
            'error': 'Not enough valid data points for permutation test',
            'feature': feature,
            'target': target
        }
    
    # Calculate observed correlation
    observed_corr, _ = stats.pearsonr(valid_data[feature], valid_data[target])
    
    # Permutation test
    permutation_corrs = []
    
    for _ in range(n_permutations):
        # Shuffle the target variable
        shuffled_target = np.random.permutation(valid_data[target].values)
        
        # Calculate correlation with shuffled data
        perm_corr, _ = stats.pearsonr(valid_data[feature], shuffled_target)
        permutation_corrs.append(perm_corr)
    
    # Calculate p-value
    p_value = np.mean(np.abs(permutation_corrs) >= np.abs(observed_corr))
    
    # Calculate 95% confidence interval
    ci_lower = np.percentile(permutation_corrs, 2.5)
    ci_upper = np.percentile(permutation_corrs, 97.5)
    
    return {
        'feature': feature,
        'target': target,
        'observed_correlation': observed_corr,
        'permutation_correlations': permutation_corrs,
        'p_value': p_value,
        'confidence_interval': (ci_lower, ci_upper),
        'significant': p_value < 0.05,
        'n_permutations': n_permutations
    }


def run_cross_validation(df, feature_cols, target, k_folds=5):
    """
    Perform cross-validation to assess prediction stability.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features and target variable
    feature_cols : list
        List of feature column names
    target : str
        Target column name
    k_folds : int, default=5
        Number of folds for cross-validation
        
    Returns:
    --------
    result : dict
        Dictionary with cross-validation results
    """
    # Check if enough data for k-fold
    if len(df) < k_folds * 2:
        return {
            'error': f'Not enough data points for {k_folds}-fold cross-validation',
            'features': feature_cols,
            'target': target
        }
    
    # Remove rows with NaN in relevant columns
    valid_data = df[feature_cols + [target]].dropna()
    
    if len(valid_data) < k_folds * 2:
        return {
            'error': f'Not enough valid data points for {k_folds}-fold cross-validation after removing NaN',
            'features': feature_cols,
            'target': target
        }
    
    # Prepare data
    X = valid_data[feature_cols].values
    y = valid_data[target].values
    
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Metrics for each fold
    r2_scores = []
    mse_scores = []
    feature_importances = []
    
    # Perform cross-validation
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        r2_scores.append(r2)
        mse_scores.append(mse)
        
        # Store feature coefficients
        feature_importances.append(model.coef_)
    
    # Calculate feature importance stability
    feature_importance_df = pd.DataFrame(feature_importances, columns=feature_cols)
    feature_stability = {
        col: {
            'mean': feature_importance_df[col].mean(),
            'std': feature_importance_df[col].std(),
            'cv': np.abs(feature_importance_df[col].std() / (feature_importance_df[col].mean() + 1e-10))
        }
        for col in feature_cols
    }
    
    # Return results
    return {
        'features': feature_cols,
        'target': target,
        'r2_scores': r2_scores,
        'mean_r2': np.mean(r2_scores),
        'std_r2': np.std(r2_scores),
        'mse_scores': mse_scores,
        'mean_mse': np.mean(mse_scores),
        'std_mse': np.std(mse_scores),
        'feature_importance': feature_stability,
        'k_folds': k_folds
    }


def calculate_effect_sizes(df, feature_cols, target_cols):
    """
    Calculate effect sizes for relationships between features and targets.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features and target variables
    feature_cols : list
        List of feature column names
    target_cols : list
        List of target column names
        
    Returns:
    --------
    effect_sizes : dict
        Dictionary with effect size calculations
    """
    effect_sizes = {}
    
    for target in target_cols:
        if target not in df.columns:
            continue
            
        target_effects = {}
        
        for feature in feature_cols:
            if feature not in df.columns:
                continue
                
            # Remove NaN values
            valid_data = df[[feature, target]].dropna()
            
            if len(valid_data) < 10:
                continue
                
            # Calculate Cohen's d
            group1 = valid_data[valid_data[feature] > valid_data[feature].median()][target]
            group2 = valid_data[valid_data[feature] <= valid_data[feature].median()][target]
            
            # Pooled standard deviation
            n1, n2 = len(group1), len(group2)
            s1, s2 = group1.std(), group2.std()
            
            # Avoid division by zero
            if s1 == 0 and s2 == 0:
                pooled_std = 1e-10
            else:
                pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
            
            # Cohen's d
            d = (group1.mean() - group2.mean()) / pooled_std
            
            # Calculate r-squared from correlation
            corr, _ = stats.pearsonr(valid_data[feature], valid_data[target])
            r_squared = corr**2
            
            target_effects[feature] = {
                'cohens_d': d,
                'r_squared': r_squared,
                'correlation': corr,
                'sample_size': len(valid_data)
            }
            
        effect_sizes[target] = target_effects
    
    return effect_sizes


def analyze_time_lag_correlations(df, feature_col, target_col, max_lag=10):
    """
    Analyze correlations between a feature and target at different time lags.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing feature and target variables
    feature_col : str
        Feature column name
    target_col : str
        Target column name
    max_lag : int, default=10
        Maximum lag to analyze
        
    Returns:
    --------
    lag_results : dict
        Dictionary with time lag correlation results
    """
    if feature_col not in df.columns or target_col not in df.columns:
        return {
            'error': 'Feature or target column not found in DataFrame',
            'feature': feature_col,
            'target': target_col
        }
    
    # Extract series
    feature_series = df[feature_col].dropna()
    target_series = df[target_col].dropna()
    
    # Ensure same index
    common_idx = feature_series.index.intersection(target_series.index)
    feature_series = feature_series.loc[common_idx]
    target_series = target_series.loc[common_idx]
    
    if len(feature_series) < max_lag * 2:
        return {
            'error': 'Not enough data points for lag analysis',
            'feature': feature_col,
            'target': target_col
        }
    
    # Calculate correlations at different lags
    lag_correlations = {}
    
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # Feature leads target (shift target forward)
            shifted_target = target_series.shift(-lag)
            valid_data = pd.DataFrame({
                'feature': feature_series,
                'target': shifted_target
            }).dropna()
        elif lag > 0:
            # Target leads feature (shift feature forward)
            shifted_feature = feature_series.shift(lag)
            valid_data = pd.DataFrame({
                'feature': shifted_feature,
                'target': target_series
            }).dropna()
        else:
            # No lag
            valid_data = pd.DataFrame({
                'feature': feature_series,
                'target': target_series
            }).dropna()
        
        if len(valid_data) < 10:
            lag_correlations[lag] = np.nan
            continue
        
        # Calculate correlation
        corr, p_value = stats.pearsonr(valid_data['feature'], valid_data['target'])
        lag_correlations[lag] = corr
    
    # Find optimal lag (maximum absolute correlation)
    valid_lags = {lag: corr for lag, corr in lag_correlations.items() 
                 if not np.isnan(corr)}
    
    if not valid_lags:
        optimal_lag = 0
        max_correlation = np.nan
    else:
        optimal_lag = max(valid_lags.items(), key=lambda x: abs(x[1]))[0]
        max_correlation = valid_lags[optimal_lag]
    
    return {
        'feature': feature_col,
        'target': target_col,
        'lag_correlations': lag_correlations,
        'optimal_lag': optimal_lag,
        'max_correlation': max_correlation
    }


def run_feature_importance_analysis(df, feature_cols, target_col):
    """
    Perform feature importance analysis for predicting a target variable.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features and target variable
    feature_cols : list
        List of feature column names
    target_col : str
        Target column name
        
    Returns:
    --------
    importance_results : dict
        Dictionary with feature importance results
    """
    # Remove rows with NaN in relevant columns
    valid_data = df[feature_cols + [target_col]].dropna()
    
    if len(valid_data) < 10:
        return {
            'error': 'Not enough valid data points for feature importance analysis',
            'features': feature_cols,
            'target': target_col
        }
    
    # Initialize results dictionary
    importance_results = {
        'features': feature_cols,
        'target': target_col,
        'univariate_importance': {},
        'multivariate_importance': {},
        'correlation_matrix': {}
    }
    
    # Calculate univariate correlations
    for feature in feature_cols:
        corr, p_val = stats.pearsonr(valid_data[feature], valid_data[target_col])
        importance_results['univariate_importance'][feature] = {
            'correlation': corr,
            'p_value': p_val,
            'r_squared': corr**2
        }
    
    # Prepare data for multivariate analysis
    X = valid_data[feature_cols].values
    y = valid_data[target_col].values
    
    # Calculate correlation matrix between features
    corr_matrix = valid_data[feature_cols].corr()
    importance_results['correlation_matrix'] = corr_matrix.to_dict()
    
    # Calculate multivariate feature importance using linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Store coefficients and calculate importance scores
    for i, feature in enumerate(feature_cols):
        importance_results['multivariate_importance'][feature] = {
            'coefficient': model.coef_[i],
            'abs_coefficient': abs(model.coef_[i])
        }
    
    # Add model performance metrics
    y_pred = model.predict(X)
    importance_results['model_performance'] = {
        'r_squared': r2_score(y, y_pred),
        'mse': mean_squared_error(y, y_pred),
        'intercept': model.intercept_
    }
    
    # Normalize feature importance
    total_importance = sum(item['abs_coefficient'] 
                          for item in importance_results['multivariate_importance'].values())
    
    if total_importance > 0:
        for feature in feature_cols:
            abs_coef = importance_results['multivariate_importance'][feature]['abs_coefficient']
            importance_results['multivariate_importance'][feature]['normalized_importance'] = abs_coef / total_importance
    
    return importance_results 