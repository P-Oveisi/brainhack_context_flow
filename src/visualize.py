#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization module for heart rate synchronization analysis.

This module provides functions to create visualizations for the analysis
of heart rate synchronization and its relationship with audio features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Set default plot style
plt.style.use('seaborn-whitegrid')
sns.set_context("paper", font_scale=1.2)


def plot_timeline(features_df, window_corr, sync_series=None, hr_feature='hr_mean', figsize=(15, 12)):
    """
    Create a timeline plot with heart rates, synchronization, and audio features.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with features for both participants
    window_corr : pd.DataFrame
        DataFrame with window-wise correlation coefficients
    sync_series : pd.Series
        Synchronization series (optional)
    hr_feature : str
        Heart rate feature to plot
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 2]})
    
    # Convert timestamps to datetime for better x-axis labeling
    timestamps = [datetime.fromtimestamp(ts) for ts in features_df['window_start']]
    
    # Plot 1: Heart rates
    ax1 = axes[0]
    p1_feature = f'p1_{hr_feature}'
    p2_feature = f'p2_{hr_feature}'
    
    ax1.plot(timestamps, features_df[p1_feature], label=f'Participant 1 {hr_feature}', color='blue')
    ax1.plot(timestamps, features_df[p2_feature], label=f'Participant 2 {hr_feature}', color='red')
    ax1.set_ylabel(f'{hr_feature}')
    ax1.legend()
    ax1.set_title(f'Heart Rate Feature: {hr_feature}')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correlation strength
    ax2 = axes[1]
    corr_col = f'corr_{hr_feature}'
    
    if corr_col in window_corr.columns:
        corr_timestamps = [datetime.fromtimestamp(ts) for ts in window_corr['window_start']]
        ax2.plot(corr_timestamps, window_corr[corr_col], color='green')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.fill_between(
            corr_timestamps, 
            window_corr[corr_col], 
            0, 
            where=(window_corr[corr_col] > 0), 
            alpha=0.3, 
            color='green'
        )
        ax2.fill_between(
            corr_timestamps, 
            window_corr[corr_col], 
            0, 
            where=(window_corr[corr_col] < 0), 
            alpha=0.3, 
            color='red'
        )
        
    ax2.set_ylabel('Correlation')
    ax2.set_ylim(-1, 1)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Synchronization strength (if provided)
    ax3 = axes[2]
    if sync_series is not None and len(sync_series) > 0:
        # For continuous synchronization measure, plot a line
        sync_timestamps = corr_timestamps if len(corr_timestamps) == len(sync_series) else timestamps[:len(sync_series)]
        ax3.plot(sync_timestamps, sync_series, color='purple')
        ax3.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, label='Threshold (0.6)')
        ax3.fill_between(
            sync_timestamps, 
            sync_series, 
            0, 
            where=(sync_series > 0), 
            alpha=0.3, 
            color='purple'
        )
        
    ax3.set_ylabel('Sync Strength')
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Audio features
    ax4 = axes[3]
    audio_features = ['audio_tempo', 'audio_dom_freq', 'audio_contrast', 'audio_complexity', 'audio_dynamic_range']
    
    # Normalize audio features for better visualization
    normalized_features = {}
    for feature in audio_features:
        if feature in features_df.columns:
            values = features_df[feature].values
            min_val = np.min(values)
            max_val = np.max(values)
            normalized_features[feature] = (values - min_val) / (max_val - min_val) if max_val > min_val else values
    
    # Plot normalized audio features
    for feature, values in normalized_features.items():
        ax4.plot(timestamps, values, label=feature)
        
    ax4.set_ylabel('Normalized Value')
    ax4.set_ylim(-0.1, 1.1)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # Format x-axis
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax4.set_xlabel('Time')
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(feature_sync_results, figsize=(10, 8)):
    """
    Create a heatmap showing correlations between audio features and synchronization.
    
    Parameters:
    -----------
    feature_sync_results : dict
        Dictionary with feature-synchronization analysis results
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Extract data
    hr_features = []
    audio_features = []
    correlation_values = []
    p_values = []
    
    for hr_feature, results in feature_sync_results.items():
        for audio_feature, stats in results.items():
            hr_features.append(hr_feature)
            audio_features.append(audio_feature)
            correlation_values.append(stats['correlation'])
            p_values.append(stats['p_value'])
    
    # Create DataFrame for heatmap
    df = pd.DataFrame({
        'hr_feature': hr_features,
        'audio_feature': audio_features,
        'correlation': correlation_values,
        'p_value': p_values
    })
    
    # Reshape for heatmap
    heatmap_data = df.pivot_table(index='audio_feature', columns='hr_feature', values='correlation')
    
    # Create significance mask
    sig_mask = df.pivot_table(index='audio_feature', columns='hr_feature', values='p_value') > 0.05
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        center=0,
        mask=sig_mask,
        ax=ax,
        fmt='.2f'
    )
    
    ax.set_title('Correlation Between Audio Features and Heart Rate Synchronization')
    
    return fig


def plot_correlation_values(feature_sync_results, hr_feature=None, figsize=(12, 8)):
    """
    Create a bar chart of correlation values with error bars.
    
    Parameters:
    -----------
    feature_sync_results : dict
        Dictionary with feature-synchronization analysis results
    hr_feature : str
        Heart rate feature to plot (None for all features)
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Extract data
    data = []
    
    for hr_feat, results in feature_sync_results.items():
        if hr_feature is not None and hr_feat != hr_feature:
            continue
            
        for audio_feat, stats in results.items():
            data.append({
                'hr_feature': hr_feat,
                'audio_feature': audio_feat,
                'correlation': stats['correlation'],
                'p_value': stats['p_value'],
                'significant': stats['p_value'] <= 0.05
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by correlation value
    df['abs_correlation'] = np.abs(df['correlation'])
    df = df.sort_values('abs_correlation', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors based on significance
    colors = ['green' if sig else 'gray' for sig in df['significant']]
    
    # Group by heart rate feature if multiple
    if hr_feature is None:
        grouped = df.groupby('hr_feature')
        
        # Plot multiple groups
        width = 0.8 / len(grouped)
        offset = -0.4 + width/2
        
        for i, (name, group) in enumerate(grouped):
            ax.bar(
                np.arange(len(group)) + offset + i*width, 
                group['correlation'],
                width=width,
                color=colors,
                alpha=0.7,
                label=name
            )
            
            # Add significance markers
            for j, (_, row) in enumerate(group.iterrows()):
                if row['significant']:
                    ax.text(j + offset + i*width, row['correlation'], '*', 
                            ha='center', va='bottom' if row['correlation'] > 0 else 'top')
            
            # Set x-tick labels from first group
            if i == 0:
                ax.set_xticks(np.arange(len(group)))
                ax.set_xticklabels(group['audio_feature'])
    else:
        # Simple bar chart for single heart rate feature
        ax.bar(
            df['audio_feature'], 
            df['correlation'],
            color=colors,
            alpha=0.7
        )
        
        # Add significance markers
        for i, (_, row) in enumerate(df.iterrows()):
            if row['significant']:
                ax.text(i, row['correlation'], '*', 
                        ha='center', va='bottom' if row['correlation'] > 0 else 'top')
    
    # Add reference line at zero
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Label axes
    ax.set_xlabel('Audio Feature')
    ax.set_ylabel("Correlation")
    
    if hr_feature is not None:
        ax.set_title(f'Correlation Values for {hr_feature}')
    else:
        ax.set_title('Correlation Values for All Features')
        ax.legend()
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    
    # Add correlation interpretation guide
    ax.axhspan(0.3, 0.5, alpha=0.1, color='yellow', xmax=0.05)
    ax.axhspan(0.5, 0.7, alpha=0.1, color='orange', xmax=0.05)
    ax.axhspan(0.7, 1.0, alpha=0.1, color='red', xmax=0.05)
    ax.axhspan(-0.5, -0.3, alpha=0.1, color='yellow', xmax=0.05)
    ax.axhspan(-0.7, -0.5, alpha=0.1, color='orange', xmax=0.05)
    ax.axhspan(-1.0, -0.7, alpha=0.1, color='red', xmax=0.05)
    
    ax.text(df.shape[0] * 1.02, 0.4, 'Weak', va='center')
    ax.text(df.shape[0] * 1.02, 0.6, 'Moderate', va='center')
    ax.text(df.shape[0] * 1.02, 0.85, 'Strong', va='center')
    ax.text(df.shape[0] * 1.02, -0.4, 'Weak', va='center')
    ax.text(df.shape[0] * 1.02, -0.6, 'Moderate', va='center')
    ax.text(df.shape[0] * 1.02, -0.85, 'Strong', va='center')
    
    plt.tight_layout()
    return fig


def plot_feature_profiles(features_df, figsize=(15, 10)):
    """
    Create a timeline plot of audio features.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with audio features over time
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Identify audio feature columns
    audio_cols = ['audio_tempo', 'audio_dom_freq', 'audio_contrast', 'audio_complexity', 'audio_dynamic_range']
    
    # Convert timestamps to datetime for better x-axis labeling
    timestamps = [datetime.fromtimestamp(ts) for ts in features_df['window_start']]
    
    # Create figure
    fig, axes = plt.subplots(len(audio_cols), 1, figsize=figsize, sharex=True)
    
    # Plot each audio feature as a separate subplot
    for i, col in enumerate(audio_cols):
        if col in features_df.columns:
            ax = axes[i]
            
            # Normalize for better visualization
            values = features_df[col].values
            
            # Plot the feature
            ax.plot(timestamps, values, color=f'C{i}')
            
            # Add labels
            ax.set_ylabel(col.replace('audio_', ''))
            ax.grid(True, alpha=0.3)
    
    # Format x-axis on the bottom subplot
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    axes[-1].set_xlabel('Time')
    
    # Add title
    fig.suptitle('Audio Feature Profiles Over Time')
    
    plt.tight_layout()
    return fig


def plot_lag_analysis(lag_results, figsize=(10, 6)):
    """
    Create a plot of time-lag correlation results.
    
    Parameters:
    -----------
    lag_results : dict
        Dictionary with time-lag correlation results
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Extract data
    features = list(lag_results.keys())
    optimal_lags = [lag_results[f]['optimum_lag'] for f in features]
    max_correlations = [lag_results[f]['max_correlation'] for f in features]
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot optimal lags as bars
    x = np.arange(len(features))
    bars = ax1.bar(x, optimal_lags, width=0.4, alpha=0.6, color='blue', label='Optimal Lag (windows)')
    
    # Create second y-axis for correlation values
    ax2 = ax1.twinx()
    ax2.plot(x, max_correlations, 'ro-', label='Max Correlation')
    
    # Add reference line at zero lag
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Label axes
    ax1.set_xlabel('Heart Rate Feature')
    ax1.set_ylabel('Optimal Lag (windows)')
    ax2.set_ylabel('Maximum Correlation')
    
    # Set x-tick labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(features, rotation=45, ha='right')
    
    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title('Time Lag Analysis Results')
    plt.tight_layout()
    return fig


def plot_sensitivity_heatmap(sensitivity_df, feature_col, hr_feature, figsize=(10, 8)):
    """
    Create a heatmap showing sensitivity analysis results.
    
    Parameters:
    -----------
    sensitivity_df : pd.DataFrame
        DataFrame with sensitivity analysis results
    feature_col : str
        Name of the audio feature column
    hr_feature : str
        Name of the heart rate feature
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Extract unique window sizes and thresholds
    window_sizes = sorted(sensitivity_df['window_size'].unique())
    thresholds = sorted(sensitivity_df['threshold'].unique())
    
    # Create a pivot table
    pivot_data = sensitivity_df.pivot_table(
        index='window_size',
        columns='threshold',
        values='correlation',
        aggfunc='mean'
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        pivot_data, 
        annot=True, 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        center=0,
        ax=ax,
        fmt='.2f'
    )
    
    ax.set_title(f'Sensitivity Analysis: Effect of Window Size and Threshold\n'
                f'Feature: {feature_col}, Heart Rate Measure: {hr_feature}')
    ax.set_xlabel('Correlation Threshold')
    ax.set_ylabel('Window Size (windows)')
    
    return fig


def plot_timeline(df, hr_features, window_index=None, figsize=(12, 8), save_path=None):
    """
    Plot timeline of heart rate data and synchronization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing heart rate data and synchronization metrics
    hr_features : list
        List of heart rate features to plot
    window_index : int, optional
        Highlight a specific window
    figsize : tuple, default=(12, 8)
        Figure size
    save_path : str or Path, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    fig, axes = plt.subplots(len(hr_features), 1, figsize=figsize, sharex=True)
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    x = np.arange(len(df))
    
    for i, feature in enumerate(hr_features):
        ax = axes[i]
        
        if feature in df.columns:
            ax.plot(x, df[feature], 'b-', linewidth=1.5, label=feature)
            
            # If sync feature exists, plot it
            sync_feature = f"sync_{feature}" if f"sync_{feature}" in df.columns else None
            if sync_feature:
                ax2 = ax.twinx()
                ax2.plot(x, df[sync_feature], 'r-', alpha=0.7, linewidth=1.5, label=sync_feature)
                ax2.set_ylabel('Synchronization', color='r')
                ax2.tick_params(axis='y', colors='r')
                
                # Add secondary legend
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(loc='upper left')
                ax2.legend(lines2, labels2, loc='upper right')
            else:
                ax.legend()
            
            # Highlight specific window if requested
            if window_index is not None and 0 <= window_index < len(df):
                ax.axvline(x=window_index, color='g', linestyle='--', alpha=0.7)
        
        ax.set_title(feature)
        ax.set_ylabel('Value')
        
    axes[-1].set_xlabel('Time Window')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def plot_correlation_heatmap(correlation_data, figsize=(12, 10), save_path=None):
    """
    Plot correlation heatmap between audio features and heart rate synchronization.
    
    Parameters:
    -----------
    correlation_data : pandas.DataFrame or dict
        Correlation data between features and synchronization
    figsize : tuple, default=(12, 10)
        Figure size
    save_path : str or Path, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Convert dict to DataFrame if needed
    if isinstance(correlation_data, dict):
        corr_df = pd.DataFrame(correlation_data)
    else:
        corr_df = correlation_data
    
    # Sort features by absolute mean correlation
    corr_mean = corr_df.abs().mean(axis=1).sort_values(ascending=False)
    sorted_features = corr_mean.index
    sorted_df = corr_df.loc[sorted_features]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(sorted_df, annot=True, cmap=cmap, center=0,
                linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    
    # Set labels and title
    ax.set_title('Correlation between Audio Features and Heart Rate Synchronization')
    ax.set_ylabel('Audio Feature')
    ax.set_xlabel('Heart Rate Synchronization Feature')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(correlation_data, top_n=10, figsize=(12, 8), save_path=None):
    """
    Plot feature importance based on correlation with synchronization.
    
    Parameters:
    -----------
    correlation_data : dict
        Dictionary with correlation results for each synchronization feature
    top_n : int, default=10
        Number of top features to display
    figsize : tuple, default=(12, 8)
        Figure size
    save_path : str or Path, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Prepare figure
    fig, axes = plt.subplots(len(correlation_data), 1, figsize=figsize, sharex=True)
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    # Plot for each synchronization feature
    for i, (sync_feature, corr_series) in enumerate(correlation_data.items()):
        ax = axes[i]
        
        # Get top features by absolute correlation
        top_features = corr_series.abs().sort_values(ascending=False).head(top_n).index
        top_corr = corr_series[top_features]
        
        # Create bar plot
        bars = ax.barh(top_features, top_corr, color=plt.cm.coolwarm(np.interp(top_corr, [-1, 1], [0, 1])))
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width >= 0 else width - 0.05
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', va='center', ha='left' if width >= 0 else 'right')
        
        # Set labels and title
        ax.set_title(f'Top {top_n} Features Correlated with {sync_feature}')
        ax.set_xlabel('Correlation Coefficient')
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Set limits
        max_abs_corr = top_corr.abs().max()
        ax.set_xlim(-max_abs_corr - 0.1, max_abs_corr + 0.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_continuous_features(df, features, sync_features, window_size=10, figsize=(14, 10), save_path=None):
    """
    Plot continuous audio features alongside synchronization metrics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing audio features and synchronization data
    features : list
        List of audio features to plot
    sync_features : list
        List of synchronization features to include
    window_size : int, default=10
        Number of windows to show in the moving average
    figsize : tuple, default=(14, 10)
        Figure size
    save_path : str or Path, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Compute moving averages
    df_smooth = df.copy()
    for col in features + sync_features:
        if col in df.columns:
            df_smooth[f"{col}_smooth"] = df[col].rolling(window=window_size, center=True).mean()
    
    # Create subplot for each feature
    fig, axes = plt.subplots(len(features), 1, figsize=figsize, sharex=True)
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    x = np.arange(len(df))
    
    # Plot each feature
    for i, feature in enumerate(features):
        ax = axes[i]
        
        if feature in df.columns:
            # Plot raw data and smoothed line
            ax.plot(x, df[feature], 'b-', alpha=0.4, linewidth=1, label=feature)
            ax.plot(x, df_smooth[f"{feature}_smooth"], 'b-', linewidth=2, label=f"{feature} (smoothed)")
            ax.set_ylabel(feature, color='b')
            ax.tick_params(axis='y', colors='b')
            
            # Create secondary axis for synchronization
            ax2 = ax.twinx()
            
            # Plot each sync feature
            colors = plt.cm.tab10(np.linspace(0, 1, len(sync_features)))
            for j, sync_feature in enumerate(sync_features):
                if sync_feature in df.columns:
                    color = colors[j]
                    ax2.plot(x, df_smooth[f"{sync_feature}_smooth"], '-', color=color, 
                            linewidth=2, label=f"{sync_feature} (smoothed)")
            
            ax2.set_ylabel('Synchronization', color='r')
            ax2.tick_params(axis='y', colors='r')
            
            # Add legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1, labels1, loc='upper left')
            ax2.legend(lines2, labels2, loc='upper right')
            
        ax.set_title(feature)
    
    axes[-1].set_xlabel('Time Window')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_lag_analysis(lag_results, feature, max_lag=10, figsize=(10, 6), save_path=None):
    """
    Plot lag analysis for a feature.
    
    Parameters:
    -----------
    lag_results : dict
        Dictionary containing lag analysis results
    feature : str
        Feature to plot lag analysis for
    max_lag : int, default=10
        Maximum lag to display
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : str or Path, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    if feature not in lag_results:
        raise ValueError(f"Feature '{feature}' not found in lag results")
    
    # Extract lag data for the feature
    lag_data = lag_results[feature]
    lags = list(range(-max_lag, max_lag + 1))
    correlations = [lag_data.get(lag, 0) for lag in lags]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot lag correlations
    ax.plot(lags, correlations, 'o-', linewidth=2, markersize=8)
    
    # Add zero line
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Find and highlight max correlation
    max_corr_idx = np.argmax(np.abs(correlations))
    max_corr_lag = lags[max_corr_idx]
    max_corr_value = correlations[max_corr_idx]
    
    ax.plot(max_corr_lag, max_corr_value, 'ro', markersize=10)
    ax.annotate(f'Max: {max_corr_value:.3f} at lag {max_corr_lag}',
                xy=(max_corr_lag, max_corr_value),
                xytext=(max_corr_lag + 1, max_corr_value + 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Set labels and title
    ax.set_title(f'Lag Analysis for {feature}')
    ax.set_xlabel('Lag (windows)')
    ax.set_ylabel('Correlation')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_comparison_boxplot(df, group_col, value_cols, figsize=(14, 8), save_path=None):
    """
    Create boxplots comparing groups for different features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    group_col : str
        Column name to use for grouping
    value_cols : list
        List of column names for values to compare
    figsize : tuple, default=(14, 8)
        Figure size
    save_path : str or Path, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Prepare data for plotting
    plot_data = []
    
    for value_col in value_cols:
        if value_col in df.columns:
            temp_df = df[[group_col, value_col]].copy()
            temp_df['Feature'] = value_col
            temp_df.rename(columns={value_col: 'Value'}, inplace=True)
            plot_data.append(temp_df)
    
    if not plot_data:
        raise ValueError("No valid value columns found in DataFrame")
    
    combined_df = pd.concat(plot_data, ignore_index=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create boxplot
    sns.boxplot(x='Feature', y='Value', hue=group_col, data=combined_df, ax=ax)
    
    # Set labels and title
    ax.set_title('Comparison of Features by Group')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Value')
    
    # Rotate x-tick labels if many features
    if len(value_cols) > 4:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_dashboard(results, session_id, save_dir=None):
    """
    Create a dashboard with multiple visualizations.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing exploration results
    session_id : str
        Session identifier
    save_dir : str or Path, optional
        Directory to save the dashboard
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Define grid layout
    gs = fig.add_gridspec(3, 3)
    
    # Add title
    fig.suptitle(f'Heart Rate Synchronization Analysis Dashboard - Session {session_id}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add correlation heatmap
    ax1 = fig.add_subplot(gs[0, :])
    if 'correlation' in results:
        corr_df = pd.DataFrame(results['correlation'])
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, ax=ax1)
        ax1.set_title('Feature-Synchronization Correlation Heatmap')
    
    # Add feature importance plot
    ax2 = fig.add_subplot(gs[1, :2])
    if 'top_features' in results and results['top_features']:
        sync_feature = list(results['top_features'].keys())[0]
        top_features = results['top_features'][sync_feature]
        features = list(top_features.keys())[:8]
        values = list(top_features.values())[:8]
        colors = plt.cm.coolwarm(np.interp(values, [-1, 1], [0, 1]))
        ax2.barh(features, values, color=colors)
        ax2.set_title(f'Top Features Correlated with {sync_feature}')
        ax2.set_xlabel('Correlation Coefficient')
        ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add statistics summary
    ax3 = fig.add_subplot(gs[1, 2])
    if 'statistics' in results:
        stats = results['statistics']
        ax3.axis('off')
        ax3.text(0.1, 0.9, 'Statistical Summary', fontsize=14, fontweight='bold')
        y_pos = 0.8
        for i, (key, value) in enumerate(stats.items()):
            if isinstance(value, dict):
                ax3.text(0.1, y_pos, f"{key}:", fontsize=12, fontweight='bold')
                y_pos -= 0.05
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        ax3.text(0.2, y_pos, f"{subkey}: {subvalue:.3f}", fontsize=10)
                    else:
                        ax3.text(0.2, y_pos, f"{subkey}: {subvalue}", fontsize=10)
                    y_pos -= 0.04
            else:
                if isinstance(value, (int, float)):
                    ax3.text(0.1, y_pos, f"{key}: {value:.3f}", fontsize=12)
                else:
                    ax3.text(0.1, y_pos, f"{key}: {value}", fontsize=12)
                y_pos -= 0.05
    
    # Add information box
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    ax4.text(0.1, 0.9, 'Analysis Information', fontsize=14, fontweight='bold')
    ax4.text(0.1, 0.8, f'Session ID: {session_id}', fontsize=12)
    ax4.text(0.1, 0.7, f'Number of synchronization features: {len(results.get("correlation", {}))}', fontsize=12)
    ax4.text(0.1, 0.6, f'Number of audio features analyzed: {len(results.get("top_features", {}).get(list(results.get("top_features", {}).keys())[0], {}))}', fontsize=12)
    ax4.text(0.1, 0.5, 'Interpretation Notes:', fontsize=12, fontweight='bold')
    ax4.text(0.1, 0.4, '• Positive correlation: Feature increases with synchronization', fontsize=11)
    ax4.text(0.1, 0.35, '• Negative correlation: Feature decreases with synchronization', fontsize=11)
    ax4.text(0.1, 0.3, '• Higher absolute correlation values indicate stronger relationships', fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_dir:
        save_path = Path(save_dir) / f"{session_id}_dashboard.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 