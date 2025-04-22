"""
Heart Rate Synchronization - Visualization Module

This module contains functions for visualizing results:
- Timeline plots of heart rates and synchronization
- Context-synchronization heatmaps
- Statistical result plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import seaborn as sns
from datetime import datetime


def plot_timeline(features_df, window_corr, binary_sync, hr_feature='hr_mean', figsize=(15, 12)):
    """
    Create a timeline plot with heart rates, synchronization, and audio features.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with features for both participants
    window_corr : pd.DataFrame
        DataFrame with window-wise correlation coefficients
    binary_sync : pd.Series
        Binary synchronization series
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
    
    # Plot 3: Binary synchronization
    ax3 = axes[2]
    if len(binary_sync) > 0:
        sync_values = binary_sync.values
        ax3.fill_between(corr_timestamps, 0, sync_values, step='mid', alpha=0.7, color='purple')
        
    ax3.set_ylabel('Sync State')
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Not Sync', 'Sync'])
    
    # Plot 4: Audio features
    ax4 = axes[3]
    audio_features = ['audio_energy', 'audio_zcr', 'audio_centroid', 'audio_flux', 'audio_voice_ratio']
    
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


def plot_context_heatmap(context_sync_results, effect_sizes_df, feature, figsize=(10, 8)):
    """
    Create a heatmap showing synchronization probability by context.
    
    Parameters:
    -----------
    context_sync_results : dict
        Dictionary with context-synchronization analysis results
    effect_sizes_df : pd.DataFrame
        DataFrame with effect sizes
    feature : str
        Heart rate feature to plot
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Extract data for the selected feature
    if feature not in context_sync_results:
        return None
    
    contexts = context_sync_results[feature]
    
    # Create data for heatmap
    context_names = []
    sync_probs_present = []
    sync_probs_absent = []
    
    for context, stats in contexts.items():
        context_names.append(context.replace('context_', ''))
        sync_probs_present.append(stats['sync_prob_context_present'])
        sync_probs_absent.append(stats['sync_prob_context_absent'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create DataFrame for heatmap
    heatmap_data = pd.DataFrame({
        'Context': context_names,
        'Context Present': sync_probs_present,
        'Context Absent': sync_probs_absent
    })
    
    # Reshape for heatmap
    heatmap_data = heatmap_data.set_index('Context').T
    
    # Create heatmap
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        cmap='YlGnBu', 
        vmin=0, 
        vmax=1, 
        ax=ax,
        fmt='.2f'
    )
    
    ax.set_title(f'Synchronization Probability by Context for {feature}')
    
    return fig


def plot_effect_sizes(effect_sizes_df, feature=None, figsize=(12, 8)):
    """
    Create a bar chart of effect sizes with confidence intervals.
    
    Parameters:
    -----------
    effect_sizes_df : pd.DataFrame
        DataFrame with effect sizes and confidence intervals
    feature : str
        Heart rate feature to plot (None for all features)
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Filter for feature if specified
    if feature is not None:
        df = effect_sizes_df[effect_sizes_df['feature'] == feature].copy()
    else:
        df = effect_sizes_df.copy()
    
    # Sort by effect size
    df['abs_effect'] = np.abs(df['effect_size'])
    df = df.sort_values('abs_effect', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors based on significance
    colors = ['green' if sig else 'gray' for sig in df['significant']]
    
    # Create bar chart
    bars = ax.bar(
        df['context'].str.replace('context_', ''), 
        df['effect_size'],
        color=colors,
        alpha=0.7
    )
    
    # Add significance markers
    for i, (_, row) in enumerate(df.iterrows()):
        if row['significant']:
            ax.text(i, row['effect_size'], '*', ha='center', va='bottom' if row['effect_size'] > 0 else 'top')
    
    # Add reference line at zero
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Label axes
    ax.set_xlabel('Audio Context')
    ax.set_ylabel("Cohen's h (Effect Size)")
    
    if feature is not None:
        ax.set_title(f'Effect Sizes for {feature}')
    else:
        ax.set_title('Effect Sizes for All Features')
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    
    # Add effect size interpretation guide
    ax.axhspan(0.2, 0.5, alpha=0.1, color='yellow', xmax=0.05)
    ax.axhspan(0.5, 0.8, alpha=0.1, color='orange', xmax=0.05)
    ax.axhspan(0.8, 1.5, alpha=0.1, color='red', xmax=0.05)
    ax.axhspan(-0.5, -0.2, alpha=0.1, color='yellow', xmax=0.05)
    ax.axhspan(-0.8, -0.5, alpha=0.1, color='orange', xmax=0.05)
    ax.axhspan(-1.5, -0.8, alpha=0.1, color='red', xmax=0.05)
    
    ax.text(df.shape[0] * 1.02, 0.35, 'Small', va='center')
    ax.text(df.shape[0] * 1.02, 0.65, 'Medium', va='center')
    ax.text(df.shape[0] * 1.02, 1.15, 'Large', va='center')
    ax.text(df.shape[0] * 1.02, -0.35, 'Small', va='center')
    ax.text(df.shape[0] * 1.02, -0.65, 'Medium', va='center')
    ax.text(df.shape[0] * 1.02, -1.15, 'Large', va='center')
    
    plt.tight_layout()
    return fig


def plot_context_profile(profile_df, figsize=(15, 10)):
    """
    Create a timeline plot of audio contexts.
    
    Parameters:
    -----------
    profile_df : pd.DataFrame
        DataFrame with context profiles over time
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Identify context columns
    context_cols = [col for col in profile_df.columns if col.startswith('context_')]
    
    # Convert timestamps to datetime for better x-axis labeling
    timestamps = [datetime.fromtimestamp(ts) for ts in profile_df['window_start']]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a colormap
    cmap = plt.cm.tab10
    
    # Plot each context as a colored band
    legend_elements = []
    for i, col in enumerate(context_cols):
        # Get context name without prefix
        context_name = col.replace('context_', '')
        
        # Plot context band
        plt.fill_between(
            timestamps, 
            i, i+0.8, 
            where=profile_df[col] == 1, 
            color=cmap(i % 10),
            alpha=0.7,
            label=context_name
        )
        
    # Set y-axis ticks at center of each context band
    plt.yticks(np.arange(len(context_cols)) + 0.4, [col.replace('context_', '') for col in context_cols])
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xlabel('Time')
    plt.ylabel('Audio Context')
    plt.title('Audio Context Profile Over Time')
    
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


def plot_sensitivity_heatmap(sensitivity_df, context_col, hr_feature, figsize=(10, 8)):
    """
    Create a heatmap showing sensitivity analysis results.
    
    Parameters:
    -----------
    sensitivity_df : pd.DataFrame
        DataFrame with sensitivity analysis results
    context_col : str
        Name of the context column
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
        values='sync_prob_context_present',
        aggfunc='mean'
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        pivot_data, 
        annot=True, 
        cmap='YlGnBu', 
        vmin=0, 
        vmax=1, 
        ax=ax,
        fmt='.2f'
    )
    
    ax.set_title(f'Sensitivity Analysis: Effect of Window Size and Threshold\n'
                f'Context: {context_col}, Feature: {hr_feature}')
    ax.set_xlabel('Synchronization Threshold')
    ax.set_ylabel('Window Size (windows)')
    
    return fig 