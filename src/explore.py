#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exploration script for analyzing continuous audio features and their relationship
with heart rate synchronization.

This script loads preprocessed data, extracts continuous audio features,
and analyzes their correlation with heart rate synchronization metrics.
It generates visualizations and saves results for further analysis.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from datetime import datetime

# Import custom modules
from preprocess import load_participant_data, preprocess_heart_rate_data
from features import extract_audio_features, load_audio_features
from synchronization import calculate_windowwise_correlation, continuous_synchronization
from context import analyze_feature_synchronization
from visualize import (
    plot_timeline,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_continuous_features
)
from analysis import compute_statistics, run_permutation_test

# Define paths
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data(session_id):
    """
    Load preprocessed data for a given session.

    Parameters:
    -----------
    session_id : str
        Identifier for the session to analyze

    Returns:
    --------
    features_df : pandas.DataFrame
        DataFrame containing audio features and heart rate data
    """
    print(f"Loading data for session: {session_id}")
    
    # Check if features file exists
    features_file = DATA_DIR / f"{session_id}_features.pkl"
    hr_p1_file = DATA_DIR / f"{session_id}_p1_hr.csv"
    hr_p2_file = DATA_DIR / f"{session_id}_p2_hr.csv"
    
    if not features_file.exists():
        print(f"Features file not found: {features_file}")
        print("Please run feature extraction first.")
        sys.exit(1)
    
    if not hr_p1_file.exists() or not hr_p2_file.exists():
        print(f"Participant data files not found: {hr_p1_file} or {hr_p2_file}")
        print("Please check data directory.")
        sys.exit(1)
    
    # Load features
    features_df = pd.read_pickle(features_file)
    
    # Load participant data
    p1_data = load_participant_data(hr_p1_file)
    p2_data = load_participant_data(hr_p2_file)
    
    # Calculate heart rate synchronization
    hr_features = ['hr_mean', 'hr_std']
    window_corr = calculate_windowwise_correlation(p1_data, p2_data, window_size=30)
    sync_results = continuous_synchronization(window_corr, hr_features)
    
    # Merge synchronization results with features
    features_df = features_df.join(sync_results)
    
    print(f"Loaded data with {len(features_df)} time windows")
    print(f"Features: {features_df.columns.tolist()}")
    
    return features_df

def explore_continuous_features(features_df, save_plots=True):
    """
    Explore continuous audio features and their relationship with heart rate synchronization.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing audio features and heart rate synchronization data
    save_plots : bool, default=True
        Whether to save the generated plots
        
    Returns:
    --------
    results : dict
        Dictionary containing exploration results
    """
    print("Exploring continuous audio features correlation with heart rate synchronization")
    
    # Define heart rate synchronization features
    hr_sync_features = [col for col in features_df.columns if col.startswith('sync_')]
    
    # Define audio features (exclude heart rate and synchronization features)
    audio_features = [col for col in features_df.columns 
                     if not col.startswith('hr_') 
                     and not col.startswith('sync_')
                     and not col == 'timestamp']
    
    # Calculate correlation between audio features and synchronization
    corr_results = {}
    for sync_feature in hr_sync_features:
        corr = features_df[audio_features].corrwith(features_df[sync_feature])
        corr_results[sync_feature] = corr.sort_values(ascending=False)
    
    # Create correlation matrix for visualization
    corr_matrix = pd.DataFrame({sync_feature: corr_results[sync_feature] 
                               for sync_feature in hr_sync_features})
    
    # Run statistical analysis
    stats_results = compute_statistics(features_df, audio_features, hr_sync_features)
    
    # Create visualizations
    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir = RESULTS_DIR / "plots" / timestamp
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation between Audio Features and Heart Rate Synchronization')
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_sync_correlation_heatmap.png")
        plt.close()
        
        # Plot feature importance
        plot_feature_importance(corr_results, save_path=plots_dir / "feature_importance.png")
        
        # Plot timeline of top features
        top_features = corr_matrix.abs().mean(axis=1).sort_values(ascending=False).head(5).index.tolist()
        plot_continuous_features(features_df, top_features, hr_sync_features, 
                                save_path=plots_dir / "top_features_timeline.png")
        
        # Plot individual feature relationships
        for feature in top_features:
            for sync_feature in hr_sync_features:
                plt.figure(figsize=(10, 6))
                sns.regplot(x=feature, y=sync_feature, data=features_df)
                plt.title(f'Relationship between {feature} and {sync_feature}')
                plt.tight_layout()
                plt.savefig(plots_dir / f"relationship_{feature}_{sync_feature}.png")
                plt.close()
        
        print(f"Saved plots to {plots_dir}")
    
    # Prepare results
    results = {
        'correlation': corr_results,
        'statistics': stats_results,
        'top_features': {
            sync_feature: corr_results[sync_feature].abs().sort_values(ascending=False).head(10).to_dict()
            for sync_feature in hr_sync_features
        }
    }
    
    # Print summary of results
    print("\nTop correlations with heart rate synchronization:")
    for sync_feature in hr_sync_features:
        print(f"\n{sync_feature}:")
        top_corr = corr_results[sync_feature].abs().sort_values(ascending=False).head(5)
        for feature, value in top_corr.items():
            print(f"  {feature}: {value:.3f}")
    
    return results

def run_exploration(session_id):
    """
    Run exploration analysis for a given session.
    
    Parameters:
    -----------
    session_id : str
        Identifier for the session to analyze
        
    Returns:
    --------
    results : dict
        Dictionary containing exploration results
    """
    # Load data
    features_df = load_data(session_id)
    
    # Explore continuous features
    results = explore_continuous_features(features_df)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"{session_id}_exploration_results_{timestamp}.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nSaved exploration results to {results_file}")
    
    return results

def main():
    """Main function to run the exploration script."""
    parser = argparse.ArgumentParser(description="Explore continuous audio features and heart rate synchronization")
    parser.add_argument("--session", type=str, default="session1",
                        help="Session identifier (default: session1)")
    args = parser.parse_args()
    
    run_exploration(args.session)

if __name__ == "__main__":
    main() 