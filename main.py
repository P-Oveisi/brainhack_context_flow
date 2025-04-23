"""
Heart Rate Synchronization - Main Execution Script

This script coordinates the overall processing pipeline:
1. Load and preprocess data
2. Extract features
3. Calculate synchronization
4. Analyze features and synchronization
5. Generate visualizations
"""

import os
import argparse
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import config
from src.preprocess import align_multimodal_data, segment_data
from src.features import extract_all_features
from src.synchronization import calculate_synchronization
from src.context import prepare_audio_features, analyze_feature_synchronization, correlation_analysis
from src.analysis import create_feature_profile, calculate_time_lag_correlations, run_cross_validation
from src.visualize import (
    plot_timeline, plot_correlation_heatmap, plot_correlation_values, 
    plot_feature_profiles, plot_lag_analysis, plot_sensitivity_heatmap
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/heart_rate_sync.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Heart Rate Synchronization Analysis')
    parser.add_argument('--hr1', required=True, help='Path to heart rate file for participant 1')
    parser.add_argument('--hr2', required=True, help='Path to heart rate file for participant 2')
    parser.add_argument('--audio', required=True, help='Path to audio file')
    parser.add_argument('--output', default='data/results', help='Output directory')
    return parser.parse_args()


def load_data(hr_file1, hr_file2, audio_file):
    """
    Load and prepare the data for analysis.
    
    Parameters:
    -----------
    hr_file1 : str
        Path to heart rate file for participant 1
    hr_file2 : str
        Path to heart rate file for participant 2
    audio_file : str
        Path to audio file
        
    Returns:
    --------
    tuple
        (features_df, window_timestamps)
    """
    logger.info("Loading data...")
    
    # Load and align data
    hr_files = [hr_file1, hr_file2]
    participant_ids = ['p1', 'p2']
    
    hr_data_list, audio_data, audio_sr = align_multimodal_data(hr_files, audio_file, participant_ids)
    
    # Segment data
    logger.info("Segmenting data...")
    hr_windows1, hr_windows2, audio_windows, window_timestamps = segment_data(
        hr_data_list[0], hr_data_list[1], audio_data, audio_sr,
        window_size=config.PROCESSING_CONFIG['segmentation']['window_size'],
        hop_size=config.PROCESSING_CONFIG['segmentation']['hop_size']
    )
    
    # Extract features
    logger.info("Extracting features...")
    features_df = extract_all_features(hr_windows1, hr_windows2, audio_windows, audio_sr, window_timestamps)
    
    # Save features
    features_path = os.path.join(config.DATA_CONFIG['features_dir'], 'all_features.csv')
    features_df.to_csv(features_path, index=False)
    logger.info(f"Features saved to {features_path}")
    
    return features_df, window_timestamps


def run_analysis(features_df):
    """
    Run the heart rate synchronization analysis.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with features
        
    Returns:
    --------
    dict
        Dictionary with analysis results
    """
    logger.info("Calculating synchronization...")
    sync_results = calculate_synchronization(features_df)
    
    # Prepare audio features (normalization, etc.)
    logger.info("Processing audio features...")
    processed_features_df = prepare_audio_features(features_df)
    
    # Analyze feature-synchronization relationships
    logger.info("Analyzing feature-synchronization relationships...")
    feature_sync_results = analyze_feature_synchronization(processed_features_df, sync_results)
    
    # Calculate correlation between audio features and synchronization
    logger.info("Calculating correlations...")
    correlations_df = correlation_analysis(processed_features_df, sync_results['window_correlations'])
    
    # Calculate time-lag correlations
    logger.info("Calculating time-lag correlations...")
    lag_results = calculate_time_lag_correlations(
        processed_features_df, 
        sync_results['window_correlations'],
        max_lag=config.SYNC_CONFIG['max_lag']
    )
    
    # Save analysis results
    results = {
        'sync_results': sync_results,
        'feature_sync_results': feature_sync_results,
        'correlations': correlations_df.to_dict(orient='records'),
        'lag_results': lag_results,
        'features_df': processed_features_df,
    }
    
    return results


def generate_visualizations(analysis_results, output_dir):
    """
    Generate visualizations from analysis results.
    
    Parameters:
    -----------
    analysis_results : dict
        Dictionary with analysis results
    output_dir : str
        Output directory for visualizations
    """
    logger.info("Generating visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract results
    sync_results = analysis_results['sync_results']
    features_df = analysis_results['features_df']
    window_corr = sync_results['window_correlations']
    
    # 1. Timeline plot
    for hr_feature in config.FEATURE_CONFIG['heart_rate_features']:
        fig = plot_timeline(
            features_df, 
            pd.DataFrame({'window_start': window_corr.index, f'corr_{hr_feature}': window_corr[hr_feature]}),
            hr_feature=hr_feature,
            figsize=config.VIZ_CONFIG['timeline_figsize']
        )
        
        plt.savefig(os.path.join(output_dir, f'timeline_{hr_feature}.png'))
        plt.close(fig)
        
    # 2. Feature profiles
    logger.info("Creating feature profiles...")
    audio_cols = [col for col in features_df.columns if col.startswith('audio_')]
    feature_profile = create_feature_profile(features_df, audio_cols)
    
    fig = plot_feature_profiles(feature_profile, figsize=config.VIZ_CONFIG['feature_profile_figsize'])
    plt.savefig(os.path.join(output_dir, 'feature_profiles.png'))
    plt.close(fig)
    
    # 3. Correlation heatmap
    logger.info("Creating correlation heatmap...")
    fig = plot_correlation_heatmap(
        analysis_results['feature_sync_results'], 
        figsize=config.VIZ_CONFIG['correlation_heatmap_figsize']
    )
    
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close(fig)
    
    # 4. Correlation plots
    logger.info("Creating correlation plots...")
    fig = plot_correlation_values(
        analysis_results['feature_sync_results'],
        figsize=config.VIZ_CONFIG['correlation_plot_figsize']
    )
    
    plt.savefig(os.path.join(output_dir, 'correlation_values.png'))
    plt.close(fig)
    
    # 5. Lag analysis
    logger.info("Creating lag analysis plot...")
    fig = plot_lag_analysis(
        analysis_results['lag_results'],
        figsize=config.VIZ_CONFIG['lag_analysis_figsize']
    )
    
    plt.savefig(os.path.join(output_dir, 'lag_analysis.png'))
    plt.close(fig)
    
    # 6. Feature-specific plots for the most significant features
    top_correlations = sorted(
        analysis_results['correlations'], 
        key=lambda x: abs(x['correlation']), 
        reverse=True
    )
    
    if top_correlations:
        top_feature = top_correlations[0]['sync_feature'].replace('corr_', '')
        top_audio = top_correlations[0]['audio_feature']
        
        logger.info(f"Creating feature-specific plots for {top_feature} and {top_audio}...")
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(
            features_df[top_audio], 
            window_corr[top_feature],
            alpha=0.7
        )
        plt.xlabel(top_audio)
        plt.ylabel(f'Synchronization ({top_feature})')
        plt.title(f'Relationship Between {top_audio} and {top_feature} Synchronization')
        plt.grid(True, alpha=0.3)
        
        # Add trendline
        z = np.polyfit(features_df[top_audio], window_corr[top_feature], 1)
        p = np.poly1d(z)
        plt.plot(
            sorted(features_df[top_audio]), 
            p(sorted(features_df[top_audio])), 
            "r--"
        )
        
        plt.savefig(os.path.join(output_dir, f'top_feature_relationship.png'))
        plt.close()
    
    logger.info("Visualization generation complete.")


def run_statistical_analysis(analysis_results):
    """
    Run additional statistical analyses.
    
    Parameters:
    -----------
    analysis_results : dict
        Dictionary with analysis results
        
    Returns:
    --------
    dict
        Dictionary with statistical results
    """
    logger.info("Running statistical analyses...")
    
    features_df = analysis_results['features_df']
    sync_results = analysis_results['sync_results']
    
    # Run cross-validation to predict synchronization from features
    logger.info("Running cross-validation...")
    cv_results = run_cross_validation(
        features_df,
        sync_results['window_correlations'],
        feature_cols=['audio_tempo', 'audio_dom_freq', 'audio_contrast', 'audio_complexity', 'audio_dynamic_range'],
        test_size=config.STATS_CONFIG['cross_validation_test_size']
    )
    
    return {
        'cross_validation': cv_results
    }


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Create directories if they don't exist
    for dir_path in config.DATA_CONFIG.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    logger.info("Starting Heart Rate Synchronization Analysis")
    
    try:
        # Load and prepare data
        features_df, window_timestamps = load_data(args.hr1, args.hr2, args.audio)
        
        # Run analysis
        analysis_results = run_analysis(features_df)
        
        # Run additional statistical analyses
        stats_results = run_statistical_analysis(analysis_results)
        
        # Generate visualizations
        generate_visualizations(analysis_results, args.output)
        
        # Save results
        results_file = os.path.join(args.output, 'analysis_results.json')
        
        # Convert complex objects to JSON-serializable format
        serializable_results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'feature_sync_correlations': analysis_results['correlations'],
            'cross_validation': stats_results['cross_validation'],
            'window_count': len(features_df)
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=4)
            
        logger.info(f"Analysis complete. Results saved to {results_file}")
        
    except Exception as e:
        logger.exception(f"Error during analysis: {str(e)}")
        raise
    

if __name__ == "__main__":
    main() 