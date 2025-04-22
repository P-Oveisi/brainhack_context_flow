"""
Heart Rate Synchronization Analysis - Main Script

This script runs the complete analysis pipeline for heart rate synchronization
between two participants in relation to environmental audio context.
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import project modules
from src.preprocess import preprocess_heart_rate, preprocess_audio, segment_data, align_multimodal_data
from src.features import extract_all_features
from src.synchronization import analyze_synchronization
from src.context import classify_audio_contexts, create_multidimensional_contexts, analyze_context_synchronization
from src.analysis import (compute_effect_sizes, apply_multiple_testing_correction, create_context_profile,
                          summarize_significant_contexts, permutation_test, perform_cross_validation,
                          sensitivity_analysis)
from src.visualize import (plot_timeline, plot_context_heatmap, plot_effect_sizes,
                           plot_context_profile, plot_lag_analysis, plot_sensitivity_heatmap)

# Import configuration
import config


def setup_logging():
    """Set up logging configuration."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'analysis_{timestamp}.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('heart_rate_sync')


def ensure_directories():
    """Ensure all required directories exist."""
    for _, directory in config.DATA_CONFIG.items():
        os.makedirs(directory, exist_ok=True)
    
    # Create results directory
    os.makedirs(config.DATA_CONFIG['results_dir'], exist_ok=True)
    
    # Create plots directory inside results
    os.makedirs(os.path.join(config.DATA_CONFIG['results_dir'], 'plots'), exist_ok=True)


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
    Run the complete analysis pipeline.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with all extracted features
        
    Returns:
    --------
    dict
        Dictionary with analysis results
    """
    logger.info("Starting analysis...")
    
    # Step 1: Analyze synchronization
    logger.info("Analyzing synchronization...")
    sync_results = analyze_synchronization(features_df)
    
    # Step 2: Classify audio contexts
    logger.info("Classifying audio contexts...")
    context_df = classify_audio_contexts(features_df)
    
    # Step 3: Create multi-dimensional contexts
    multi_context_df = create_multidimensional_contexts(context_df)
    
    # Step 4: Analyze context-synchronization relationships
    logger.info("Analyzing context-synchronization relationships...")
    context_sync_results = analyze_context_synchronization(multi_context_df, sync_results)
    
    # Step 5: Compute effect sizes
    logger.info("Computing effect sizes...")
    effect_sizes_df = compute_effect_sizes(context_sync_results)
    
    # Step 6: Apply multiple testing correction
    corrected_results = apply_multiple_testing_correction(
        effect_sizes_df, 
        method=config.STATS_CONFIG['multiple_testing_correction']
    )
    
    # Step 7: Summarize significant contexts
    sig_results = summarize_significant_contexts(corrected_results)
    
    # Save results
    results_path = os.path.join(config.DATA_CONFIG['results_dir'], 'analysis_results.csv')
    corrected_results.to_csv(results_path, index=False)
    
    sig_results_path = os.path.join(config.DATA_CONFIG['results_dir'], 'significant_results.csv')
    sig_results.to_csv(sig_results_path, index=False)
    
    logger.info(f"Analysis results saved to {results_path}")
    
    return {
        'sync_results': sync_results,
        'context_df': multi_context_df,
        'context_sync_results': context_sync_results,
        'effect_sizes': corrected_results,
        'significant_results': sig_results
    }


def create_visualizations(features_df, analysis_results):
    """
    Create visualizations of the analysis results.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with all extracted features
    analysis_results : dict
        Dictionary with analysis results
    """
    logger.info("Creating visualizations...")
    
    plots_dir = os.path.join(config.DATA_CONFIG['results_dir'], 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract results
    sync_results = analysis_results['sync_results']
    context_df = analysis_results['context_df']
    context_sync_results = analysis_results['context_sync_results']
    effect_sizes = analysis_results['effect_sizes']
    
    # Plot 1: Timeline for each heart rate feature
    for hr_feature in config.FEATURE_CONFIG['heart_rate_features']:
        logger.info(f"Creating timeline plot for {hr_feature}...")
        
        # Check if feature exists in results
        if hr_feature in sync_results['binary_synchronization']:
            fig = plot_timeline(
                features_df,
                sync_results['window_correlation'],
                sync_results['binary_synchronization'][hr_feature],
                hr_feature=hr_feature,
                figsize=config.VIZ_CONFIG['timeline_figsize']
            )
            
            # Save plot
            for fmt in config.VIZ_CONFIG['plot_formats']:
                path = os.path.join(plots_dir, f'timeline_{hr_feature}.{fmt}')
                fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    # Plot 2: Context heatmap for each heart rate feature
    for hr_feature in config.FEATURE_CONFIG['heart_rate_features']:
        logger.info(f"Creating context heatmap for {hr_feature}...")
        
        if hr_feature in context_sync_results:
            fig = plot_context_heatmap(
                context_sync_results,
                effect_sizes,
                hr_feature,
                figsize=config.VIZ_CONFIG['heatmap_figsize']
            )
            
            if fig is not None:
                for fmt in config.VIZ_CONFIG['plot_formats']:
                    path = os.path.join(plots_dir, f'heatmap_{hr_feature}.{fmt}')
                    fig.savefig(path, dpi=300, bbox_inches='tight')
                plt.close(fig)
    
    # Plot 3: Effect sizes
    logger.info("Creating effect size plot...")
    fig = plot_effect_sizes(
        effect_sizes,
        figsize=config.VIZ_CONFIG['effect_sizes_figsize']
    )
    
    for fmt in config.VIZ_CONFIG['plot_formats']:
        path = os.path.join(plots_dir, f'effect_sizes.{fmt}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Plot 4: Context profile
    logger.info("Creating context profile plot...")
    context_cols = [col for col in context_df.columns if col.startswith('context_')]
    profile_df = create_context_profile(context_df, context_cols)
    
    fig = plot_context_profile(
        profile_df,
        figsize=config.VIZ_CONFIG['context_profile_figsize']
    )
    
    for fmt in config.VIZ_CONFIG['plot_formats']:
        path = os.path.join(plots_dir, f'context_profile.{fmt}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Plot 5: Time lag analysis
    logger.info("Creating time lag analysis plot...")
    fig = plot_lag_analysis(
        sync_results['lag_results'],
        figsize=config.VIZ_CONFIG['lag_analysis_figsize']
    )
    
    for fmt in config.VIZ_CONFIG['plot_formats']:
        path = os.path.join(plots_dir, f'lag_analysis.{fmt}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info("Visualization complete.")


def run_validation(features_df, analysis_results):
    """
    Run validation analyses.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with all extracted features
    analysis_results : dict
        Dictionary with analysis results
        
    Returns:
    --------
    dict
        Dictionary with validation results
    """
    logger.info("Running validation analyses...")
    
    # Extract results
    context_df = analysis_results['context_df']
    sig_results = analysis_results['significant_results']
    
    validation_results = {
        'permutation_tests': [],
        'cross_validation': [],
        'sensitivity_analysis': []
    }
    
    # If no significant results, skip validation
    if len(sig_results) == 0:
        logger.info("No significant results to validate.")
        return validation_results
    
    # Get top 3 most significant context-feature pairs
    top_pairs = []
    for _, row in sig_results.head(3).iterrows():
        top_pairs.append((row['feature'], row['context']))
    
    # Run permutation tests
    logger.info("Running permutation tests...")
    for feature, context in top_pairs:
        # Get binary synchronization series
        sync_series = analysis_results['sync_results']['binary_synchronization'].get(feature)
        
        if sync_series is not None and len(sync_series) > 0:
            obs_diff, p_value = permutation_test(
                context_df, 
                sync_series, 
                context,
                n_permutations=config.STATS_CONFIG['permutation_tests']
            )
            
            validation_results['permutation_tests'].append({
                'feature': feature,
                'context': context,
                'observed_diff': obs_diff,
                'p_value': p_value
            })
    
    # Run cross-validation
    logger.info("Running cross-validation...")
    for feature, context in top_pairs:
        # Create synchronization column in context_df for cross-validation
        sync_series = analysis_results['sync_results']['binary_synchronization'].get(feature)
        
        if sync_series is not None and len(sync_series) > 0:
            sync_col = f'sync_{feature}'
            df_for_cv = context_df.copy()
            df_for_cv[sync_col] = sync_series.reset_index(drop=True)
            
            train_diff, test_diff, generalizes = perform_cross_validation(
                df_for_cv,
                context,
                sync_col,
                test_size=config.STATS_CONFIG['cross_validation_test_size']
            )
            
            validation_results['cross_validation'].append({
                'feature': feature,
                'context': context,
                'train_diff': train_diff,
                'test_diff': test_diff,
                'generalizes': generalizes
            })
    
    # Run sensitivity analysis for the top context-feature pair
    if len(top_pairs) > 0:
        logger.info("Running sensitivity analysis...")
        feature, context = top_pairs[0]
        
        sens_results = sensitivity_analysis(
            features_df,
            config.SENSITIVITY_CONFIG['window_sizes'],
            config.SENSITIVITY_CONFIG['sync_thresholds'],
            context,
            feature
        )
        
        validation_results['sensitivity_analysis'] = sens_results
        
        # Visualize sensitivity analysis
        fig = plot_sensitivity_heatmap(
            sens_results,
            context,
            feature,
            figsize=config.VIZ_CONFIG['sensitivity_figsize']
        )
        
        plots_dir = os.path.join(config.DATA_CONFIG['results_dir'], 'plots')
        for fmt in config.VIZ_CONFIG['plot_formats']:
            path = os.path.join(plots_dir, f'sensitivity_analysis.{fmt}')
            fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Save validation results
    validation_path = os.path.join(config.DATA_CONFIG['results_dir'], 'validation_results.csv')
    
    # Save permutation tests
    if validation_results['permutation_tests']:
        pd.DataFrame(validation_results['permutation_tests']).to_csv(
            os.path.join(config.DATA_CONFIG['results_dir'], 'permutation_tests.csv'),
            index=False
        )
    
    # Save cross-validation
    if validation_results['cross_validation']:
        pd.DataFrame(validation_results['cross_validation']).to_csv(
            os.path.join(config.DATA_CONFIG['results_dir'], 'cross_validation.csv'),
            index=False
        )
    
    # Save sensitivity analysis
    if isinstance(validation_results['sensitivity_analysis'], pd.DataFrame):
        validation_results['sensitivity_analysis'].to_csv(
            os.path.join(config.DATA_CONFIG['results_dir'], 'sensitivity_analysis.csv'),
            index=False
        )
    
    logger.info("Validation complete.")
    return validation_results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run heart rate synchronization analysis.')
    
    parser.add_argument('--hr1', required=True, help='Path to heart rate file for participant 1')
    parser.add_argument('--hr2', required=True, help='Path to heart rate file for participant 2')
    parser.add_argument('--audio', required=True, help='Path to audio file')
    parser.add_argument('--skip-validation', action='store_true', help='Skip validation analyses')
    
    return parser.parse_args()


def main():
    """Main function to run the complete analysis pipeline."""
    args = parse_args()
    
    # Ensure all directories exist
    ensure_directories()
    
    # Load data
    features_df, window_timestamps = load_data(args.hr1, args.hr2, args.audio)
    
    # Run analysis
    analysis_results = run_analysis(features_df)
    
    # Create visualizations
    create_visualizations(features_df, analysis_results)
    
    # Run validation (optional)
    if not args.skip_validation:
        validation_results = run_validation(features_df, analysis_results)
    
    logger.info("Analysis pipeline completed successfully!")


if __name__ == "__main__":
    # Set up logging
    logger = setup_logging()
    
    try:
        main()
    except Exception as e:
        logger.exception(f"Error in main execution: {e}")
        raise 