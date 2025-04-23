# Heart Rate Synchronization Analysis

This project analyzes heart rate synchronization between pairs of participants during conversations, examining how audio features correlate with synchronization patterns.

## Overview

This codebase provides tools to analyze how continuous audio features in conversation (such as pitch, intensity, speech rate) relate to heart rate synchronization between participants. The analysis pipeline includes:

1. **Preprocessing** heart rate data from multiple participants
2. **Feature extraction** from audio recordings
3. **Synchronization calculation** between participants' heart rates
4. **Correlation analysis** between audio features and synchronization
5. **Visualization** of results through various plots
6. **Statistical analysis** to validate findings

## Project Structure

```
project/
├── data/                # Raw and processed data
├── results/             # Output results and visualizations
├── src/                 # Source code
│   ├── preprocess.py    # Heart rate data preprocessing
│   ├── features.py      # Audio feature extraction
│   ├── synchronization.py # Heart rate synchronization calculation
│   ├── context.py       # Analysis of audio features
│   ├── visualize.py     # Visualization functions
│   ├── analysis.py      # Statistical analysis functions
│   └── explore.py       # Exploration script
└── README.md            # This file
```

## Key Components

### Preprocessing
- Cleans and processes raw heart rate data
- Handles missing values and aligns time series
- Segments data into analysis windows

### Feature Extraction
- Extracts audio features from conversation recordings
- Includes acoustic features like pitch, intensity, and speech rate
- Aligns features with heart rate data windows

### Synchronization
- Calculates correlation-based synchronization between participants' heart rates
- Offers multiple metrics (mean, standard deviation, min, max)
- Supports windowed analysis for time-based patterns

### Context Analysis
- Analyzes relationships between audio features and synchronization
- Includes correlation analysis and statistical validation
- Identifies significant audio features that predict synchronization

### Visualization
- Creates timeline plots showing heart rates and synchronization
- Generates correlation heatmaps between features and synchronization
- Produces bar charts with correlation values and confidence intervals
- Visualizes audio feature profiles over time

### Statistical Analysis
- Performs statistical tests to validate findings
- Includes permutation tests, cross-validation, and sensitivity analysis
- Calculates effect sizes and applies multiple testing correction

## Usage

### Data Format
The system expects heart rate data for each participant in CSV format with timestamps. Audio features should be extracted and stored in a pickled DataFrame.

### Example Workflow

1. **Preprocess heart rate data**:
   ```python
   from src.preprocess import preprocess_heart_rate_data
   p1_data = preprocess_heart_rate_data('data/p1_raw.csv')
   p2_data = preprocess_heart_rate_data('data/p2_raw.csv')
   ```

2. **Extract audio features**:
   ```python
   from src.features import extract_audio_features
   features_df = extract_audio_features('data/audio.wav', window_size=30)
   ```

3. **Calculate synchronization**:
   ```python
   from src.synchronization import calculate_windowwise_correlation, continuous_synchronization
   window_corr = calculate_windowwise_correlation(features_df, ['hr_mean', 'hr_std'])
   sync_results = continuous_synchronization(window_corr, ['hr_mean', 'hr_std'])
   ```

4. **Run correlation analysis**:
   ```python
   from src.context import analyze_feature_synchronization
   feature_sync_results = analyze_feature_synchronization(features_df, sync_results)
   ```

5. **Visualize results**:
   ```python
   from src.visualize import plot_correlation_heatmap
   plot_correlation_heatmap(feature_sync_results)
   ```

6. **Run exploration script**:
   ```
   python -m src.explore --session session1
   ```

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- scipy
- scikit-learn

## Contribution

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request 