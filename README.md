# Heart Rate Synchronization Analysis

This project analyzes the relationship between environmental audio context and heart rate synchronization between two individuals (dyads).

## Project Overview

The primary research question is: **Under what environmental audio contexts does heart rate synchronization appear between two people?**

This analysis pipeline processes heart rate data from two participants along with environmental audio recordings to identify patterns of physiological synchrony and correlate them with acoustic features to determine under what contexts synchronization is most likely to occur.

## Requirements

### Dependencies

Install all required packages:

```bash
pip install numpy pandas scipy matplotlib seaborn librosa scikit-learn statsmodels
```

### Data Format

1. **Heart Rate Data**
   - CSV files with at least 'timestamp' and 'heart_rate' columns
   - Timestamp should be in UNIX time (seconds since epoch)
   - Heart rate in beats per minute (BPM)

2. **Audio Data**
   - Common audio format (WAV, MP3)
   - The audio recording should temporally align with the heart rate data

## Project Structure

```
.
├── data/
│   ├── raw/                  # Raw data files
│   ├── processed/            # Preprocessed data
│   ├── features/             # Extracted features
│   └── results/              # Analysis results and figures
├── src/
│   ├── preprocess.py         # Data preprocessing
│   ├── features.py           # Feature extraction
│   ├── synchronization.py    # Synchronization analysis
│   ├── context.py            # Context classification
│   ├── analysis.py           # Context-synchronization analysis
│   └── visualize.py          # Visualization functions
├── logs/                     # Log files
├── config.py                 # Configuration parameters
├── main.py                   # Main execution script
└── README.md                 # This file
```

## Usage

Run the complete analysis pipeline:

```bash
python main.py --hr1 [path_to_participant1_heart_rate_file] --hr2 [path_to_participant2_heart_rate_file] --audio [path_to_audio_file]
```

Optional flags:
- `--skip-validation`: Skip validation analyses (permutation tests, cross-validation, sensitivity analysis)

Example:
```bash
python main.py --hr1 data/raw/participant1_hr.csv --hr2 data/raw/participant2_hr.csv --audio data/raw/session_audio.wav
```

## Analysis Pipeline

1. **Data Preprocessing**
   - Clean heart rate data (remove outliers, interpolate missing values)
   - Normalize and filter audio data
   - Segment data into 30-second non-overlapping windows

2. **Feature Extraction**
   - Heart Rate Features: mean, standard deviation, trend, max delta, RMSSD
   - Audio Features: energy, zero-crossing rate, spectral centroid, spectral flux, voice activity ratio

3. **Synchronization Analysis**
   - Calculate correlation between participants for each heart rate feature
   - Define binary synchronization based on correlation thresholds
   - Analyze time-lagged correlations

4. **Context Classification**
   - Classify audio into binary contexts: loud/quiet, high/low frequency, changing/steady, voice present/absent, complex/simple
   - Create compound contexts by combining binary categories

5. **Context-Synchronization Analysis**
   - Analyze relationships between audio contexts and heart rate synchronization
   - Calculate effect sizes and apply multiple testing correction
   - Identify significant context-synchronization associations

6. **Visualization**
   - Timeline plots of heart rates and synchronization
   - Context-synchronization heatmaps
   - Effect size bar charts
   - Context profile timelines

7. **Validation**
   - Permutation testing
   - Cross-validation
   - Sensitivity analysis

## Results

Analysis results are saved in the `data/results/` directory:
- CSV files with statistical results
- Visualizations in the `data/results/plots/` directory
- Logs of the analysis process in the `logs/` directory

## Configuration

Adjust parameters in `config.py` to customize the analysis:
- Data processing parameters (window size, sample rates)
- Feature extraction parameters
- Synchronization thresholds
- Statistical analysis settings
- Visualization options

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite:

```
@misc{heart-rate-sync,
  author = {Your Name},
  title = {Heart Rate Synchronization Analysis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/heart-rate-sync}
}
``` 