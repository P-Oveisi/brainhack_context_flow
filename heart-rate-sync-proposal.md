# Heart Rate Synchronization in Dyads: Environmental Audio Context Analysis

## 1. Project Overview

### 1.1 Introduction
This project aims to investigate the relationship between environmental audio context and heart rate synchronization between two individuals. Using smartwatch data and environmental audio recordings, we will identify patterns of physiological synchrony and correlate them with acoustic features to determine under what contexts synchronization is most likely to occur.

### 1.2 Research Question
**Primary Question:** Under what environmental audio contexts does heart rate synchronization appear between two people?

### 1.3 Significance
Understanding the conditions under which physiological synchronization occurs can provide insights into:
- Social dynamics and non-verbal communication
- Environmental influences on physiological processes
- Potential applications in team dynamics, therapeutic settings, and interpersonal relationships

## 2. Methodology

### 2.1 Data Collection Requirements

#### 2.1.1 Heart Rate Data
- **Source:** Consumer smartwatches worn by two participants
- **Sampling Rate:** Minimum 1 Hz (1 sample per second)
- **Format:** Time-series data with timestamps
- **Duration:** Multiple sessions of at least 30 minutes each
- **Additional Metadata:** Age, gender, relationship between participants (e.g., strangers, friends, partners)

#### 2.1.2 Audio Data
- **Source:** Environmental audio recording device
- **Sampling Rate:** 44.1 kHz
- **Format:** WAV or MP3
- **Placement:** Central location between participants
- **Synchronization:** Must have precise timestamp alignment with heart rate data

### 2.2 Data Preprocessing

#### 2.2.1 Heart Rate Data Preprocessing
1. Clean outliers (values outside physiological range or sensor errors)
2. Interpolate missing values (if any)
3. Align timestamps between participants
4. Segment into 30-second non-overlapping windows

#### 2.2.2 Audio Data Preprocessing
1. Convert to mono if stereo
2. Normalize amplitude
3. Filter background noise (minimal filtering to preserve environmental context)
4. Segment into the same 30-second non-overlapping windows as heart rate data

### 2.3 Feature Extraction (per 30-second window)

#### 2.3.1 Heart Rate Features

For each participant (p1, p2), extract the following features:

1. **Mean Heart Rate (HR_mean):**
   - Average heart rate in beats per minute within the 30-second window
   - Formula: `HR_mean = sum(heart_rate_values) / number_of_samples`

2. **Heart Rate Standard Deviation (HR_std):**
   - Standard deviation of heart rate within the window
   - Formula: `HR_std = sqrt(sum((heart_rate_values - HR_mean)^2) / number_of_samples)`

3. **Heart Rate Trend (HR_trend):**
   - Slope of the linear regression line fit to heart rate values over time
   - Formula: Use linear regression to find slope: `heart_rate ~ time`

4. **Maximum Heart Rate Change (HR_max_delta):**
   - Maximum absolute difference between consecutive heart rate values
   - Formula: `HR_max_delta = max(|heart_rate[i+1] - heart_rate[i]|)` for all i

5. **HRV Index (RMSSD):**
   - Root Mean Square of Successive Differences between adjacent RR intervals
   - Formula: `RMSSD = sqrt(mean((RR_intervals[i+1] - RR_intervals[i])^2))` for all i
   - Note: If RR intervals are not directly available, derive from heart rate values

#### 2.3.2 Audio Features

For each audio window, extract:

1. **RMS Energy (audio_energy):**
   - Root Mean Square Energy, representing loudness
   - Formula: `audio_energy = sqrt(mean(amplitude^2))`

2. **Zero-Crossing Rate (audio_zcr):**
   - Rate at which the signal changes from positive to negative or vice versa
   - Formula: `audio_zcr = (1/(N-1)) * sum(|sign(x[n]) - sign(x[n-1])|) / 2` for all n

3. **Spectral Centroid (audio_centroid):**
   - Center of gravity of the spectrum; indicates brightness/sharpness
   - Formula: `audio_centroid = sum(f[k] * X[k]) / sum(X[k])` where X[k] is the magnitude of bin k and f[k] is the center frequency of bin k

4. **Spectral Flux (audio_flux):**
   - How quickly the power spectrum changes from frame to frame
   - Formula: `audio_flux = sum((X_t[k] - X_{t-1}[k])^2)` for all k

5. **Voice Activity Ratio (audio_voice_ratio):**
   - Proportion of frames containing human voice
   - Implementation: Use a simple voice activity detector (VAD) based on energy and zero-crossing rate thresholds
   - Formula: `audio_voice_ratio = (number_of_voice_frames / total_frames)`

### 2.4 Synchronization Analysis

#### 2.4.1 Pairwise Correlation
For each heart rate feature and each 30-second window:
1. Calculate Pearson correlation coefficient between participant 1 and participant 2
2. Formula: `r = cov(p1_feature, p2_feature) / (std(p1_feature) * std(p2_feature))`

#### 2.4.2 Synchronization Definition
1. **Binary Synchronization:**
   - Define a threshold for each feature (e.g., r > 0.6)
   - Mark windows as "synchronized" or "not synchronized" based on threshold
   - Create a binary time series: sync_binary[window] = 1 if synchronized, 0 if not

2. **Weighted Synchronization:**
   - Use the actual correlation coefficient as a continuous measure of synchronization strength
   - Create a continuous time series: sync_strength[window] = correlation_coefficient

### 2.5 Context Classification

#### 2.5.1 Audio Context Categories
Define the following binary categories based on the 5 audio features:

1. **Loudness Context:**
   - "Quiet" if audio_energy < energy_threshold
   - "Loud" if audio_energy ≥ energy_threshold
   - Determine energy_threshold adaptively based on distribution (e.g., median)

2. **Frequency Content Context:**
   - "Low Frequency Dominant" if audio_centroid < centroid_threshold
   - "High Frequency Dominant" if audio_centroid ≥ centroid_threshold
   - Determine centroid_threshold based on distribution

3. **Stability Context:**
   - "Steady" if audio_flux < flux_threshold
   - "Changing" if audio_flux ≥ flux_threshold
   - Determine flux_threshold based on distribution

4. **Voice Context:**
   - "Voice Present" if audio_voice_ratio ≥ voice_threshold (e.g., 0.3)
   - "Voice Absent" if audio_voice_ratio < voice_threshold

5. **Complexity Context:**
   - "Simple" if audio_zcr < zcr_threshold
   - "Complex" if audio_zcr ≥ zcr_threshold
   - Determine zcr_threshold based on distribution

#### 2.5.2 Multi-dimensional Context
- Create compound contexts by combining binary categories
- Example: "Quiet + Voice Present + Steady" vs. "Loud + Voice Absent + Changing"

### 2.6 Context-Synchronization Analysis

#### 2.6.1 Direct Association
1. Calculate the proportion of synchronized windows within each context category
2. Formula: `P(sync|context) = count(sync=1 & context) / count(context)`
3. Compare proportions between contexts to identify conditions with higher synchronization

#### 2.6.2 Statistical Testing
1. Chi-square test for independence between context categories and synchronization
2. Point-biserial correlation between continuous audio features and binary synchronization
3. Pearson correlation between continuous audio features and synchronization strength

#### 2.6.3 Time-lag Analysis
1. Test for temporal precedence (does context change precede synchronization?)
2. Cross-correlation with varying lags (e.g., -3 to +3 windows)
3. Identify the lag with maximum correlation for each audio feature

## 3. Implementation Plan

### 3.1 Code Structure

```
project/
├── data/
│   ├── raw/                  # Raw data files
│   ├── processed/            # Preprocessed data
│   └── features/             # Extracted features
├── src/
│   ├── preprocess.py         # Data preprocessing
│   ├── features.py           # Feature extraction
│   ├── synchronization.py    # Synchronization analysis
│   ├── context.py            # Context classification
│   ├── analysis.py           # Context-synchronization analysis
│   └── visualize.py          # Visualization functions
├── notebooks/
│   ├── exploration.ipynb     # Data exploration
│   └── results.ipynb         # Results and findings
├── config.py                 # Configuration parameters
└── main.py                   # Main execution script
```

### 3.2 Processing Pipeline

1. **Input:**
   - Heart rate time series for two participants
   - Environmental audio recording
   - Optional metadata

2. **Processing Steps:**
   - Preprocess heart rate and audio data
   - Segment into 30-second windows
   - Extract heart rate and audio features
   - Calculate synchronization metrics
   - Classify audio contexts
   - Analyze context-synchronization relationships

3. **Output:**
   - Time series of feature values
   - Synchronization events
   - Context classifications
   - Statistical associations
   - Visualizations

### 3.3 Library Requirements

- **Data Processing:** numpy, pandas
- **Audio Analysis:** librosa
- **Statistical Analysis:** scipy, statsmodels
- **Visualization:** matplotlib, seaborn
- **Heart Rate Analysis:** heartpy or custom functions

## 4. Visualization and Reporting

### 4.1 Key Visualizations

1. **Timeline Plot:**
   - X-axis: Time (30-second windows)
   - Top panel: Heart rates of both participants
   - Middle panel: Synchronization strength
   - Bottom panel: Audio feature values
   - Color-coding for context categories

2. **Context-Synchronization Heatmap:**
   - X-axis: Audio feature bins
   - Y-axis: Heart rate feature bins
   - Color: Synchronization probability

3. **Bar Charts:**
   - X-axis: Context categories
   - Y-axis: Proportion of synchronization
   - Error bars for confidence intervals

### 4.2 Statistical Reports

1. **Correlation Matrix:**
   - Correlations between all heart rate features, audio features, and synchronization

2. **Context Effect Sizes:**
   - For each context, effect size of association with synchronization
   - 95% confidence intervals

3. **Time-Lag Analysis:**
   - Optimal lag for maximum correlation
   - Correlation value at optimal lag

## 5. Evaluation and Validation

### 5.1 Validation Methods

1. **Permutation Testing:**
   - Randomly shuffle time windows to break true temporal relationships
   - Compare observed associations with null distribution

2. **Cross-Validation:**
   - Split data into training and testing sets
   - Verify if context-synchronization relationships generalize

3. **Sensitivity Analysis:**
   - Vary window size (e.g., 15s, 30s, 60s)
   - Vary synchronization thresholds
   - Verify robustness of findings

### 5.2 Success Criteria

1. **Statistical significance:** Context-synchronization associations with p < 0.05
2. **Effect size:** Medium to large effect sizes (Cohen's d > 0.5)
3. **Consistency:** Findings replicate across multiple sessions/participant pairs
4. **Interpretability:** Clear patterns that can be explained theoretically

## 6. Limitations and Considerations

### 6.1 Technical Limitations

1. **Smartwatch Accuracy:**
   - Consumer-grade smartwatches may have limited precision for heart rate
   - Mitigate by using longer time windows (30s instead of shorter)

2. **Audio Context Simplification:**
   - Basic acoustic features capture limited aspects of environment
   - Using just 5 features may miss complex contextual elements

3. **Temporal Resolution:**
   - 30-second windows may miss brief synchronization events
   - Trade-off between temporal precision and feature reliability

### 6.2 Analytical Considerations

1. **Correlation vs. Causation:**
   - Observed associations don't necessarily imply causal relationships
   - Time-lag analysis provides some evidence of temporal precedence

2. **Individual Differences:**
   - Participants may vary in baseline physiology and reactivity
   - Consider normalized features or individual-specific baselines

3. **Context Interactions:**
   - Audio contexts may interact with non-recorded factors (visual cues, etc.)
   - Interpret findings as partial explanation of synchronization phenomena

## 7. Future Directions

1. **Expanded Feature Set:**
   - Include more sophisticated HRV metrics
   - Add semantic audio features (speech content, emotion)

2. **Multi-modal Contexts:**
   - Incorporate visual information
   - Add movement/activity data

3. **Machine Learning Approaches:**
   - Train models to predict synchronization from context
   - Identify complex, non-linear relationships

## 8. Conclusion

This project provides a structured framework for analyzing heart rate synchronization between two people in relation to environmental audio context. By systematically extracting features, defining synchronization, and classifying contexts, we aim to identify specific acoustic conditions that correlate with physiological synchrony. The findings will contribute to our understanding of interpersonal physiological dynamics and environmental influences on human physiology.
