# Model Training and Testing Documentation

## Overview

This project implements a complete machine learning pipeline for predicting League of Legends match outcomes. The system uses team compositions (champion picks) and player statistics to achieve high accuracy predictions.

## Model Training Process

### 1. Data Preparation
- **Input Data**: 
  - Team compositions (10 champion picks per match)
  - Team statistics (kills, towers, dragons, barons)
  - Player statistics (KDA, gold, damage, vision)
- **Feature Engineering**:
  - One-hot encoding for 171 unique champions
  - Aggregated player stats (mean and sum) by team
  - Total features: 386 (10 team stats + 34 player stats + 342 champion encodings)

### 2. Training Pipeline
```python
# Command
python train.py --csv TeamMatchTbl.csv --summoner SummonerMatchTbl.csv --stats MatchStatsTbl.csv --nrows 500 --out model_test.joblib

# Process
1. Load and preprocess data
2. 5-fold stratified cross-validation
3. Train multiple models (Logistic Regression, Random Forest)
4. Select best model based on ROC-AUC
5. Fit best model on full training set
6. Save model pipeline to disk
```

### 3. Models Evaluated

| Model | ROC-AUC | Std Dev | Training Time |
|-------|---------|---------|---------------|
| **Logistic Regression** | **98.64%** | ±0.77% | ~5 seconds |
| Random Forest (200 trees) | 98.32% | ±0.51% | ~30 seconds |

**Winner**: Logistic Regression (better AUC, faster training)

## Model Testing & Evaluation

### Testing Process
```python
# Command
python test.py --model model_test.joblib --csv TeamMatchTbl.csv --summoner SummonerMatchTbl.csv --stats MatchStatsTbl.csv --nrows 500

# Metrics Generated
- ROC-AUC Score
- Accuracy
- Precision/Recall/F1 per class
- Classification report
```

### Performance Results (500 samples)

#### Logistic Regression
```
              precision    recall  f1-score   support

     Red Win       0.94      0.95      0.94       221
    Blue Win       0.96      0.95      0.95       279

    accuracy                           0.95       500
   macro avg       0.95      0.95      0.95       500
weighted avg       0.95      0.95      0.95       500

ROC-AUC: 0.9864
```

#### Random Forest
```
              precision    recall  f1-score   support

     Red Win       0.92      0.90      0.91       221
    Blue Win       0.93      0.94      0.93       279

    accuracy                           0.92       500
   macro avg       0.92      0.92      0.92       500
weighted avg       0.92      0.92      0.92       500

ROC-AUC: 0.9832
```

### Key Metrics Explained

- **ROC-AUC (98.64%)**: Model's ability to distinguish between Blue Win and Red Win
  - 0.5 = random guessing
  - 1.0 = perfect classification
  - Our 98.64% is excellent performance

- **Accuracy (95%)**: Percentage of correct predictions
  - 95 out of 100 matches predicted correctly

- **Precision (Blue: 96%, Red: 94%)**: When model predicts a team wins, how often is it correct?
  - Blue Win predictions are correct 96% of the time
  - Red Win predictions are correct 94% of the time

- **Recall (Blue: 95%, Red: 95%)**: Of all actual wins, how many did we catch?
  - We correctly identify 95% of actual Blue wins
  - We correctly identify 95% of actual Red wins

- **F1-Score (95%)**: Harmonic mean of precision and recall
  - Balanced metric showing overall classification quality

## Visualization & Analysis

### Generate Visualizations
```python
python visualize.py --csv TeamMatchTbl.csv --summoner SummonerMatchTbl.csv --stats MatchStatsTbl.csv --nrows 500 --output visualizations
```

### Generated Graphs

#### 1. ROC Curves (`roc_curves.png`)
- Compares model performance visually
- Shows True Positive Rate vs False Positive Rate
- Area Under Curve (AUC) quantifies performance
- **Result**: Both models achieve >98% AUC

#### 2. Confusion Matrices (`confusion_matrices.png`)
- Shows correct vs incorrect predictions
- Visualizes False Positives and False Negatives
- **Result**: 
  - Logistic Regression: 210 correct Red, 265 correct Blue
  - Random Forest: 199 correct Red, 263 correct Blue

#### 3. Cross-Validation Scores (`cv_scores_comparison.png`)
- Box plot showing score distribution across 5 folds
- Bar chart with mean ± standard deviation
- **Result**: Consistent performance across all folds (low variance)

#### 4. Feature Importance (`feature_importance.png`)
- Ranks top 20 most predictive features
- Shows which game factors most influence outcome
- **Top 5 Features**:
  1. Red Tower Kills (11.0%)
  2. Blue Kills (7.0%)
  3. Red Kills (6.7%)
  4. Blue Tower Kills (6.4%)
  5. Red Dragon Kills (4.9%)

#### 5. Class Distribution (`class_distribution.png`)
- Shows balance between Blue Win and Red Win
- **Result**: 
  - Blue Win: 279 samples (55.8%)
  - Red Win: 221 samples (44.2%)
  - Relatively balanced dataset

## Feature Importance Insights

### Top 10 Most Important Features

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | Red Tower Kills | 11.0% | Objective |
| 2 | Blue Kills | 7.0% | Combat |
| 3 | Red Kills | 6.7% | Combat |
| 4 | Blue Tower Kills | 6.4% | Objective |
| 5 | Red Dragon Kills | 4.9% | Objective |
| 6 | Red Turret Damage (mean) | 2.9% | Player Stat |
| 7 | Red Baron Kills | 2.7% | Objective |
| 8 | Red Turret Damage (sum) | 2.6% | Player Stat |
| 9 | Blue Dragon Kills | 2.3% | Objective |
| 10 | Blue Turret Damage (sum) | 1.6% | Player Stat |

### Key Findings

1. **Objectives Matter Most**: Tower kills and dragon kills are top predictors
2. **Combat Stats Important**: Kill counts for both teams are highly predictive
3. **Player Stats Add Value**: Turret damage metrics improve predictions
4. **Champion Picks**: While included, individual champion picks have lower importance than game objectives (distributed across 342 features)

## Model Strengths

✅ **High Accuracy (95%)**: Predicts match outcomes correctly 95% of the time
✅ **Excellent ROC-AUC (98.6%)**: Strong ability to distinguish between outcomes
✅ **Balanced Performance**: Similar precision/recall for both classes
✅ **Consistent CV Scores**: Low variance across folds (±0.77%)
✅ **Interpretable Features**: Clear importance rankings for game factors
✅ **Fast Training**: Logistic regression trains in seconds
✅ **Fast Prediction**: Real-time predictions for interactive use

## Model Limitations

⚠️ **Player Stats Required**: Model trained with player stats achieves best performance
⚠️ **Pre-match Predictions**: When predicting before match (no game stats), relies only on champion composition
⚠️ **Class Imbalance**: Slight imbalance toward Blue Win (55.8% vs 44.2%)
⚠️ **Limited Data**: Trained on 500 samples for demo; more data would improve generalization
⚠️ **Static Features**: Doesn't account for player skill level, patch changes, or meta shifts

## Discussion Points

### Why Logistic Regression Outperforms Random Forest?

1. **Feature Independence**: One-hot encoded champions are linearly separable
2. **Regularization**: Built-in L2 regularization prevents overfitting
3. **Balanced Classes**: Class weighting handles slight imbalance
4. **Sufficient Data**: 500 samples sufficient for linear model
5. **Less Noise**: Random Forest may overfit to noise in training data

### What Makes a Good Prediction?

1. **Objective Control**: Tower and dragon kills most predictive
2. **Kill Advantage**: Team kill counts strongly correlate with wins
3. **Sustained Damage**: Turret damage indicates map pressure
4. **Champion Synergy**: While individual picks matter less, team composition still contributes

### Real-World Application

- **Draft Analysis**: Predict match outcomes during champion select
- **Live Updates**: Update predictions as game events occur
- **Coach Tool**: Identify key objectives for winning strategy
- **Research**: Analyze which factors most influence match outcomes

## Reproducibility

All results are reproducible with:
```powershell
# Set random seed
python train.py --csv TeamMatchTbl.csv --summoner SummonerMatchTbl.csv --stats MatchStatsTbl.csv --nrows 500 --out model_test.joblib

# Generate all visualizations
python visualize.py --csv TeamMatchTbl.csv --summoner SummonerMatchTbl.csv --stats MatchStatsTbl.csv --nrows 500 --output visualizations
```

Random seed: 42 (default)

## Conclusion

The model successfully predicts League of Legends match outcomes with 98.6% ROC-AUC and 95% accuracy. Visualizations clearly show model performance, feature importance, and prediction quality. The system is ready for interactive use and can be deployed for real-time match outcome predictions.

**Key Takeaway**: Objective control (towers, dragons) and team kills are the strongest predictors of match outcome, outweighing individual champion selections.
