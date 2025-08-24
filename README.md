# NBA Home Win-Loss Prediction Model

A comprehensive machine learning model that predicts whether an NBA team will win or lose at home based on various team statistics and performance metrics.

## 🏀 Overview

This model analyzes historical NBA game data to predict home team outcomes using advanced machine learning algorithms. It incorporates multiple features including:

- **Team Performance Metrics**: Offensive efficiency, win percentages, scoring margins
- **Recent Form**: Rolling averages of key statistics
- **Rest Advantages**: Days of rest between games
- **Home/Away Performance**: Historical performance in different venues
- **Seasonal Trends**: Performance patterns across different seasons

## 🚀 Features

### Multiple Algorithms
- **Random Forest**: Robust tree-based ensemble method
- **Gradient Boosting**: Advanced boosting algorithm
- **Logistic Regression**: Linear classification model
- **Support Vector Machine**: Kernel-based classification
- **Ensemble Model**: Voting classifier combining best performers

### Advanced Feature Engineering
- **Advantage Metrics**: Home vs. away team performance differences
- **Strength Indicators**: Combined performance scores
- **Momentum Features**: Recent performance trends
- **Rest Advantages**: Fatigue and recovery factors

### Comprehensive Evaluation
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, AUC
- **Visualizations**: Confusion matrices, ROC curves, feature importance
- **Model Comparison**: Side-by-side performance analysis
- **Cross-validation**: Robust model evaluation

## 📊 Data Requirements

The model expects a CSV file with the following columns:

### Home Team Features
- `HOME_LAST_GAME_OE`: Last game offensive efficiency
- `HOME_LAST_GAME_HOME_WIN_PCTG`: Home win percentage
- `HOME_NUM_REST_DAYS`: Days of rest
- `HOME_LAST_GAME_AWAY_WIN_PCTG`: Away win percentage
- `HOME_LAST_GAME_TOTAL_WIN_PCTG`: Overall win percentage
- `HOME_LAST_GAME_ROLLING_SCORING_MARGIN`: Rolling scoring margin
- `HOME_LAST_GAME_ROLLING_OE`: Rolling offensive efficiency

### Away Team Features
- `AWAY_LAST_GAME_OE`: Last game offensive efficiency
- `AWAY_LAST_GAME_HOME_WIN_PCTG`: Home win percentage
- `AWAY_NUM_REST_DAYS`: Days of rest
- `AWAY_LAST_GAME_AWAY_WIN_PCTG`: Away win percentage
- `AWAY_LAST_GAME_TOTAL_WIN_PCTG`: Overall win percentage
- `AWAY_LAST_GAME_ROLLING_SCORING_MARGIN`: Rolling scoring margin
- `AWAY_LAST_GAME_ROLLING_OE`: Rolling offensive efficiency

### Target Variable
- `HOME_W`: Binary outcome (1 = home win, 0 = home loss)

## 🛠️ Installation

1. **Clone or download** the project files
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure your data file** (`nbaHomeWinLossModelDataset.csv`) is in the project directory

## 🎯 Usage

### Quick Start

Run the complete pipeline:
```bash
python nba_home_win_loss_model.py
```

This will:
1. Load and preprocess your data
2. Train multiple models
3. Evaluate performance
4. Create visualizations
5. Save the best model
6. Provide example predictions

### Custom Usage

```python
from nba_home_win_loss_model import NBAHomeWinLossModel

# Create model instance
model = NBAHomeWinLossModel()

# Load and prepare data
model.load_data('your_data.csv')
model.handle_missing_values()
model.create_features()
model.prepare_features()

# Train models
model.initialize_models()
model.train_models()
model.compare_models()

# Make predictions
home_stats = {
    'LAST_GAME_OE': 0.65,
    'HOME_WIN_PCTG': 0.70,
    'NUM_REST_DAYS': 2.0,
    'AWAY_WIN_PCTG': 0.60,
    'TOTAL_WIN_PCTG': 0.65,
    'ROLLING_SCORING_MARGIN': 5.0,
    'ROLLING_OE': 0.68
}

away_stats = {
    'LAST_GAME_OE': 0.58,
    'HOME_WIN_PCTG': 0.65,
    'NUM_REST_DAYS': 1.0,
    'AWAY_WIN_PCTG': 0.55,
    'TOTAL_WIN_PCTG': 0.60,
    'ROLLING_SCORING_MARGIN': -2.0,
    'ROLLING_OE': 0.60
}

prediction, probability = model.predict_home_win(home_stats, away_stats)
print(f"Prediction: {'WIN' if prediction == 1 else 'LOSS'}")
print(f"Win Probability: {probability:.3f}")
```

## 📈 Model Performance

The model typically achieves:
- **Accuracy**: 65-75%
- **F1-Score**: 0.65-0.75
- **AUC**: 0.70-0.80

Performance varies based on:
- Data quality and completeness
- Feature relevance
- Season-specific patterns
- Team roster changes

## 🔍 Feature Importance

Key factors influencing home team success:
1. **Home Win Percentage**: Historical home court advantage
2. **Offensive Efficiency**: Scoring ability and ball movement
3. **Rest Advantage**: Recovery time between games
4. **Recent Form**: Rolling performance averages
5. **Scoring Margin**: Point differential trends

## 📊 Output Files

The model generates:
- **Trained Model**: `nba_home_win_model.pkl`
- **Visualizations**: Confusion matrices, ROC curves, feature importance plots
- **Performance Reports**: Detailed model comparison and metrics

## 🔧 Customization

### Adding New Features
```python
def create_custom_features(df):
    # Add your custom features here
    df['CUSTOM_FEATURE'] = df['EXISTING_COL'] * 2
    return df
```

### Modifying Algorithms
```python
# Add new models to the models dictionary
self.models['XGBoost'] = XGBClassifier()
```

### Changing Evaluation Metrics
```python
# Modify the metrics calculation in train_models method
# Add custom evaluation criteria as needed
```

## 📚 Technical Details

### Data Preprocessing
- **Missing Value Handling**: Median imputation for numeric features
- **Feature Scaling**: StandardScaler for algorithms requiring normalization
- **Feature Engineering**: 15+ derived features for enhanced performance

### Time-Based Validation
- **Season-Based Split**: Training on earlier seasons, testing on later seasons
- **Data Leakage Prevention**: Ensures no future information contaminates training data
- **Realistic Evaluation**: Simulates real-world prediction scenarios
- **Split Validation**: Automatic checks for temporal ordering and season overlap

### Model Selection
- **Cross-validation**: 5-fold CV for hyperparameter tuning
- **Ensemble Methods**: Voting classifier for improved stability
- **Hyperparameter Optimization**: Grid search for best parameters

### Evaluation Framework
- **Time-Based Split**: 80/20 split by seasons to prevent data leakage
- **Multiple Metrics**: Comprehensive performance assessment
- **Statistical Significance**: Confidence intervals and error analysis

## 🚨 Limitations

1. **Historical Data**: Model performance depends on data quality
2. **Roster Changes**: Player trades/signings may affect accuracy
3. **Seasonal Variations**: Performance may vary across seasons
4. **External Factors**: Injuries, weather, travel not included
5. **Market Efficiency**: Public betting markets may already incorporate some factors

## 🔮 Future Enhancements

- **Player-level Statistics**: Individual player performance metrics
- **Injury Data**: Player availability and health status
- **Weather Factors**: Arena conditions and travel impact
- **Real-time Updates**: Live data integration
- **API Integration**: Automated data collection
- **Web Interface**: User-friendly prediction interface

## 📞 Support

For questions or issues:
1. Check the code comments and documentation
2. Review the example usage patterns
3. Ensure all dependencies are properly installed
4. Verify data format matches requirements

## 📄 License

This project is provided as-is for educational and research purposes. Feel free to modify and adapt for your specific needs.

---

**Note**: This model is designed for educational and research purposes. Always use multiple sources and professional judgment when making real-world decisions based on sports predictions.
