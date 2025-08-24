#!/usr/bin/env python3
"""
Test NBA Prediction Model on 2024-25 Season Data
Calculate real success rate on unseen data
"""

import pandas as pd
import numpy as np
from nba_home_win_loss_model import NBAHomeWinLossModel
import pickle

def test_24_25_season():
    """Test the model on 2024-25 season data"""
    print("="*60)
    print("TESTING NBA PREDICTION MODEL ON 2024-25 SEASON")
    print("="*60)
    
    # Load the trained model
    print("Loading trained model...")
    model = NBAHomeWinLossModel()
    model.load_model('nba_home_win_model.pkl')
    
    # Load the original data
    print("Loading original dataset...")
    df = pd.read_csv('nbaHomeWinLossModelDataset.csv')
    
    # Filter for 2024-25 season only
    test_data = df[df['SEASON'] == '2024-25'].copy()
    print(f"Found {len(test_data)} games in 2024-25 season")
    
    if len(test_data) == 0:
        print("No 2024-25 season data found!")
        return
    
    # Apply the same feature engineering
    print("Applying feature engineering...")
    test_data_enhanced = test_data.copy()
    
    # Create additional features manually
    test_data_enhanced['HOME_OE_ADVANTAGE'] = test_data_enhanced['HOME_LAST_GAME_ROLLING_OE'] - test_data_enhanced['AWAY_LAST_GAME_ROLLING_OE']
    test_data_enhanced['HOME_WIN_PCTG_ADVANTAGE'] = test_data_enhanced['HOME_LAST_GAME_HOME_WIN_PCTG'] - test_data_enhanced['AWAY_LAST_GAME_AWAY_WIN_PCTG']
    test_data_enhanced['HOME_SCORING_MARGIN_ADVANTAGE'] = test_data_enhanced['HOME_LAST_GAME_ROLLING_SCORING_MARGIN'] - test_data_enhanced['AWAY_LAST_GAME_ROLLING_SCORING_MARGIN']
    test_data_enhanced['REST_ADVANTAGE'] = test_data_enhanced['HOME_NUM_REST_DAYS'] - test_data_enhanced['AWAY_NUM_REST_DAYS']
    test_data_enhanced['HOME_STRENGTH'] = (test_data_enhanced['HOME_LAST_GAME_ROLLING_OE'] + test_data_enhanced['HOME_LAST_GAME_HOME_WIN_PCTG'] + test_data_enhanced['HOME_LAST_GAME_ROLLING_SCORING_MARGIN']) / 3
    test_data_enhanced['AWAY_STRENGTH'] = (test_data_enhanced['AWAY_LAST_GAME_ROLLING_OE'] + test_data_enhanced['AWAY_LAST_GAME_AWAY_WIN_PCTG'] + test_data_enhanced['AWAY_LAST_GAME_ROLLING_SCORING_MARGIN']) / 3
    test_data_enhanced['STRENGTH_DIFFERENCE'] = test_data_enhanced['HOME_STRENGTH'] - test_data_enhanced['AWAY_STRENGTH']
    test_data_enhanced['HOME_MOMENTUM'] = test_data_enhanced['HOME_LAST_GAME_ROLLING_OE'] * test_data_enhanced['HOME_LAST_GAME_HOME_WIN_PCTG']
    test_data_enhanced['AWAY_MOMENTUM'] = test_data_enhanced['AWAY_LAST_GAME_ROLLING_OE'] * test_data_enhanced['AWAY_LAST_GAME_AWAY_WIN_PCTG']
    test_data_enhanced['SEASON_NUM'] = test_data_enhanced['SEASON'].str.extract('(\d+)').astype(int)
    
    # Handle missing values (same as in training)
    print("Handling missing values...")
    numeric_columns = test_data_enhanced.select_dtypes(include=[np.number]).columns
    test_data_enhanced[numeric_columns] = test_data_enhanced[numeric_columns].fillna(test_data_enhanced[numeric_columns].median())
    
    print(f"After handling missing values, shape: {test_data_enhanced.shape}")
    
    # Prepare features for prediction
    X_test = test_data_enhanced[model.feature_columns]
    y_true = test_data_enhanced['HOME_W']
    
    print(f"Test data shape: {X_test.shape}")
    print(f"True outcomes: {y_true.value_counts().to_dict()}")
    
    # Make predictions using different approaches
    print("\n" + "="*50)
    print("MAKING PREDICTIONS ON 2024-25 SEASON")
    print("="*50)
    
    results = {}
    
    # 1. Random Forest (Best individual model)
    print("\n1. Random Forest Predictions:")
    rf_model = model.results['Random Forest']['model']
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    
    rf_accuracy = (y_pred_rf == y_true).mean()
    rf_correct = (y_pred_rf == y_true).sum()
    print(f"   Accuracy: {rf_accuracy:.4f} ({rf_correct}/{len(y_true)})")
    
    results['Random Forest'] = {
        'predictions': y_pred_rf,
        'probabilities': y_prob_rf,
        'accuracy': rf_accuracy,
        'correct': rf_correct
    }
    
    # 2. Weighted Ensemble
    print("\n2. Weighted Ensemble Predictions:")
    if 'Weighted Ensemble' in model.results:
        weights = model.results['Weighted Ensemble']['weights']
        print(f"   Using weights: {weights}")
        
        # Calculate weighted probabilities
        weighted_probs = np.zeros(len(y_true))
        for name, weight in weights.items():
            if name in model.results and 'model' in model.results[name]:
                if name in ['SVM', 'Logistic Regression']:
                    X_scaled = model.scaler.transform(X_test)
                    prob = model.results[name]['model'].predict_proba(X_scaled)[:, 1]
                else:
                    prob = model.results[name]['model'].predict_proba(X_test)[:, 1]
                weighted_probs += weight * prob
        
        y_pred_ensemble = (weighted_probs > 0.5).astype(int)
        ensemble_accuracy = (y_pred_ensemble == y_true).mean()
        ensemble_correct = (y_pred_ensemble == y_true).sum()
        
        print(f"   Accuracy: {ensemble_accuracy:.4f} ({ensemble_correct}/{len(y_true)})")
        
        results['Weighted Ensemble'] = {
            'predictions': y_pred_ensemble,
            'probabilities': weighted_probs,
            'accuracy': ensemble_accuracy,
            'correct': ensemble_correct
        }
    
    # 3. Traditional Ensemble
    print("\n3. Traditional Ensemble Predictions:")
    if 'Ensemble' in model.results:
        ensemble_model = model.results['Ensemble']['model']
        y_pred_trad = ensemble_model.predict(X_test)
        y_prob_trad = ensemble_model.predict_proba(X_test)[:, 1]
        
        trad_accuracy = (y_pred_trad == y_true).mean()
        trad_correct = (y_pred_trad == y_true).sum()
        print(f"   Accuracy: {trad_accuracy:.4f} ({trad_correct}/{len(y_true)})")
        
        results['Traditional Ensemble'] = {
            'predictions': y_pred_trad,
            'probabilities': y_prob_trad,
            'accuracy': trad_accuracy,
            'correct': trad_correct
        }
    
    # 4. All individual models
    print("\n4. Individual Model Performance:")
    for name, result in model.results.items():
        if name not in ['Ensemble', 'Weighted Ensemble'] and 'model' in result:
            try:
                if name in ['SVM', 'Logistic Regression']:
                    X_scaled = model.scaler.transform(X_test)
                    y_pred = result['model'].predict(X_scaled)
                else:
                    y_pred = result['model'].predict(X_test)
                
                accuracy = (y_pred == y_true).mean()
                correct = (y_pred == y_true).sum()
                print(f"   {name}: {accuracy:.4f} ({correct}/{len(y_true)})")
                
                results[name] = {
                    'predictions': y_pred,
                    'accuracy': accuracy,
                    'correct': correct
                }
            except Exception as e:
                print(f"   {name}: Error - {e}")
    
    # Summary
    print("\n" + "="*50)
    print("2024-25 SEASON PREDICTION SUMMARY")
    print("="*50)
    
    print(f"Total Games: {len(y_true)}")
    print(f"Home Wins: {y_true.sum()}")
    print(f"Home Losses: {len(y_true) - y_true.sum()}")
    print(f"Baseline (Always Predict Home Win): {(y_true == 1).mean():.4f}")
    
    print(f"\nModel Performance Ranking:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for i, (name, result) in enumerate(sorted_results, 1):
        print(f"{i}. {name}: {result['accuracy']:.4f} ({result['correct']}/{len(y_true)})")
    
    # Calculate profit/loss if betting (assuming $100 per bet)
    print(f"\nBetting Simulation ($100 per bet):")
    for name, result in sorted_results:
        if 'probabilities' in result:
            # Simple betting strategy: bet when probability > 0.6
            high_confidence = result['probabilities'] > 0.6
            if high_confidence.sum() > 0:
                confident_preds = result['predictions'][high_confidence]
                confident_true = y_true[high_confidence]
                confident_accuracy = (confident_preds == confident_true).mean()
                confident_games = high_confidence.sum()
                
                # Calculate profit/loss
                wins = (confident_preds == confident_true).sum()
                losses = confident_games - wins
                profit = wins * 100 * 0.91 - losses * 100  # Assuming -110 odds
                
                print(f"   {name} (High Confidence >0.6):")
                print(f"     Games: {confident_games}, Accuracy: {confident_accuracy:.4f}")
                print(f"     Profit/Loss: ${profit:.2f}")
    
    return results

if __name__ == "__main__":
    results = test_24_25_season()
