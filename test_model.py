#!/usr/bin/env python3
"""
Test script for the NBA Home Win-Loss Prediction Model
"""

import pandas as pd
import numpy as np
from nba_home_win_loss_model import NBAHomeWinLossModel

def test_model():
    """Test the NBA prediction model with available data"""
    print("Testing NBA Home Win-Loss Prediction Model...")
    
    try:
        # Create model instance
        model = NBAHomeWinLossModel()
        print("✓ Model instance created successfully")
        
        # Test data loading
        model.load_data()
        print("✓ Data loaded successfully")
        print(f"  Dataset shape: {model.df.shape}")
        
        # Test missing value handling
        model.handle_missing_values()
        print("✓ Missing values handled successfully")
        
        # Test feature creation
        model.create_features()
        print("✓ Features created successfully")
        print(f"  Enhanced features: {len(model.df_enhanced.columns)}")
        
        # Test feature preparation
        X, y = model.prepare_features()
        print("✓ Features prepared successfully")
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Target vector shape: {y.shape}")
        
        # Test model initialization
        model.initialize_models()
        print("✓ Models initialized successfully")
        print(f"  Models created: {list(model.models.keys())}")
        
        # Test model training
        model.train_models()
        print("✓ Models trained successfully")
        
        # Test model comparison
        comparison_df = model.compare_models()
        print("✓ Model comparison completed successfully")
        print(f"  Best model: {model.best_model_name}")
        
        # Test ensemble creation
        model.create_ensemble()
        print("✓ Ensemble model created successfully")
        
        # Test prediction function
        model.create_prediction_function()
        print("✓ Prediction function created successfully")
        
        # Test example prediction
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
        print("✓ Example prediction completed successfully")
        print(f"  Prediction: {'WIN' if prediction == 1 else 'LOSS'}")
        print(f"  Win Probability: {probability:.3f}" if probability else "Probability not available")
        
        # Test model saving
        model.save_model('test_model.pkl')
        print("✓ Model saved successfully")
        
        # Test model loading
        new_model = NBAHomeWinLossModel()
        new_model.load_model('test_model.pkl')
        print("✓ Model loaded successfully")
        
        print("\n🎉 All tests passed successfully!")
        print("\nModel is ready to use!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n✅ Model testing completed successfully!")
    else:
        print("\n❌ Model testing failed!")
