#!/usr/bin/env python3
"""
NBA Home Win-Loss Prediction Model

This script builds a comprehensive model to predict whether an NBA team will win 
or lose at home based on various team statistics and performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import pickle
import os
import time

warnings.filterwarnings('ignore')

# Set matplotlib to non-interactive backend to prevent hanging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

class NBAHomeWinLossModel:
    """NBA Home Win-Loss Prediction Model Class"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, filepath='nbaHomeWinLossModelDataset.csv'):
        """Load and explore the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(filepath)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Basic data info
        print("\nDataset Info:")
        print(self.df.info())
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        
        return self.df
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("\nHandling missing values...")
        missing_counts = self.df.isnull().sum()
        print("Missing values per column:")
        print(missing_counts[missing_counts > 0])
        
        # Fill missing values with appropriate methods
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].median())
        
        print(f"After handling missing values, shape: {self.df.shape}")
        
    def create_features(self):
        """Create additional features for better model performance"""
        print("\nCreating additional features...")
        df_copy = self.df.copy()
        
        # Home team advantage features
        df_copy['HOME_OE_ADVANTAGE'] = df_copy['HOME_LAST_GAME_ROLLING_OE'] - df_copy['AWAY_LAST_GAME_ROLLING_OE']
        df_copy['HOME_WIN_PCTG_ADVANTAGE'] = df_copy['HOME_LAST_GAME_HOME_WIN_PCTG'] - df_copy['AWAY_LAST_GAME_AWAY_WIN_PCTG']
        df_copy['HOME_SCORING_MARGIN_ADVANTAGE'] = df_copy['HOME_LAST_GAME_ROLLING_SCORING_MARGIN'] - df_copy['AWAY_LAST_GAME_ROLLING_SCORING_MARGIN']
        
        # Rest advantage
        df_copy['REST_ADVANTAGE'] = df_copy['HOME_NUM_REST_DAYS'] - df_copy['AWAY_NUM_REST_DAYS']
        
        # Overall team strength (combined metrics)
        df_copy['HOME_STRENGTH'] = (df_copy['HOME_LAST_GAME_ROLLING_OE'] + 
                                   df_copy['HOME_LAST_GAME_HOME_WIN_PCTG'] + 
                                   df_copy['HOME_LAST_GAME_ROLLING_SCORING_MARGIN']) / 3
        
        df_copy['AWAY_STRENGTH'] = (df_copy['AWAY_LAST_GAME_ROLLING_OE'] + 
                                   df_copy['AWAY_LAST_GAME_AWAY_WIN_PCTG'] + 
                                   df_copy['AWAY_LAST_GAME_ROLLING_SCORING_MARGIN']) / 3
        
        df_copy['STRENGTH_DIFFERENCE'] = df_copy['HOME_STRENGTH'] - df_copy['AWAY_STRENGTH']
        
        # Momentum features (recent performance)
        df_copy['HOME_MOMENTUM'] = df_copy['HOME_LAST_GAME_ROLLING_OE'] * df_copy['HOME_LAST_GAME_HOME_WIN_PCTG']
        df_copy['AWAY_MOMENTUM'] = df_copy['AWAY_LAST_GAME_ROLLING_OE'] * df_copy['AWAY_LAST_GAME_AWAY_WIN_PCTG']
        
        # Season encoding
        df_copy['SEASON_NUM'] = df_copy['SEASON'].str.extract('(\d+)').astype(int)
        
        self.df_enhanced = df_copy
        print(f"Original features: {len(self.df.columns)}")
        print(f"Enhanced features: {len(self.df_enhanced.columns)}")
        
        new_features = [col for col in self.df_enhanced.columns if col not in self.df.columns]
        print(f"New features created: {new_features}")
        
        return self.df_enhanced
    
    def prepare_features(self):
        """Prepare features and target for modeling"""
        print("\nPreparing features and target...")
        target_column = 'HOME_W'
        
        # Select features (exclude target and non-numeric columns)
        self.feature_columns = [col for col in self.df_enhanced.columns 
                               if col != target_column and col != 'SEASON' and 
                               self.df_enhanced[col].dtype in ['int64', 'float64']]
        
        X = self.df_enhanced[self.feature_columns]
        y = self.df_enhanced[target_column]
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Feature columns: {self.feature_columns}")
        print(f"Target distribution:")
        print(y.value_counts(normalize=True))
        
        # Time-based split: train on earlier seasons, test on later seasons
        print("\nImplementing time-based split...")
        train_mask, test_mask = self.create_time_based_split(train_ratio=0.8, method='season_based')
        
        # Validate the time-based split
        self.validate_time_split(train_mask, test_mask)
        
        # Split the data based on time periods
        self.X_train = X[train_mask]
        self.X_test = X[test_mask]
        self.y_train = y[train_mask]
        self.y_test = y[test_mask]
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Get season information for display
        train_seasons = sorted(self.df_enhanced.loc[train_mask, 'SEASON'].unique())
        test_seasons = sorted(self.df_enhanced.loc[test_mask, 'SEASON'].unique())
        
        print(f"\nFinal Split Summary:")
        print(f"Training set: {self.X_train.shape} (seasons: {train_seasons})")
        print(f"Test set: {self.X_test.shape} (seasons: {test_seasons})")
        print(f"Training target distribution: {np.bincount(self.y_train)}")
        print(f"Test target distribution: {np.bincount(self.y_test)}")
        
        return X, y
    
    def initialize_models(self, include_svm=False):
        """Initialize all models"""
        print("\nInitializing models...")
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        }
        
        # Only include SVM if explicitly requested (it's very slow on large datasets)
        if include_svm:
            print("⚠️  Including SVM (this will be very slow on large datasets)")
            self.models['SVM'] = SVC(probability=True, random_state=42)
        else:
            print("✓ Skipping SVM for faster training (use include_svm=True if needed)")
        
    def train_models(self):
        """Train and evaluate all models"""
        print("\nTraining and evaluating models...")
        self.results = {}
        
        total_models = len(self.models)
        for idx, (name, model) in enumerate(self.models.items(), 1):
            print(f"\nTraining {name}... ({idx}/{total_models})")
            start_time = time.time()
            
            # Use scaled data for SVM and Logistic Regression
            if name in ['SVM', 'Logistic Regression']:
                print(f"  Using scaled data for {name}...")
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            else:
                print(f"  Using raw data for {name}...")
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            training_time = time.time() - start_time
            print(f"  ✓ Training completed in {training_time:.2f} seconds")
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None
            
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  AUC: {auc:.4f}" if auc else "  AUC: N/A")
    
    def compare_models(self):
        """Compare model performance"""
        print("\nComparing model performance...")
        comparison_data = []
        
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1'],
                'AUC': result['auc'] if result['auc'] is not None else np.nan
            })
        
        self.comparison_df = pd.DataFrame(comparison_data)
        print("Model Performance Comparison:")
        print(self.comparison_df.round(4))
        
        # Find best model
        self.best_model_name = self.comparison_df.loc[self.comparison_df['F1-Score'].idxmax(), 'Model']
        self.best_model = self.results[self.best_model_name]['model']
        print(f"\nBest model based on F1-Score: {self.best_model_name}")
        
        return self.comparison_df
    
    def create_ensemble(self):
        """Create an ensemble model"""
        print("\nCreating ensemble model...")
        best_models = []
        for name, result in self.results.items():
            if result['f1'] > 0.6:  # Only include models with good performance
                best_models.append((name, result['model']))
        
        if len(best_models) > 1:
            # Create voting classifier
            estimators = [(name, model) for name, model in best_models]
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            
            # Train ensemble
            ensemble.fit(self.X_train_scaled, self.y_train)
            
            # Evaluate ensemble
            y_pred_ensemble = ensemble.predict(self.X_test_scaled)
            y_pred_proba_ensemble = ensemble.predict_proba(self.X_test_scaled)[:, 1]
            
            ensemble_accuracy = accuracy_score(self.y_test, y_pred_ensemble)
            ensemble_f1 = f1_score(self.y_test, y_pred_ensemble)
            ensemble_auc = roc_auc_score(self.y_test, y_pred_proba_ensemble)
            
            print(f"Ensemble Model Results:")
            print(f"Accuracy: {ensemble_accuracy:.4f}")
            print(f"F1-Score: {ensemble_f1:.4f}")
            print(f"AUC: {ensemble_auc:.4f}")
            
            # Add to results
            self.results['Ensemble'] = {
                'model': ensemble,
                'accuracy': ensemble_accuracy,
                'precision': precision_score(self.y_test, y_pred_ensemble),
                'recall': recall_score(self.y_test, y_pred_ensemble),
                'f1': ensemble_f1,
                'auc': ensemble_auc,
                'predictions': y_pred_ensemble,
                'probabilities': y_pred_proba_ensemble
            }
        else:
            print("Not enough high-performing models for ensemble")
    
    def create_weighted_ensemble(self, weights=None):
        """Create a weighted ensemble based on model performance"""
        print("\nCreating weighted ensemble...")
        
        if weights is None:
            # Auto-calculate weights based on F1 scores
            total_f1 = sum(result['f1'] for result in self.results.values())
            weights = {name: result['f1'] / total_f1 for name, result in self.results.items()}
        
        print("Model Weights:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.3f}")
        
        # Create weighted predictions
        weighted_probs = np.zeros(len(self.y_test))
        
        for name, weight in weights.items():
            if name in self.results and 'probabilities' in self.results[name]:
                weighted_probs += weight * self.results[name]['probabilities']
        
        # Convert to predictions
        y_pred_weighted = (weighted_probs > 0.5).astype(int)
        
        # Calculate metrics
        weighted_accuracy = accuracy_score(self.y_test, y_pred_weighted)
        weighted_f1 = f1_score(self.y_test, y_pred_weighted)
        weighted_auc = roc_auc_score(self.y_test, weighted_probs)
        
        print(f"\nWeighted Ensemble Results:")
        print(f"Accuracy: {weighted_accuracy:.4f}")
        print(f"F1-Score: {weighted_f1:.4f}")
        print(f"AUC: {weighted_auc:.4f}")
        
        # Add to results
        self.results['Weighted Ensemble'] = {
            'model': 'Weighted Ensemble',
            'accuracy': weighted_accuracy,
            'precision': precision_score(self.y_test, y_pred_weighted),
            'recall': recall_score(self.y_test, y_pred_weighted),
            'f1': weighted_f1,
            'auc': weighted_auc,
            'predictions': y_pred_weighted,
            'probabilities': weighted_probs,
            'weights': weights
        }
        
        return weights
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        print("\nAnalyzing feature importance...")
        for name, result in self.results.items():
            if name in ['Random Forest', 'Gradient Boosting']:
                model = result['model']
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    plt.figure(figsize=(12, 8))
                    plt.title(f'Feature Importance - {name}')
                    plt.bar(range(len(importances)), importances[indices])
                    plt.xticks(range(len(importances)), 
                              [self.feature_columns[i] for i in indices], 
                              rotation=45, ha='right')
                    plt.ylabel('Feature Importance')
                    plt.tight_layout()
                    
                    # Save plot instead of showing it to avoid blocking
                    plot_filename = f'feature_importance_{name.replace(" ", "_").lower()}.png'
                    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                    print(f"  ✓ Feature importance plot saved as: {plot_filename}")
                    plt.close()  # Close the plot to free memory
                    
                    # Print top 10 features
                    print(f"\nTop 10 Most Important Features - {name}:")
                    for i in range(min(10, len(indices))):
                        print(f"{i+1:2d}. {self.feature_columns[indices[i]]:30s} - {importances[indices[i]]:.4f}")
                else:
                    print(f"{name} doesn't support feature importance")
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        print("\nPlotting confusion matrices...")
        n_models = len(self.results)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        fig.suptitle('Confusion Matrices for All Models', fontsize=16)
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (name, result) in enumerate(self.results.items()):
            row = idx // cols
            col = idx % cols
            
            if 'predictions' in result:
                cm = confusion_matrix(self.y_test, result['predictions'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'],
                            ax=axes[row, col])
                axes[row, col].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}')
                axes[row, col].set_xlabel('Predicted')
                axes[row, col].set_ylabel('Actual')
            else:
                axes[row, col].text(0.5, 0.5, f'{name}\nNo predictions', 
                                    ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(name)
        
        # Hide empty subplots
        for idx in range(n_models, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot instead of showing it to avoid blocking
        plot_filename = 'confusion_matrices_all_models.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Confusion matrices plot saved as: {plot_filename}")
        plt.close()  # Close the plot to free memory
    
    def plot_roc_curves(self):
        """Plot ROC curves for models that support probabilities"""
        print("\nPlotting ROC curves...")
        plt.figure(figsize=(10, 8))
        for name, result in self.results.items():
            if 'probabilities' in result and result['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
                auc_score = result['auc']
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        
        # Save plot instead of showing it to avoid blocking
        plot_filename = 'roc_curves_comparison.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ ROC curves plot saved as: {plot_filename}")
        plt.close()  # Close the plot to free memory
    
    def create_prediction_function(self):
        """Create a prediction function for new games"""
        print("\nCreating prediction function...")
        
        def predict_home_win(home_stats, away_stats, model=None, use_scaled=True, use_ensemble=True):
            """
            Predict home team win probability for a new game
            
            Parameters:
            home_stats: dict with home team statistics
            away_stats: dict with away team statistics
            model: trained model (uses best model if None)
            use_scaled: whether to use scaled features
            use_ensemble: whether to use ensemble prediction (recommended)
            
            Returns:
            prediction: 0 (loss) or 1 (win)
            probability: win probability
            confidence: confidence level in prediction
            """
            if model is None:
                if use_ensemble and 'Weighted Ensemble' in self.results:
                    model = 'Weighted Ensemble'
                else:
                    model = self.best_model
            
            # Create feature vector
            features = {}
            
            # Home team features
            for key, value in home_stats.items():
                features[f'HOME_{key}'] = value
            
            # Away team features
            for key, value in away_stats.items():
                features[f'AWAY_{key}'] = value
            
            # Create additional features
            features['HOME_OE_ADVANTAGE'] = home_stats.get('ROLLING_OE', 0) - away_stats.get('ROLLING_OE', 0)
            features['HOME_WIN_PCTG_ADVANTAGE'] = home_stats.get('HOME_WIN_PCTG', 0) - away_stats.get('AWAY_WIN_PCTG', 0)
            features['HOME_SCORING_MARGIN_ADVANTAGE'] = home_stats.get('ROLLING_SCORING_MARGIN', 0) - away_stats.get('ROLLING_SCORING_MARGIN', 0)
            features['REST_ADVANTAGE'] = home_stats.get('NUM_REST_DAYS', 0) - away_stats.get('NUM_REST_DAYS', 0)
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Ensure all required features are present
            for col in self.feature_columns:
                if col not in feature_df.columns:
                    feature_df[col] = 0  # Default value for missing features
            
            # Reorder columns to match training data
            feature_df = feature_df[self.feature_columns]
            
            if model == 'Weighted Ensemble':
                # Use weighted ensemble prediction
                weighted_probs = np.zeros(1)
                weights = self.results['Weighted Ensemble']['weights']
                
                for name, weight in weights.items():
                    if name in self.results and 'model' in self.results[name]:
                        # Scale features for models that need it
                        if name in ['SVM', 'Logistic Regression']:
                            feature_scaled = self.scaler.transform(feature_df)
                            prob = self.results[name]['model'].predict_proba(feature_scaled)[0, 1]
                        else:
                            prob = self.results[name]['model'].predict_proba(feature_df)[0, 1]
                        
                        weighted_probs += weight * prob
                
                prediction = (weighted_probs[0] > 0.5).astype(int)
                probability = weighted_probs[0]
                
            else:
                # Use single model prediction
                # Scale if needed
                if use_scaled and name in ['SVM', 'Logistic Regression']:
                    feature_df = self.scaler.transform(feature_df)
                
                # Make prediction
                if hasattr(model, 'predict_proba'):
                    prediction = model.predict(feature_df)[0]
                    probability = model.predict_proba(feature_df)[0, 1]
                else:
                    prediction = model.predict(feature_df)[0]
                    probability = None
            
            # Calculate confidence based on probability distance from 0.5
            if probability is not None:
                confidence = abs(probability - 0.5) * 2  # Scale to 0-1
            else:
                confidence = None
            
            return prediction, probability, confidence
        
        self.predict_home_win = predict_home_win
        print("Prediction function created!")
        print("\nTo use it:")
        print("1. Prepare home and away team statistics")
        print("2. Call model.predict_home_win(home_stats, away_stats)")
        print("3. Get prediction (0=loss, 1=win) and probability")
        
        return predict_home_win
    
    def create_time_based_split(self, train_ratio=0.8, method='season_based'):
        """
        Create time-based split for sports modeling
        
        Parameters:
        train_ratio: proportion of time periods to use for training
        method: 'season_based' or 'date_based'
        
        Returns:
        train_mask, test_mask: boolean masks for splitting
        """
        if method == 'season_based':
            # Split by seasons (most appropriate for NBA data)
            seasons = self.df_enhanced['SEASON'].unique()
            seasons_sorted = sorted(seasons)
            split_idx = int(len(seasons_sorted) * train_ratio)
            train_seasons = seasons_sorted[:split_idx]
            test_seasons = seasons_sorted[split_idx:]
            
            train_mask = self.df_enhanced['SEASON'].isin(train_seasons)
            test_mask = self.df_enhanced['SEASON'].isin(test_seasons)
            
            print(f"Season-based split:")
            print(f"  Training seasons: {train_seasons}")
            print(f"  Testing seasons: {test_seasons}")
            
        elif method == 'date_based':
            # Split by actual game dates (if available)
            if 'GAME_DATE' in self.df_enhanced.columns:
                dates = pd.to_datetime(self.df_enhanced['GAME_DATE'])
                date_threshold = dates.quantile(train_ratio)
                
                train_mask = dates <= date_threshold
                test_mask = dates > date_threshold
                
                print(f"Date-based split:")
                print(f"  Training until: {date_threshold.strftime('%Y-%m-%d')}")
                print(f"  Testing from: {date_threshold.strftime('%Y-%m-%d')}")
            else:
                print("GAME_DATE column not found, falling back to season-based split")
                return self.create_time_based_split(train_ratio, 'season_based')
        
        return train_mask, test_mask
    
    def validate_time_split(self, train_mask, test_mask):
        """
        Validate that the time-based split prevents data leakage
        
        Parameters:
        train_mask, test_mask: boolean masks for train/test split
        """
        print("\nValidating time-based split...")
        
        # Check for any overlap in seasons
        train_seasons = set(self.df_enhanced.loc[train_mask, 'SEASON'].unique())
        test_seasons = set(self.df_enhanced.loc[test_mask, 'SEASON'].unique())
        
        overlap = train_seasons.intersection(test_seasons)
        if overlap:
            print(f"⚠️  WARNING: Season overlap detected: {overlap}")
            print("   This could lead to data leakage!")
        else:
            print("✓ No season overlap - time-based split is valid")
        
        # Check for any overlap in dates if available
        if 'GAME_DATE' in self.df_enhanced.columns:
            train_dates = pd.to_datetime(self.df_enhanced.loc[train_mask, 'GAME_DATE'])
            test_dates = pd.to_datetime(self.df_enhanced.loc[test_mask, 'GAME_DATE'])
            
            if train_dates.max() >= test_dates.min():
                print(f"⚠️  WARNING: Date overlap detected!")
                print(f"   Training data extends to: {train_dates.max()}")
                print(f"   Testing data starts from: {test_dates.min()}")
            else:
                print("✓ No date overlap - temporal ordering is preserved")
        
        # Print split statistics
        print(f"\nSplit Statistics:")
        print(f"  Training samples: {train_mask.sum():,}")
        print(f"  Testing samples: {test_mask.sum():,}")
        print(f"  Total samples: {len(self.df_enhanced):,}")
        print(f"  Split ratio: {train_mask.sum() / len(self.df_enhanced):.1%} / {test_mask.sum() / len(self.df_enhanced):.1%}")
        
        return len(overlap) == 0
    
    def save_model(self, filepath='nba_home_win_model.pkl'):
        """Save the trained model and scaler"""
        print(f"\nSaving model to {filepath}...")
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'results': self.results,
            'comparison_df': self.comparison_df
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved successfully to {filepath}")
    
    def load_model(self, filepath='nba_home_win_model.pkl'):
        """Load a previously trained model"""
        print(f"Loading model from {filepath}...")
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.results = model_data['results']
        self.comparison_df = model_data['comparison_df']
        
        print("Model loaded successfully!")
    
    def run_full_pipeline(self, fast_mode=True, include_svm=False):
        """Run the complete modeling pipeline"""
        print("="*60)
        print("NBA HOME WIN-LOSS PREDICTION MODEL - FULL PIPELINE")
        print("="*60)
        
        if fast_mode:
            print("🚀 Running in FAST MODE (skipping slow models)")
        else:
            print("🐌 Running in FULL MODE (including all models)")
        
        # Load and prepare data
        self.load_data()
        self.handle_missing_values()
        self.create_features()
        self.prepare_features()
        
        # Train models
        self.initialize_models(include_svm=include_svm)
        self.train_models()
        self.compare_models()
        
        # Create ensemble
        self.create_ensemble()
        
        # Create weighted ensemble
        self.create_weighted_ensemble()
        
        # Analyze results
        self.plot_feature_importance()
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        
        # Create prediction function
        self.create_prediction_function()
        
        # Save model
        self.save_model()
        
        # Final summary
        self.print_summary()
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*60)
        print("NBA HOME WIN-LOSS PREDICTION MODEL - FINAL SUMMARY")
        print("="*60)
        
        print(f"\nDataset Information:")
        print(f"- Total samples: {len(self.df)}")
        print(f"- Features used: {len(self.feature_columns)}")
        print(f"- Target distribution: {self.y_train.value_counts().to_dict()}")
        
        print(f"\nModel Performance Summary:")
        for name, result in self.results.items():
            if 'accuracy' in result:
                print(f"- {name}: Accuracy={result['accuracy']:.3f}, F1={result['f1']:.3f}" + 
                      (f", AUC={result['auc']:.3f}" if result['auc'] else ""))
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best F1-Score: {self.comparison_df['F1-Score'].max():.4f}")
        print(f"Best Accuracy: {self.comparison_df['Accuracy'].max():.4f}")
        
        print(f"\nKey Features Created:")
        print(f"- Home/Away advantage metrics")
        print(f"- Rest advantage calculations")
        print(f"- Team strength indicators")
        print(f"- Momentum features")
        
        print(f"\nRecommendations:")
        print(f"1. Use {self.best_model_name} for predictions")
        print(f"2. Consider ensemble approach for improved stability")
        print(f"3. Monitor model performance on new data")
        print(f"4. Retrain periodically with new seasons")
        print(f"5. Use feature importance to understand key factors")


def main():
    """Main function to run the NBA prediction model"""
    # Create model instance
    model = NBAHomeWinLossModel()
    
    # Run the full pipeline in fast mode (skips SVM)
    print("Starting NBA prediction model in FAST MODE...")
    model.run_full_pipeline(fast_mode=True, include_svm=False)
    
    # Example prediction
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    
    # Sample home team stats
    home_stats = {
        'LAST_GAME_OE': 0.65,
        'HOME_WIN_PCTG': 0.70,
        'NUM_REST_DAYS': 2.0,
        'AWAY_WIN_PCTG': 0.60,
        'TOTAL_WIN_PCTG': 0.65,
        'ROLLING_SCORING_MARGIN': 5.0,
        'ROLLING_OE': 0.68
    }
    
    # Sample away team stats
    away_stats = {
        'LAST_GAME_OE': 0.58,
        'HOME_WIN_PCTG': 0.65,
        'NUM_REST_DAYS': 1.0,
        'AWAY_WIN_PCTG': 0.55,
        'TOTAL_WIN_PCTG': 0.60,
        'ROLLING_SCORING_MARGIN': -2.0,
        'ROLLING_OE': 0.60
    }
    
    prediction, probability, confidence = model.predict_home_win(home_stats, away_stats, use_ensemble=True)
    
    print(f"Home Team Stats: {home_stats}")
    print(f"Away Team Stats: {away_stats}")
    print(f"Prediction: {'WIN' if prediction == 1 else 'LOSS'}")
    print(f"Win Probability: {probability:.3f}" if probability else "Probability not available")
    print(f"Confidence: {confidence:.3f}" if confidence else "Confidence not available")
    
    # Show which models contributed to the prediction
    if 'Weighted Ensemble' in model.results:
        print(f"\nEnsemble Model Weights:")
        for name, weight in model.results['Weighted Ensemble']['weights'].items():
            print(f"  {name}: {weight:.3f}")


if __name__ == "__main__":
    main()
