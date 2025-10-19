"""
Enhanced Machine Learning model for PE fund selection with 95% accuracy target.

This module implements XGBoost with advanced feature engineering and
hyperparameter tuning to achieve superior prediction performance.
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing functions
from data_preprocessing import prepare_data, preprocess_single_fund
from visualizations import create_all_visualizations

def engineer_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced engineered features for better model performance.
    
    Args:
        df: DataFrame with original features
    
    Returns:
        DataFrame with additional engineered features
    """
    print("Engineering advanced features...")
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Interaction features
    if 'fund_size_mm' in df.columns and 'manager_track_record' in df.columns:
        df['size_x_track_record'] = df['fund_size_mm'] * df['manager_track_record']
        df['size_per_prior_fund'] = df['fund_size_mm'] / (df['manager_track_record'] + 1)
    
    # Performance ratios
    if 'tvpi' in df.columns and 'dpi' in df.columns:
        df['tvpi_to_dpi_ratio'] = df['tvpi'] / (df['dpi'] + 0.001)
        df['unrealized_value'] = df['tvpi'] - df['dpi']
        df['realization_rate'] = df['dpi'] / (df['tvpi'] + 0.001)
    
    # Age-adjusted metrics
    if 'fund_age_years' in df.columns:
        if 'tvpi' in df.columns:
            df['age_adjusted_tvpi'] = df['tvpi'] / (df['fund_age_years'] + 0.5)
            df['annual_tvpi_growth'] = (df['tvpi'] - 1) / (df['fund_age_years'] + 0.5)
        if 'dpi' in df.columns:
            df['age_adjusted_dpi'] = df['dpi'] / (df['fund_age_years'] + 0.5)
            df['distribution_velocity'] = df['dpi'] / (df['fund_age_years'] + 0.5)
    
    # Categorical feature engineering
    if 'vintage_year' in df.columns:
        df['years_since_vintage'] = 2025 - df['vintage_year']
        df['is_recent_vintage'] = (df['vintage_year'] >= 2020).astype(int)
        df['market_cycle'] = pd.cut(df['vintage_year'], 
                                   bins=[2009, 2012, 2016, 2020, 2024],
                                   labels=['recovery', 'growth', 'mature', 'covid_era'])
    
    # Size categories with more granularity
    if 'fund_size_mm' in df.columns:
        df['fund_size_category'] = pd.cut(df['fund_size_mm'],
                                         bins=[0, 250, 500, 1000, 2000],
                                         labels=['small', 'medium', 'large', 'mega'])
        df['log_fund_size'] = np.log1p(df['fund_size_mm'])
    
    # Manager experience levels
    if 'manager_track_record' in df.columns:
        df['is_first_time_manager'] = (df['manager_track_record'] == 0).astype(int)
        df['is_experienced_manager'] = (df['manager_track_record'] >= 3).astype(int)
        df['manager_experience_level'] = pd.cut(df['manager_track_record'],
                                               bins=[-1, 0, 2, 4, 10],
                                               labels=['first_time', 'emerging', 'established', 'veteran'])
    
    # Performance flags
    if 'tvpi' in df.columns:
        df['is_outperforming'] = (df['tvpi'] > 2.0).astype(int)
        df['tvpi_squared'] = df['tvpi'] ** 2
        df['tvpi_sqrt'] = np.sqrt(df['tvpi'])
    
    if 'dpi' in df.columns:
        df['has_strong_distributions'] = (df['dpi'] > 1.0).astype(int)
        df['dpi_squared'] = df['dpi'] ** 2
    
    # Polynomial features for key metrics
    if 'fund_age_years' in df.columns:
        df['fund_age_squared'] = df['fund_age_years'] ** 2
        df['fund_age_log'] = np.log1p(df['fund_age_years'])
    
    print(f"Created {len(df.columns)} total features (including engineered)")
    
    return df

def train_xgboost_with_tuning(X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              random_state: int = 42) -> Tuple[XGBClassifier, Dict]:
    """
    Train XGBoost with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features for evaluation
        y_test: Test labels for evaluation
        random_state: Random seed
    
    Returns:
        Tuple of (best model, best parameters)
    """
    print("\nTraining XGBoost with hyperparameter tuning...")
    print(f"Training set shape: {X_train.shape}")
    
    # Define parameter grid for XGBoost
    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [6, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    
    # Initialize XGBoost
    xgb = XGBClassifier(
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])  # Handle imbalance
    )
    
    # Use RandomizedSearchCV for faster tuning
    print("Performing randomized search for best hyperparameters...")
    random_search = RandomizedSearchCV(
        xgb,
        param_distributions=param_grid,
        n_iter=50,  # Number of parameter combinations to try
        scoring='roc_auc',
        cv=5,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Fit the random search
    random_search.fit(X_train, y_train)
    
    # Get best model
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    print(f"\nBest parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nTest Performance with best XGBoost:")
    print(f"  Accuracy: {test_accuracy:.2%}")
    print(f"  ROC-AUC: {test_roc_auc:.3f}")
    
    return best_model, best_params

def create_ensemble_model(X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         random_state: int = 42) -> VotingClassifier:
    """
    Create an ensemble model combining multiple algorithms.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        random_state: Random seed
    
    Returns:
        Trained VotingClassifier ensemble
    """
    print("\nCreating ensemble model...")
    
    # Train individual models with optimized parameters
    
    # 1. XGBoost (tuned)
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        gamma=0.1,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])
    )
    
    # 2. Random Forest (optimized)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # 3. Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=random_state
    )
    
    # Create voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb),
            ('rf', rf),
            ('gb', gb)
        ],
        voting='soft',  # Use probability averaging
        n_jobs=-1
    )
    
    print("Training ensemble model...")
    ensemble.fit(X_train, y_train)
    
    # Evaluate ensemble
    y_pred = ensemble.predict(X_test)
    y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nEnsemble Model Performance:")
    print(f"  Accuracy:  {accuracy:.2%}")
    print(f"  ROC-AUC:   {roc_auc:.3f}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1 Score:  {f1:.2%}")
    
    # Cross-validation score
    cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"\n5-Fold Cross-Validation ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return ensemble

def prepare_enhanced_data(data_path: str) -> Tuple:
    """
    Load data and apply advanced feature engineering.
    
    Args:
        data_path: Path to the raw data file
    
    Returns:
        Tuple of processed data components
    """
    print("Loading and preparing enhanced dataset...")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Handle missing values before feature engineering
    # Fill missing DPI values with median
    if 'dpi' in df.columns:
        df['dpi'].fillna(df['dpi'].median(), inplace=True)
    
    # Apply advanced feature engineering
    df = engineer_advanced_features(df)
    
    # Fill any remaining NaN values that may have been created during feature engineering
    # Only fill numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # Prepare data with original preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Define target (top quartile based on IRR)
    irr_75th = df['irr_percent'].quantile(0.75)
    df['is_top_quartile'] = (df['irr_percent'] > irr_75th).astype(int)
    
    # Select features (exclude target columns)
    feature_cols = [col for col in df.columns if col not in ['irr_percent', 'is_top_quartile', 'fund_id']]
    
    # Handle categorical variables
    df_encoded = pd.get_dummies(df[feature_cols], columns=['sector', 'geography', 
                                                           'fund_size_category', 
                                                           'market_cycle',
                                                           'manager_experience_level'])
    
    # Split data
    X = df_encoded.values
    y = df['is_top_quartile'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    feature_names = list(df_encoded.columns)
    
    print(f"Enhanced feature count: {len(feature_names)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, scaler, feature_names

def main():
    """Main function to train enhanced PE fund selection model."""
    print("="*60)
    print("PE Fund Selection Model - Enhanced Version")
    print("Target: 95% Accuracy & 0.95 ROC-AUC")
    print("="*60)
    
    # Prepare enhanced data
    data_path = os.path.join('data', 'raw', 'pe_funds.csv')
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return
    
    # Load and preprocess data with enhanced features
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_enhanced_data(data_path)
    
    # Train XGBoost with hyperparameter tuning
    print("\n" + "="*60)
    print("PHASE 1: XGBoost with Hyperparameter Tuning")
    print("="*60)
    xgb_model, xgb_params = train_xgboost_with_tuning(X_train, y_train, X_test, y_test)
    
    # Create ensemble model
    print("\n" + "="*60)
    print("PHASE 2: Ensemble Model Creation")
    print("="*60)
    ensemble_model = create_ensemble_model(X_train, y_train, X_test, y_test)
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION")
    print("="*60)
    
    y_pred = ensemble_model.predict(X_test)
    y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
    
    final_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    print(f"\n{'='*40}")
    print(f"FINAL PERFORMANCE METRICS")
    print(f"{'='*40}")
    print(f"Accuracy:  {final_metrics['accuracy']:.2%}")
    print(f"Precision: {final_metrics['precision']:.2%}")
    print(f"Recall:    {final_metrics['recall']:.2%}")
    print(f"F1 Score:  {final_metrics['f1_score']:.2%}")
    print(f"ROC-AUC:   {final_metrics['roc_auc']:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(final_metrics['confusion_matrix'])
    
    # Save enhanced model
    print("\n" + "="*60)
    os.makedirs('models', exist_ok=True)
    
    # Save ensemble model
    model_path = os.path.join('models', 'pe_fund_selector_enhanced.pkl')
    joblib.dump(ensemble_model, model_path)
    print(f"Enhanced model saved to {model_path}")
    
    # Save scaler
    scaler_path = os.path.join('models', 'scaler_enhanced.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Enhanced scaler saved to {scaler_path}")
    
    # Save feature names
    features_path = os.path.join('models', 'feature_names_enhanced.pkl')
    joblib.dump(feature_names, features_path)
    print(f"Enhanced feature names saved to {features_path}")
    
    # Check if we achieved our target
    print("\n" + "="*60)
    if final_metrics['accuracy'] >= 0.95 and final_metrics['roc_auc'] >= 0.95:
        print("✅ SUCCESS! Target metrics achieved!")
        print(f"   Accuracy: {final_metrics['accuracy']:.2%} (target: 95%)")
        print(f"   ROC-AUC: {final_metrics['roc_auc']:.3f} (target: 0.95)")
    else:
        print("⚠️  Target metrics not fully achieved. Further tuning needed.")
        print(f"   Accuracy: {final_metrics['accuracy']:.2%} (target: 95%)")
        print(f"   ROC-AUC: {final_metrics['roc_auc']:.3f} (target: 0.95)")
    
    print("\n" + "="*60)
    print("Enhanced model training complete!")
    
    return ensemble_model, final_metrics, feature_names

if __name__ == "__main__":
    main()
