"""
Advanced Stacking Ensemble Model for PE Fund Selection
Target: 95% Accuracy & 0.95 ROC-AUC

This module implements a sophisticated stacking ensemble with:
- CatBoost, LightGBM, XGBoost, Random Forest as base models
- Logistic Regression as meta-learner
- Optuna for hyperparameter optimization
- Feature selection via RFE/LASSO
- Probability calibration
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
from typing import Dict, Tuple, Optional, List
import optuna
from optuna.samplers import TPESampler

from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    StackingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

# Import preprocessing functions
from data_preprocessing import prepare_data


def engineer_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced engineered features for superior model performance.
    Includes interaction terms, ratios, and domain-specific features.
    """
    print("Engineering advanced features...")
    df = df.copy()
    
    # === INTERACTION FEATURES ===
    if 'fund_size_mm' in df.columns and 'manager_track_record' in df.columns:
        df['size_x_track_record'] = df['fund_size_mm'] * df['manager_track_record']
        df['size_per_prior_fund'] = df['fund_size_mm'] / (df['manager_track_record'] + 1)
        df['log_size_x_track'] = np.log1p(df['fund_size_mm']) * df['manager_track_record']
    
    # === PERFORMANCE RATIOS ===
    if 'tvpi' in df.columns and 'dpi' in df.columns:
        df['tvpi_to_dpi_ratio'] = df['tvpi'] / (df['dpi'] + 0.001)
        df['unrealized_value'] = df['tvpi'] - df['dpi']
        df['realization_rate'] = df['dpi'] / (df['tvpi'] + 0.001)
        df['value_creation_mult'] = df['tvpi'] * df['dpi']
    
    # === AGE-ADJUSTED METRICS ===
    if 'fund_age_years' in df.columns:
        if 'tvpi' in df.columns:
            df['age_adjusted_tvpi'] = df['tvpi'] / (df['fund_age_years'] + 0.5)
            df['annual_tvpi_growth'] = (df['tvpi'] - 1) / (df['fund_age_years'] + 0.5)
            df['tvpi_velocity'] = df['tvpi'] ** (1 / (df['fund_age_years'] + 0.5))
        if 'dpi' in df.columns:
            df['age_adjusted_dpi'] = df['dpi'] / (df['fund_age_years'] + 0.5)
            df['distribution_velocity'] = df['dpi'] / (df['fund_age_years'] + 0.5)
            df['dpi_acceleration'] = df['dpi'] / ((df['fund_age_years'] + 0.5) ** 2)
    
    # === VINTAGE FEATURES ===
    if 'vintage_year' in df.columns:
        df['years_since_vintage'] = 2025 - df['vintage_year']
        df['is_recent_vintage'] = (df['vintage_year'] >= 2020).astype(int)
        df['is_mature_vintage'] = (df['vintage_year'] <= 2015).astype(int)
        
        # Market cycle encoding
        df['market_cycle'] = pd.cut(df['vintage_year'], 
                                   bins=[2009, 2012, 2016, 2020, 2024],
                                   labels=['recovery', 'growth', 'mature', 'covid_era'])
        
        # Vintage cohort effects
        df['vintage_cohort'] = pd.cut(df['vintage_year'],
                                     bins=[2009, 2014, 2019, 2024],
                                     labels=['early_2010s', 'mid_2010s', 'late_2010s'])
    
    # === SIZE CATEGORIZATION ===
    if 'fund_size_mm' in df.columns:
        df['fund_size_category'] = pd.cut(df['fund_size_mm'],
                                         bins=[0, 250, 500, 1000, 2000],
                                         labels=['small', 'medium', 'large', 'mega'])
        df['log_fund_size'] = np.log1p(df['fund_size_mm'])
        df['fund_size_squared'] = df['fund_size_mm'] ** 2
        df['fund_size_sqrt'] = np.sqrt(df['fund_size_mm'])
    
    # === MANAGER EXPERIENCE ===
    if 'manager_track_record' in df.columns:
        df['is_first_time_manager'] = (df['manager_track_record'] == 0).astype(int)
        df['is_experienced_manager'] = (df['manager_track_record'] >= 3).astype(int)
        df['manager_experience_level'] = pd.cut(df['manager_track_record'],
                                               bins=[-1, 0, 2, 4, 10],
                                               labels=['first_time', 'emerging', 'established', 'veteran'])
        df['track_record_squared'] = df['manager_track_record'] ** 2
    
    # === PERFORMANCE FLAGS ===
    if 'tvpi' in df.columns:
        df['is_outperforming'] = (df['tvpi'] > 2.0).astype(int)
        df['is_underperforming'] = (df['tvpi'] < 1.5).astype(int)
        df['tvpi_squared'] = df['tvpi'] ** 2
        df['tvpi_cubed'] = df['tvpi'] ** 3
        df['tvpi_sqrt'] = np.sqrt(df['tvpi'])
        df['tvpi_log'] = np.log1p(df['tvpi'])
    
    if 'dpi' in df.columns:
        df['has_strong_distributions'] = (df['dpi'] > 1.0).astype(int)
        df['has_weak_distributions'] = (df['dpi'] < 0.5).astype(int)
        df['dpi_squared'] = df['dpi'] ** 2
        df['dpi_sqrt'] = np.sqrt(df['dpi'])
    
    # === POLYNOMIAL FEATURES ===
    if 'fund_age_years' in df.columns:
        df['fund_age_squared'] = df['fund_age_years'] ** 2
        df['fund_age_cubed'] = df['fund_age_years'] ** 3
        df['fund_age_log'] = np.log1p(df['fund_age_years'])
    
    # === SECTOR-GEOGRAPHY INTERACTIONS ===
    if 'sector' in df.columns and 'geography' in df.columns:
        df['sector_geo_combo'] = df['sector'] + '_' + df['geography']
    
    print(f"Created {len(df.columns)} total features (including engineered)")
    return df


def select_features_with_rfe(X_train, y_train, n_features_to_select=30):
    """
    Select top features using Recursive Feature Elimination.
    """
    print(f"Selecting top {n_features_to_select} features using RFE...")
    
    # Use a fast estimator for RFE
    estimator = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    selector = RFE(
        estimator=estimator,
        n_features_to_select=n_features_to_select,
        step=5,
        verbose=1
    )
    
    selector.fit(X_train, y_train)
    return selector


def create_optuna_objective(X_train, y_train, X_val, y_val):
    """
    Create Optuna objective function for hyperparameter optimization.
    """
    def objective(trial):
        # Select model type
        model_type = trial.suggest_categorical('model_type', 
                                              ['xgboost', 'catboost', 'lightgbm', 'random_forest'])
        
        if model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 500),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 15),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('xgb_min_child', 1, 10),
                'gamma': trial.suggest_float('xgb_gamma', 0, 0.5)
            }
            model = XGBClassifier(**params, random_state=42, use_label_encoder=False, 
                                eval_metric='logloss')
        
        elif model_type == 'catboost':
            params = {
                'iterations': trial.suggest_int('cat_iterations', 100, 500),
                'depth': trial.suggest_int('cat_depth', 4, 10),
                'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('cat_l2_reg', 1, 10),
                'bagging_temperature': trial.suggest_float('cat_bagging_temp', 0, 1)
            }
            model = CatBoostClassifier(**params, random_state=42, verbose=0)
        
        elif model_type == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 500),
                'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 300),
                'max_depth': trial.suggest_int('lgb_max_depth', 3, 15),
                'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.5, 1.0)
            }
            model = LGBMClassifier(**params, random_state=42, verbosity=-1)
        
        else:  # random_forest
            params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 100, 500),
                'max_depth': trial.suggest_int('rf_max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('rf_min_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('rf_min_leaf', 1, 10),
                'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2'])
            }
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        
        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred)
    
    return objective


def optimize_with_optuna(X_train, y_train, X_val, y_val, n_trials=100):
    """
    Optimize hyperparameters using Optuna.
    """
    print(f"Running Optuna optimization with {n_trials} trials...")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    
    objective = create_optuna_objective(X_train, y_train, X_val, y_val)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"Best ROC-AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    return study.best_params


def create_stacking_ensemble(X_train, y_train, X_val, y_val):
    """
    Create a sophisticated stacking ensemble with multiple base models.
    """
    print("\n" + "="*60)
    print("CREATING ADVANCED STACKING ENSEMBLE")
    print("="*60)
    
    # Define base models with optimized parameters
    base_models = []
    
    # 1. XGBoost (optimized)
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    base_models.append(('xgboost', xgb))
    
    # 2. CatBoost (new)
    cat = CatBoostClassifier(
        iterations=400,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3,
        bagging_temperature=0.1,
        random_state=42,
        verbose=0
    )
    base_models.append(('catboost', cat))
    
    # 3. LightGBM (new)
    lgb = LGBMClassifier(
        n_estimators=400,
        num_leaves=50,
        max_depth=8,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        random_state=42,
        verbosity=-1
    )
    base_models.append(('lightgbm', lgb))
    
    # 4. Random Forest (optimized)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    base_models.append(('random_forest', rf))
    
    # 5. Extra Trees (new)
    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    base_models.append(('extra_trees', et))
    
    # 6. Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    base_models.append(('gradient_boosting', gb))
    
    # Meta-learner: Logistic Regression with regularization
    meta_learner = LogisticRegression(
        C=1.0,
        penalty='l2',
        max_iter=1000,
        random_state=42
    )
    
    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,  # 5-fold cross-validation for generating meta-features
        stack_method='predict_proba',  # Use probabilities as meta-features
        n_jobs=-1,
        verbose=2
    )
    
    print("Training stacking ensemble...")
    stacking_clf.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = stacking_clf.predict(X_val)
    y_pred_proba = stacking_clf.predict_proba(X_val)[:, 1]
    
    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"\nValidation Performance:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    
    return stacking_clf


def add_probability_calibration(model, X_train, y_train):
    """
    Add probability calibration to improve prediction confidence.
    """
    print("Applying probability calibration...")
    
    calibrated_clf = CalibratedClassifierCV(
        model,
        method='isotonic',  # or 'sigmoid'
        cv=3
    )
    
    calibrated_clf.fit(X_train, y_train)
    return calibrated_clf


def prepare_enhanced_data(data_path: str) -> Tuple:
    """
    Load data and apply comprehensive feature engineering.
    """
    print("Loading and preparing enhanced dataset...")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records")
    
    # Handle missing values before feature engineering
    if 'dpi' in df.columns:
        df['dpi'].fillna(df['dpi'].median(), inplace=True)
    
    # Apply advanced feature engineering
    df = engineer_advanced_features(df)
    
    # Fill any remaining NaN values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # Prepare data with original preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Define target (top quartile based on IRR)
    irr_75th = df['irr_percent'].quantile(0.75)
    df['is_top_quartile'] = (df['irr_percent'] > irr_75th).astype(int)
    
    print(f"Top quartile threshold: {irr_75th:.1f}% IRR")
    print(f"Positive class ratio: {df['is_top_quartile'].mean():.2%}")
    
    # Select features (exclude target columns)
    feature_cols = [col for col in df.columns if col not in ['irr_percent', 'is_top_quartile', 'fund_id']]
    
    # Handle categorical variables
    df_encoded = pd.get_dummies(df[feature_cols], 
                               columns=['sector', 'geography', 'fund_size_category', 
                                       'market_cycle', 'manager_experience_level',
                                       'vintage_cohort', 'sector_geo_combo'])
    
    # Split data - using larger dataset!
    X = df_encoded.values
    y = df['is_top_quartile'].values
    
    # Create train/val/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 to get 15% val
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    feature_names = list(df_encoded.columns)
    
    print(f"\nDataset splits:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Features: {len(feature_names)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names


def main():
    """Main function to train advanced stacking model for 95% accuracy."""
    print("="*60)
    print("PE FUND SELECTION - ADVANCED STACKING MODEL")
    print("Target: 95% Accuracy & 0.95 ROC-AUC")
    print("="*60)
    
    # Load enhanced data
    data_path = os.path.join('data', 'raw', 'pe_funds.csv')
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Please run generate_synthetic_data.py first")
        return
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = prepare_enhanced_data(data_path)
    
    # Feature selection with RFE
    print("\n" + "="*60)
    print("PHASE 1: FEATURE SELECTION")
    print("="*60)
    selector = select_features_with_rfe(X_train, y_train, n_features_to_select=50)
    
    X_train_selected = selector.transform(X_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selector.support_[i]]
    print(f"Selected {len(selected_features)} features")
    
    # Hyperparameter optimization with Optuna (optional - takes time)
    print("\n" + "="*60)
    print("PHASE 2: HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    # Comment out for faster iteration
    # best_params = optimize_with_optuna(X_train_selected, y_train, X_val_selected, y_val, n_trials=50)
    
    # Create stacking ensemble
    print("\n" + "="*60)
    print("PHASE 3: STACKING ENSEMBLE CREATION")
    print("="*60)
    stacking_model = create_stacking_ensemble(
        X_train_selected, y_train, 
        X_val_selected, y_val
    )
    
    # Add probability calibration
    print("\n" + "="*60)
    print("PHASE 4: PROBABILITY CALIBRATION")
    print("="*60)
    calibrated_model = add_probability_calibration(
        stacking_model, 
        X_train_selected, 
        y_train
    )
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION ON TEST SET")
    print("="*60)
    
    y_pred = calibrated_model.predict(X_test_selected)
    y_pred_proba = calibrated_model.predict_proba(X_test_selected)[:, 1]
    
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
    print(f"ROC-AUC:   {final_metrics['roc_auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(final_metrics['confusion_matrix'])
    
    # Cross-validation for robustness check
    print("\n" + "="*60)
    print("CROSS-VALIDATION ROBUSTNESS CHECK")
    print("="*60)
    cv_scores = cross_val_score(
        calibrated_model, 
        X_train_selected, 
        y_train, 
        cv=5, 
        scoring='roc_auc',
        n_jobs=-1
    )
    print(f"5-Fold CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save model
    print("\n" + "="*60)
    os.makedirs('models', exist_ok=True)
    
    # Save calibrated stacking model
    model_path = os.path.join('models', 'pe_fund_selector_stacking.pkl')
    joblib.dump(calibrated_model, model_path)
    print(f"Stacking model saved to {model_path}")
    
    # Save feature selector
    selector_path = os.path.join('models', 'feature_selector.pkl')
    joblib.dump(selector, selector_path)
    print(f"Feature selector saved to {selector_path}")
    
    # Save scaler
    scaler_path = os.path.join('models', 'scaler_stacking.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Save selected feature names
    features_path = os.path.join('models', 'selected_features_stacking.pkl')
    joblib.dump(selected_features, features_path)
    print(f"Selected features saved to {features_path}")
    
    # Check if we achieved our target
    print("\n" + "="*60)
    if final_metrics['accuracy'] >= 0.95 and final_metrics['roc_auc'] >= 0.95:
        print("🎉 SUCCESS! TARGET METRICS ACHIEVED! 🎉")
        print(f"   Accuracy: {final_metrics['accuracy']:.2%} (target: 95%)")
        print(f"   ROC-AUC: {final_metrics['roc_auc']:.4f} (target: 0.95)")
    else:
        gap_accuracy = 0.95 - final_metrics['accuracy']
        gap_roc = 0.95 - final_metrics['roc_auc']
        print("⚠️  Getting closer to target metrics...")
        print(f"   Accuracy: {final_metrics['accuracy']:.2%} (gap: {gap_accuracy:.2%})")
        print(f"   ROC-AUC: {final_metrics['roc_auc']:.4f} (gap: {gap_roc:.4f})")
        print("\nNext steps to improve:")
        if gap_accuracy > 0.02:
            print("  - Run with more Optuna trials for better hyperparameters")
            print("  - Try neural network as meta-learner")
            print("  - Add more base models to ensemble")
    
    print("\n" + "="*60)
    print("Advanced stacking model training complete!")
    
    return calibrated_model, final_metrics, selected_features


if __name__ == "__main__":
    main()
