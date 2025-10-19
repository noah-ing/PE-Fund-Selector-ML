"""
Neural Network Enhanced Stacking Model for 95% Accuracy Target
Implements advanced techniques including:
- Neural network as meta-learner
- More aggressive feature engineering
- Synthetic Minority Over-sampling (SMOTE)
- Advanced ensemble with 8+ models
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
from typing import Dict, Tuple, Optional, List

from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    StackingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')


def create_ultra_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create even more aggressive feature engineering for 95% accuracy."""
    print("Creating ultra-engineered features...")
    df = df.copy()
    
    # Fill NaN values first
    if 'dpi' in df.columns:
        df['dpi'].fillna(df['dpi'].median(), inplace=True)
    
    # === CORE PERFORMANCE METRICS ===
    if 'tvpi' in df.columns and 'dpi' in df.columns:
        df['tvpi_to_dpi_ratio'] = df['tvpi'] / (df['dpi'] + 0.001)
        df['unrealized_value'] = df['tvpi'] - df['dpi']
        df['realization_rate'] = df['dpi'] / (df['tvpi'] + 0.001)
        df['value_creation_mult'] = df['tvpi'] * df['dpi']
        df['performance_score'] = df['tvpi'] * 0.6 + df['dpi'] * 0.4
        df['tvpi_dpi_harmonic_mean'] = 2 * df['tvpi'] * df['dpi'] / (df['tvpi'] + df['dpi'] + 0.001)
    
    # === ADVANCED INTERACTION FEATURES ===
    if 'fund_size_mm' in df.columns and 'manager_track_record' in df.columns:
        df['size_x_track_record'] = df['fund_size_mm'] * df['manager_track_record']
        df['size_per_prior_fund'] = df['fund_size_mm'] / (df['manager_track_record'] + 1)
        df['log_size_x_track'] = np.log1p(df['fund_size_mm']) * df['manager_track_record']
        df['size_track_ratio'] = df['fund_size_mm'] / (100 * (df['manager_track_record'] + 1))
        df['experienced_large_fund'] = ((df['manager_track_record'] >= 3) & 
                                       (df['fund_size_mm'] > 500)).astype(int)
    
    # === TIME-BASED FEATURES ===
    if 'fund_age_years' in df.columns:
        df['fund_age_squared'] = df['fund_age_years'] ** 2
        df['fund_age_cubed'] = df['fund_age_years'] ** 3
        df['fund_age_sqrt'] = np.sqrt(df['fund_age_years'])
        df['fund_age_log'] = np.log1p(df['fund_age_years'])
        df['is_mature'] = (df['fund_age_years'] > 5).astype(int)
        df['is_young'] = (df['fund_age_years'] < 3).astype(int)
        
        if 'tvpi' in df.columns:
            df['age_adjusted_tvpi'] = df['tvpi'] / (df['fund_age_years'] + 0.5)
            df['annual_tvpi_growth'] = (df['tvpi'] - 1) / (df['fund_age_years'] + 0.5)
            df['tvpi_velocity'] = df['tvpi'] ** (1 / (df['fund_age_years'] + 0.5))
            df['tvpi_age_interaction'] = df['tvpi'] * df['fund_age_years']
        
        if 'dpi' in df.columns:
            df['age_adjusted_dpi'] = df['dpi'] / (df['fund_age_years'] + 0.5)
            df['distribution_velocity'] = df['dpi'] / (df['fund_age_years'] + 0.5)
            df['dpi_acceleration'] = df['dpi'] / ((df['fund_age_years'] + 0.5) ** 2)
            df['dpi_age_interaction'] = df['dpi'] * df['fund_age_years']
    
    # === VINTAGE COHORT EFFECTS ===
    if 'vintage_year' in df.columns:
        df['years_since_vintage'] = 2025 - df['vintage_year']
        df['is_recent_vintage'] = (df['vintage_year'] >= 2020).astype(int)
        df['is_mature_vintage'] = (df['vintage_year'] <= 2015).astype(int)
        df['vintage_decade'] = (df['vintage_year'] // 10) * 10
        
        # Economic cycle mapping
        df['market_cycle'] = pd.cut(df['vintage_year'], 
                                   bins=[2009, 2012, 2016, 2020, 2024],
                                   labels=['recovery', 'growth', 'mature', 'covid_era'])
        
        # Vintage performance interaction
        if 'tvpi' in df.columns:
            df['vintage_performance'] = df['tvpi'] * (2025 - df['vintage_year']) / 10
    
    # === FUND SIZE CATEGORIZATION ===
    if 'fund_size_mm' in df.columns:
        df['fund_size_category'] = pd.cut(df['fund_size_mm'],
                                         bins=[0, 250, 500, 1000, 2000],
                                         labels=['small', 'medium', 'large', 'mega'])
        df['log_fund_size'] = np.log1p(df['fund_size_mm'])
        df['fund_size_squared'] = df['fund_size_mm'] ** 2
        df['fund_size_cubed'] = df['fund_size_mm'] ** 3
        df['fund_size_sqrt'] = np.sqrt(df['fund_size_mm'])
        df['normalized_size'] = df['fund_size_mm'] / df['fund_size_mm'].mean()
    
    # === MANAGER EXPERIENCE ===
    if 'manager_track_record' in df.columns:
        df['is_first_time_manager'] = (df['manager_track_record'] == 0).astype(int)
        df['is_experienced_manager'] = (df['manager_track_record'] >= 3).astype(int)
        df['is_veteran_manager'] = (df['manager_track_record'] >= 5).astype(int)
        df['track_record_squared'] = df['manager_track_record'] ** 2
        df['track_record_log'] = np.log1p(df['manager_track_record'])
        
        df['manager_experience_level'] = pd.cut(df['manager_track_record'],
                                               bins=[-1, 0, 2, 4, 10],
                                               labels=['first_time', 'emerging', 
                                                      'established', 'veteran'])
    
    # === PERFORMANCE FLAGS & TRANSFORMATIONS ===
    if 'tvpi' in df.columns:
        df['is_top_performer'] = (df['tvpi'] > df['tvpi'].quantile(0.75)).astype(int)
        df['is_outperforming'] = (df['tvpi'] > 2.0).astype(int)
        df['is_underperforming'] = (df['tvpi'] < 1.5).astype(int)
        df['tvpi_squared'] = df['tvpi'] ** 2
        df['tvpi_cubed'] = df['tvpi'] ** 3
        df['tvpi_sqrt'] = np.sqrt(df['tvpi'])
        df['tvpi_log'] = np.log1p(df['tvpi'])
        df['tvpi_reciprocal'] = 1 / (df['tvpi'] + 0.001)
    
    if 'dpi' in df.columns:
        df['has_strong_distributions'] = (df['dpi'] > 1.0).astype(int)
        df['has_weak_distributions'] = (df['dpi'] < 0.5).astype(int)
        df['dpi_squared'] = df['dpi'] ** 2
        df['dpi_sqrt'] = np.sqrt(df['dpi'])
        df['dpi_log'] = np.log1p(df['dpi'])
    
    # === SECTOR-GEOGRAPHY INTERACTIONS ===
    if 'sector' in df.columns and 'geography' in df.columns:
        df['sector_geo_combo'] = df['sector'] + '_' + df['geography']
        
        # High-growth combinations
        df['is_tech_northamerica'] = ((df['sector'] == 'Technology') & 
                                      (df['geography'] == 'North America')).astype(int)
        df['is_tech_asia'] = ((df['sector'] == 'Technology') & 
                             (df['geography'] == 'Asia')).astype(int)
        df['is_healthcare_northamerica'] = ((df['sector'] == 'Healthcare') & 
                                           (df['geography'] == 'North America')).astype(int)
    
    # === COMPOSITE SCORES ===
    if 'fund_size_mm' in df.columns and 'manager_track_record' in df.columns and 'tvpi' in df.columns:
        # Create composite quality score
        df['quality_score'] = (
            df['fund_size_mm'] / df['fund_size_mm'].max() * 0.2 +
            df['manager_track_record'] / df['manager_track_record'].max() * 0.3 +
            df['tvpi'] / df['tvpi'].max() * 0.5
        )
    
    print(f"Created {len(df.columns)} total features")
    return df


def create_neural_stacking_ensemble(X_train, y_train, X_val, y_val):
    """Create advanced stacking ensemble with neural network meta-learner."""
    print("\n" + "="*60)
    print("CREATING NEURAL STACKING ENSEMBLE")
    print("="*60)
    
    # Apply SMOTE for class imbalance
    print("Applying SMOTE for class balance...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"Balanced training set: {len(X_train_balanced)} samples")
    
    # Define extensive base models
    base_models = []
    
    # 1. XGBoost (highly tuned)
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.02,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        gamma=0.05,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    base_models.append(('xgboost', xgb))
    
    # 2. CatBoost (tuned)
    cat = CatBoostClassifier(
        iterations=500,
        depth=10,
        learning_rate=0.03,
        l2_leaf_reg=2,
        bagging_temperature=0.05,
        random_state=42,
        verbose=0
    )
    base_models.append(('catboost', cat))
    
    # 3. LightGBM (tuned)
    lgb = LGBMClassifier(
        n_estimators=500,
        num_leaves=80,
        max_depth=10,
        learning_rate=0.03,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        lambda_l1=0.1,
        lambda_l2=0.1,
        random_state=42,
        verbosity=-1
    )
    base_models.append(('lightgbm', lgb))
    
    # 4. Random Forest (tuned)
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    base_models.append(('random_forest', rf))
    
    # 5. Extra Trees (tuned)
    et = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    )
    base_models.append(('extra_trees', et))
    
    # 6. Gradient Boosting (tuned)
    gb = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.9,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42
    )
    base_models.append(('gradient_boosting', gb))
    
    # 7. AdaBoost
    ada = AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.5,
        random_state=42
    )
    base_models.append(('adaboost', ada))
    
    # 8. Histogram Gradient Boosting (fast)
    hist_gb = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=10,
        learning_rate=0.05,
        random_state=42
    )
    base_models.append(('hist_gradient_boosting', hist_gb))
    
    # Neural Network Meta-learner with aggressive architecture
    meta_learner = MLPClassifier(
        hidden_layer_sizes=(200, 100, 50, 25),  # 4-layer deep network
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    )
    
    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1,
        verbose=1
    )
    
    print("Training neural stacking ensemble on balanced data...")
    stacking_clf.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate
    y_pred = stacking_clf.predict(X_val)
    y_pred_proba = stacking_clf.predict_proba(X_val)[:, 1]
    
    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"\nValidation Performance:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    
    return stacking_clf


def main():
    """Main function for neural stacking model to achieve 95% accuracy."""
    print("="*60)
    print("PE FUND SELECTION - NEURAL STACKING MODEL")
    print("Target: 95% Accuracy & 0.95 ROC-AUC")
    print("="*60)
    
    # Load data
    data_path = os.path.join('data', 'raw', 'pe_funds.csv')
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return
    
    # Load and engineer features
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records")
    
    # Apply ultra feature engineering
    df = create_ultra_engineered_features(df)
    
    # Fill any remaining NaN values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # Define target
    irr_75th = df['irr_percent'].quantile(0.75)
    df['is_top_quartile'] = (df['irr_percent'] > irr_75th).astype(int)
    
    print(f"Top quartile threshold: {irr_75th:.1f}% IRR")
    print(f"Positive class ratio: {df['is_top_quartile'].mean():.2%}")
    
    # Select features
    feature_cols = [col for col in df.columns if col not in ['irr_percent', 'is_top_quartile', 'fund_id']]
    
    # Handle categorical variables
    df_encoded = pd.get_dummies(df[feature_cols])
    
    # Create polynomial features for top numeric features
    print("\nCreating polynomial features...")
    numeric_features = df[['fund_size_mm', 'tvpi', 'dpi', 'fund_age_years', 'manager_track_record']]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(numeric_features)
    poly_feature_names = poly.get_feature_names_out(numeric_features.columns)
    
    # Combine with other features
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
    df_final = pd.concat([df_encoded, poly_df], axis=1)
    
    # Remove duplicate columns
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]
    
    print(f"Final feature count: {len(df_final.columns)}")
    
    # Split data
    X = df_final.values
    y = df['is_top_quartile'].values
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"\nDataset splits:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Create neural stacking ensemble
    model = create_neural_stacking_ensemble(X_train, y_train, X_val, y_val)
    
    # Calibrate probabilities
    print("\n" + "="*60)
    print("PROBABILITY CALIBRATION")
    print("="*60)
    calibrated_model = CalibratedClassifierCV(
        model,
        method='isotonic',
        cv=3
    )
    calibrated_model.fit(X_train, y_train)
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    y_pred = calibrated_model.predict(X_test)
    y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
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
    
    # Save model
    print("\n" + "="*60)
    os.makedirs('models', exist_ok=True)
    
    model_path = os.path.join('models', 'pe_fund_selector_neural.pkl')
    joblib.dump(calibrated_model, model_path)
    print(f"Neural stacking model saved to {model_path}")
    
    # Check if target achieved
    print("\n" + "="*60)
    if final_metrics['accuracy'] >= 0.95 and final_metrics['roc_auc'] >= 0.95:
        print("🎉🎉🎉 SUCCESS! TARGET METRICS ACHIEVED! 🎉🎉🎉")
        print(f"   Accuracy: {final_metrics['accuracy']:.2%} ✓")
        print(f"   ROC-AUC: {final_metrics['roc_auc']:.4f} ✓")
        
        # Save best model indicator
        best_model_path = os.path.join('models', 'best_model.pkl')
        joblib.dump(calibrated_model, best_model_path)
        print(f"Best model saved to {best_model_path}")
    else:
        print("⚠️  Approaching target metrics...")
        print(f"   Accuracy: {final_metrics['accuracy']:.2%} (gap: {0.95 - final_metrics['accuracy']:.2%})")
        print(f"   ROC-AUC: {final_metrics['roc_auc']:.4f} (gap: {0.95 - final_metrics['roc_auc']:.4f})")
    
    print("\n" + "="*60)
    print("Neural stacking model training complete!")
    
    return calibrated_model, final_metrics


if __name__ == "__main__":
    main()
