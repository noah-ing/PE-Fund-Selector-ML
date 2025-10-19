"""
Data preprocessing pipeline for PE fund selection model.

This module handles data loading, cleaning, feature engineering,
and preparation for machine learning model training.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def prepare_data(filepath: str, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """
    Prepare PE fund data for model training.
    
    This function performs the complete preprocessing pipeline:
    1. Loads CSV data
    2. Handles missing values
    3. One-hot encodes categorical features
    4. Creates binary target variable (top quartile)
    5. Splits into train/test sets
    6. Scales numerical features
    
    Args:
        filepath: Path to the PE fund CSV file
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple containing:
        - X_train: Training features
        - X_test: Test features
        - y_train: Training labels
        - y_test: Test labels
        - scaler: Fitted StandardScaler object
        - feature_names: List of feature names after preprocessing
    """
    print("Loading PE fund data...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} fund records")
    
    # Handle missing values
    print("\nHandling missing values...")
    missing_pct = df.isnull().sum() / len(df) * 100
    print(f"Missing value percentages:\n{missing_pct[missing_pct > 0]}")
    
    # For DPI, use median imputation (common for unrealized funds)
    numerical_features = ['vintage_year', 'fund_size_mm', 'manager_track_record', 
                         'irr_percent', 'tvpi', 'dpi', 'fund_age_years']
    
    imputer = SimpleImputer(strategy='median')
    df[numerical_features] = imputer.fit_transform(df[numerical_features])
    
    # Create target variable: top_quartile (1 if IRR > 75th percentile)
    irr_75th_percentile = df['irr_percent'].quantile(0.75)
    df['top_quartile'] = (df['irr_percent'] > irr_75th_percentile).astype(int)
    print(f"\nTop quartile IRR threshold: {irr_75th_percentile:.1f}%")
    print(f"Top quartile funds: {df['top_quartile'].sum()} ({df['top_quartile'].mean()*100:.1f}%)")
    
    # Separate features and target
    # We'll use IRR to create the target but not as a feature (data leakage)
    feature_cols = ['vintage_year', 'fund_size_mm', 'sector', 'geography', 
                   'manager_track_record', 'tvpi', 'dpi', 'fund_age_years']
    
    X = df[feature_cols]
    y = df['top_quartile']
    
    # One-hot encode categorical features
    print("\nOne-hot encoding categorical features...")
    categorical_features = ['sector', 'geography']
    numerical_features_for_model = ['vintage_year', 'fund_size_mm', 
                                   'manager_track_record', 'tvpi', 'dpi', 'fund_age_years']
    
    # Create one-hot encoded features
    encoded_features = []
    feature_names = []
    
    for cat_col in categorical_features:
        # Get dummy variables
        dummies = pd.get_dummies(df[cat_col], prefix=cat_col)
        encoded_features.append(dummies)
        feature_names.extend(dummies.columns.tolist())
    
    # Combine numerical and encoded features
    X_numerical = X[numerical_features_for_model]
    feature_names = numerical_features_for_model + feature_names
    
    X_processed = pd.concat([X_numerical] + encoded_features, axis=1)
    
    print(f"Total features after encoding: {X_processed.shape[1]}")
    print(f"Feature names: {feature_names[:10]}..." if len(feature_names) > 10 else f"Feature names: {feature_names}")
    
    # Train/test split (80/20, stratified by target)
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, 
        test_size=0.2, 
        random_state=random_state,
        stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Training set top quartile ratio: {y_train.mean():.2%}")
    print(f"Test set top quartile ratio: {y_test.mean():.2%}")
    
    # Scale numerical features
    print("\nScaling numerical features...")
    scaler = StandardScaler()
    
    # Fit on training data only
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Scale only numerical features (not one-hot encoded)
    X_train_scaled[numerical_features_for_model] = scaler.fit_transform(
        X_train[numerical_features_for_model]
    )
    X_test_scaled[numerical_features_for_model] = scaler.transform(
        X_test[numerical_features_for_model]
    )
    
    # Convert to numpy arrays
    X_train_final = X_train_scaled.values
    X_test_final = X_test_scaled.values
    y_train_final = y_train.values
    y_test_final = y_test.values
    
    print("\nPreprocessing complete!")
    print(f"Final training features shape: {X_train_final.shape}")
    print(f"Final test features shape: {X_test_final.shape}")
    
    return X_train_final, X_test_final, y_train_final, y_test_final, scaler, feature_names

def preprocess_single_fund(fund_data: dict, scaler: StandardScaler, feature_names: List[str]) -> np.ndarray:
    """
    Preprocess a single fund for prediction.
    
    Args:
        fund_data: Dictionary containing fund features
        scaler: Fitted StandardScaler from training
        feature_names: List of expected feature names
    
    Returns:
        Preprocessed feature array ready for prediction
    """
    # Create DataFrame from single fund
    df = pd.DataFrame([fund_data])
    
    # Handle categorical encoding
    categorical_features = ['sector', 'geography']
    numerical_features = ['vintage_year', 'fund_size_mm', 
                         'manager_track_record', 'tvpi', 'dpi', 'fund_age_years']
    
    # Initialize all features with zeros
    processed_features = {name: 0 for name in feature_names}
    
    # Set numerical features
    for feat in numerical_features:
        if feat in fund_data:
            processed_features[feat] = fund_data[feat]
    
    # Set categorical features (one-hot encoding)
    for cat_col in categorical_features:
        if cat_col in fund_data:
            feature_name = f"{cat_col}_{fund_data[cat_col]}"
            if feature_name in processed_features:
                processed_features[feature_name] = 1
    
    # Create DataFrame with correct column order
    df_processed = pd.DataFrame([processed_features])[feature_names]
    
    # Scale numerical features
    df_scaled = df_processed.copy()
    df_scaled[numerical_features] = scaler.transform(df_processed[numerical_features])
    
    return df_scaled.values[0]

def main():
    """Main function to test data preprocessing pipeline."""
    # Path to data file
    data_path = os.path.join('data', 'raw', 'pe_funds.csv')
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Please run generate_synthetic_data.py first.")
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(data_path)
    
    # Test single fund preprocessing
    print("\n" + "="*50)
    print("Testing single fund preprocessing...")
    test_fund = {
        'vintage_year': 2020,
        'fund_size_mm': 500,
        'sector': 'Technology',
        'geography': 'North America',
        'manager_track_record': 3,
        'tvpi': 2.0,
        'dpi': 1.2,
        'fund_age_years': 3
    }
    
    processed_fund = preprocess_single_fund(test_fund, scaler, feature_names)
    print(f"Test fund: {test_fund}")
    print(f"Processed features shape: {processed_fund.shape}")
    print(f"Processed features (first 5): {processed_fund[:5]}")
    
    return X_train, X_test, y_train, y_test, scaler, feature_names

if __name__ == "__main__":
    main()
