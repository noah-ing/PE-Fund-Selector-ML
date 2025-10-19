"""
Machine Learning model for PE fund selection.

This module implements a Random Forest classifier to predict
top-quartile PE fund performance based on fund characteristics.
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing functions
from data_preprocessing import prepare_data, preprocess_single_fund
from visualizations import create_all_visualizations

def train_model(X_train: np.ndarray, y_train: np.ndarray, 
                random_state: int = 42) -> RandomForestClassifier:
    """
    Train a Random Forest classifier for PE fund selection.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed for reproducibility
    
    Returns:
        Trained RandomForestClassifier model
    """
    print("Training Random Forest model...")
    print(f"Training set shape: {X_train.shape}")
    print(f"Positive class ratio: {np.mean(y_train):.2%}")
    
    # Initialize Random Forest with optimized parameters for PE fund selection
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
        class_weight='balanced'  # Handle class imbalance
    )
    
    # Train the model
    print("Fitting model to training data...")
    model.fit(X_train, y_train)
    
    # Calculate training accuracy
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    print(f"Training accuracy: {train_accuracy:.2%}")
    
    return model

def evaluate_model(model: RandomForestClassifier, 
                  X_test: np.ndarray, 
                  y_test: np.ndarray) -> Dict:
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained RandomForestClassifier
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary containing evaluation metrics
    """
    print("\nEvaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Print results
    print(f"\nTest Set Performance Metrics:")
    print(f"{'='*40}")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1 Score:  {metrics['f1_score']:.2%}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print(f"True Negatives:  {metrics['confusion_matrix'][0,0]}")
    print(f"False Positives: {metrics['confusion_matrix'][0,1]}")
    print(f"False Negatives: {metrics['confusion_matrix'][1,0]}")
    print(f"True Positives:  {metrics['confusion_matrix'][1,1]}")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Bottom 75%', 'Top Quartile']))
    
    return metrics

def get_feature_importance(model: RandomForestClassifier, 
                          feature_names: list) -> pd.DataFrame:
    """
    Extract and rank feature importance from the trained model.
    
    Args:
        model: Trained RandomForestClassifier
        feature_names: List of feature names
    
    Returns:
        DataFrame with features ranked by importance
    """
    print("\nExtracting feature importance...")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Calculate cumulative importance
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
    
    # Add percentage
    importance_df['importance_pct'] = importance_df['importance'] * 100
    
    print(f"\nTop 10 Most Important Features:")
    print(f"{'='*50}")
    for idx, row in importance_df.head(10).iterrows():
        print(f"{row['feature']:<30} {row['importance_pct']:>6.2f}%")
    
    return importance_df

def predict_fund_quality(model: RandomForestClassifier, 
                        scaler, 
                        fund_features: dict,
                        feature_names: list) -> float:
    """
    Predict the probability of a fund being top-quartile.
    
    Args:
        model: Trained RandomForestClassifier
        scaler: Fitted StandardScaler
        fund_features: Dictionary of fund characteristics
        feature_names: List of feature names used in training
    
    Returns:
        Probability of the fund being top-quartile (0-1)
    """
    # Preprocess the fund features
    processed_features = preprocess_single_fund(fund_features, scaler, feature_names)
    
    # Reshape for single prediction
    processed_features = processed_features.reshape(1, -1)
    
    # Get probability of being top-quartile (class 1)
    probability = model.predict_proba(processed_features)[0, 1]
    
    return probability

def save_model(model: RandomForestClassifier, 
               scaler,
               feature_names: list,
               output_dir: str = 'models') -> None:
    """
    Save the trained model and associated objects.
    
    Args:
        model: Trained RandomForestClassifier
        scaler: Fitted StandardScaler
        feature_names: List of feature names
        output_dir: Directory to save model files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'pe_fund_selector_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Save feature names
    features_path = os.path.join(output_dir, 'feature_names.pkl')
    joblib.dump(feature_names, features_path)
    print(f"Feature names saved to {features_path}")

def load_model(model_dir: str = 'models') -> Tuple:
    """
    Load a saved model and associated objects.
    
    Args:
        model_dir: Directory containing saved model files
    
    Returns:
        Tuple of (model, scaler, feature_names)
    """
    model_path = os.path.join(model_dir, 'pe_fund_selector_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    features_path = os.path.join(model_dir, 'feature_names.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    
    return model, scaler, feature_names

def main():
    """Main function to train and evaluate the PE fund selection model."""
    print("PE Fund Selection Model Training")
    print("="*50)
    
    # Prepare data
    data_path = os.path.join('data', 'raw', 'pe_funds.csv')
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Please run generate_synthetic_data.py first.")
        return
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(data_path)
    
    # Train model
    print("\n" + "="*50)
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("\n" + "="*50)
    metrics = evaluate_model(model, X_test, y_test)
    
    # Get feature importance
    print("\n" + "="*50)
    importance_df = get_feature_importance(model, feature_names)
    
    # Save feature importance
    os.makedirs('results', exist_ok=True)
    importance_df.to_csv('results/feature_importance.csv', index=False)
    print("\nFeature importance saved to results/feature_importance.csv")
    
    # Generate visualizations
    print("\n" + "="*50)
    create_all_visualizations(model, X_test, y_test, feature_names)
    
    # Test prediction on a sample fund
    print("\n" + "="*50)
    print("Testing Prediction Function")
    print("-"*50)
    
    # High-quality fund profile
    high_quality_fund = {
        'vintage_year': 2020,
        'fund_size_mm': 800,
        'sector': 'Technology',
        'geography': 'North America',
        'manager_track_record': 4,
        'tvpi': 2.5,
        'dpi': 1.5,
        'fund_age_years': 3
    }
    
    prob_high = predict_fund_quality(model, scaler, high_quality_fund, feature_names)
    print(f"\nHigh-quality fund profile:")
    for key, value in high_quality_fund.items():
        print(f"  {key}: {value}")
    print(f"Probability of top-quartile performance: {prob_high:.1%}")
    
    # Lower-quality fund profile
    lower_quality_fund = {
        'vintage_year': 2015,
        'fund_size_mm': 150,
        'sector': 'Energy',
        'geography': 'Europe',
        'manager_track_record': 0,
        'tvpi': 1.2,
        'dpi': 0.8,
        'fund_age_years': 8
    }
    
    prob_low = predict_fund_quality(model, scaler, lower_quality_fund, feature_names)
    print(f"\nLower-quality fund profile:")
    for key, value in lower_quality_fund.items():
        print(f"  {key}: {value}")
    print(f"Probability of top-quartile performance: {prob_low:.1%}")
    
    # Save model
    print("\n" + "="*50)
    save_model(model, scaler, feature_names)
    
    print("\n" + "="*50)
    print("Model training complete!")
    print(f"\nSummary:")
    print(f"  - Model Accuracy: {metrics['accuracy']:.1%}")
    print(f"  - ROC-AUC Score: {metrics['roc_auc']:.3f}")
    print(f"  - Most important feature: {importance_df.iloc[0]['feature']}")
    print(f"  - Model and artifacts saved to 'models/' directory")
    
    return model, metrics, importance_df

if __name__ == "__main__":
    main()
