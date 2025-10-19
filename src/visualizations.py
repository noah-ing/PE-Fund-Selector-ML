"""
Visualization functions for PE fund selection model.

This module creates publication-ready visualizations for model results,
including feature importance, confusion matrix, ROC curve, and prediction distributions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_feature_importance(importance_df: pd.DataFrame, 
                           save_path: str = 'results/feature_importance.png',
                           top_n: int = 10) -> None:
    """
    Create a horizontal bar chart of feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance_pct' columns
        save_path: Path to save the visualization
        top_n: Number of top features to display
    """
    # Select top features
    top_features = importance_df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(top_features)), 
                   top_features['importance_pct'],
                   color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features))))
    
    # Customize plot
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Feature Importance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Features Driving PE Fund Performance', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_features['importance_pct'])):
        ax.text(value + 0.3, bar.get_y() + bar.get_height()/2, 
               f'{value:.1f}%', 
               va='center', fontsize=10)
    
    # Add grid for readability
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add subtitle with PE context
    fig.text(0.5, 0.02, 
            'Higher importance indicates stronger predictive power for top-quartile IRR performance',
            ha='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to {save_path}")
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         save_path: str = 'results/confusion_matrix.png') -> None:
    """
    Create a heatmap of the confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the visualization
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
               square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    
    # Add custom annotations with counts and percentages
    for i in range(2):
        for j in range(2):
            text = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
            ax.text(j + 0.5, i + 0.5, text,
                   ha='center', va='center', 
                   fontsize=12, fontweight='bold',
                   color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    # Customize plot
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance: Confusion Matrix', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Set tick labels
    labels = ['Bottom 75%', 'Top Quartile']
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11, rotation=0)
    
    # Add performance metrics as text
    accuracy = np.trace(cm) / np.sum(cm)
    precision = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    
    metrics_text = f'Accuracy: {accuracy:.1%}  |  Precision: {precision:.1%}  |  Recall: {recall:.1%}'
    fig.text(0.5, 0.02, metrics_text, 
            ha='center', fontsize=11, fontweight='bold', color='darkblue')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot saved to {save_path}")
    plt.close()

def plot_roc_curve(y_true: np.ndarray, 
                  y_proba: np.ndarray, 
                  save_path: str = 'results/roc_curve.png') -> None:
    """
    Create ROC curve with AUC score.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        save_path: Path to save the visualization
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='#2E86AB', linewidth=2.5,
           label=f'ROC Curve (AUC = {roc_auc:.3f})')
    
    # Plot random classifier line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, 
           label='Random Classifier (AUC = 0.500)')
    
    # Fill area under curve
    ax.fill_between(fpr, 0, tpr, alpha=0.2, color='#2E86AB')
    
    # Customize plot
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve - PE Fund Selection Model', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Set equal aspect ratio
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    
    # Add legend
    ax.legend(loc='lower right', fontsize=11)
    
    # Add interpretation text
    if roc_auc >= 0.9:
        performance = "Excellent"
    elif roc_auc >= 0.8:
        performance = "Good"
    elif roc_auc >= 0.7:
        performance = "Fair"
    else:
        performance = "Poor"
    
    interpretation = f'Model Performance: {performance} ({roc_auc:.3f} AUC)\n'
    interpretation += 'Higher AUC indicates better ability to distinguish top-quartile funds'
    
    fig.text(0.5, 0.02, interpretation,
            ha='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")
    plt.close()

def plot_prediction_distribution(y_true: np.ndarray, 
                                y_proba: np.ndarray,
                                save_path: str = 'results/prediction_distribution.png') -> None:
    """
    Create histogram of predicted probabilities by class.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        save_path: Path to save the visualization
    """
    # Separate probabilities by true class
    proba_negative = y_proba[y_true == 0]
    proba_positive = y_proba[y_true == 1]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot distribution for negative class (Bottom 75%)
    ax1.hist(proba_negative, bins=30, alpha=0.7, color='#E63946', 
            edgecolor='black', linewidth=0.5)
    ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, 
               label='Decision Threshold')
    ax1.set_xlabel('Predicted Probability of Top Quartile', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Funds', fontsize=11, fontweight='bold')
    ax1.set_title('Bottom 75% Funds', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add statistics
    mean_prob_neg = np.mean(proba_negative)
    correctly_classified_neg = np.sum(proba_negative < 0.5) / len(proba_negative)
    ax1.text(0.95, 0.95, f'Mean: {mean_prob_neg:.3f}\nCorrectly Classified: {correctly_classified_neg:.1%}',
            transform=ax1.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot distribution for positive class (Top Quartile)
    ax2.hist(proba_positive, bins=30, alpha=0.7, color='#2A9D8F',
            edgecolor='black', linewidth=0.5)
    ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5,
               label='Decision Threshold')
    ax2.set_xlabel('Predicted Probability of Top Quartile', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Funds', fontsize=11, fontweight='bold')
    ax2.set_title('Top Quartile Funds', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add statistics
    mean_prob_pos = np.mean(proba_positive)
    correctly_classified_pos = np.sum(proba_positive >= 0.5) / len(proba_positive)
    ax2.text(0.05, 0.95, f'Mean: {mean_prob_pos:.3f}\nCorrectly Classified: {correctly_classified_pos:.1%}',
            transform=ax2.transAxes, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Overall title
    fig.suptitle('Distribution of Predicted Probabilities by True Class',
                fontsize=14, fontweight='bold', y=1.05)
    
    # Add interpretation
    fig.text(0.5, -0.02, 
            'Good separation between distributions indicates strong model discrimination ability',
            ha='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Prediction distribution plot saved to {save_path}")
    plt.close()

def create_all_visualizations(model, X_test: np.ndarray, y_test: np.ndarray, 
                             feature_names: list) -> None:
    """
    Generate all visualizations for the model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
    """
    print("\nGenerating visualizations...")
    print("="*50)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Get feature importance
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'importance_pct': importances * 100
    }).sort_values('importance', ascending=False)
    
    # Create visualizations
    plot_feature_importance(importance_df)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    plot_prediction_distribution(y_test, y_proba)
    
    print("\nAll visualizations generated successfully!")
    print("Check the 'results/' directory for saved plots.")

def main():
    """Main function to test visualization generation."""
    # This would typically be called after model training
    print("Visualization module loaded.")
    print("Run model.py to generate visualizations with actual data.")

if __name__ == "__main__":
    main()
