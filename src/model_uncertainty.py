"""
Uncertainty-Aware PE Fund Selection Model
Implements Monte Carlo Dropout and Evidential Deep Learning for confidence bounds

This module adds uncertainty quantification to predictions, providing:
- Prediction intervals with confidence bounds
- Epistemic and aleatoric uncertainty separation
- Calibrated probabilities for risk-aware decisions
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Optional, List
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class MCDropoutNet(nn.Module):
    """
    Neural network with Monte Carlo Dropout for uncertainty estimation.
    Keeps dropout active during inference for uncertainty quantification.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], 
                 dropout_rate: float = 0.2):
        super(MCDropoutNet, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Build architecture
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, 1)
        self.dropout_rate = dropout_rate
        
    def forward(self, x, training=True):
        """
        Forward pass with optional dropout for MC sampling.
        """
        for layer, dropout in zip(self.layers, self.dropouts):
            x = F.relu(layer(x))
            # Apply dropout even during inference for MC sampling
            if training or self.training:
                x = dropout(x)
            else:
                # For deterministic prediction
                x = x
        
        return torch.sigmoid(self.output_layer(x))
    
    def predict_with_uncertainty(self, x, n_samples: int = 100):
        """
        Generate predictions with uncertainty using Monte Carlo sampling.
        
        Returns:
            mean_pred: Mean prediction across MC samples
            std_pred: Standard deviation (epistemic uncertainty)
            percentiles: Confidence intervals (2.5%, 50%, 97.5%)
        """
        self.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, training=True)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions).squeeze()
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Calculate confidence intervals
        percentiles = np.percentile(predictions, [2.5, 50, 97.5], axis=0)
        
        return mean_pred, std_pred, percentiles


class EvidentialNet(nn.Module):
    """
    Evidential Deep Learning network for uncertainty quantification.
    Outputs distribution parameters instead of point predictions.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super(EvidentialNet, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Build hidden layers
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Output 4 parameters for Beta distribution (for binary classification)
        # alpha and beta parameters for positive and negative class
        self.evidence_layer = nn.Linear(prev_dim, 2)
        
    def forward(self, x):
        """
        Forward pass outputting evidence for each class.
        """
        for layer in self.layers:
            x = F.relu(layer(x))
        
        # Output evidence (non-negative)
        evidence = F.softplus(self.evidence_layer(x))
        
        # Add 1 to get alpha parameters for Dirichlet
        alpha = evidence + 1
        
        return alpha
    
    def predict_with_uncertainty(self, x):
        """
        Generate predictions with uncertainty using evidential reasoning.
        
        Returns:
            mean_pred: Expected probability
            epistemic_uncertainty: Uncertainty due to lack of evidence
            aleatoric_uncertainty: Inherent data uncertainty
        """
        alpha = self.forward(x)
        
        # Calculate strength (total evidence)
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # Expected probability (mean of Dirichlet)
        prob = alpha / S
        
        # Epistemic uncertainty (based on total evidence)
        # Less evidence = higher uncertainty
        epistemic = 2 / S  # Simplified measure
        
        # Aleatoric uncertainty (expected variance)
        # Variance of the Dirichlet distribution
        aleatoric = prob * (1 - prob) / (S + 1)
        
        return prob[:, 1], epistemic.squeeze(), aleatoric[:, 1]


class UncertaintyQuantifier:
    """
    Main class for uncertainty-aware predictions combining MC Dropout and Evidential Learning.
    """
    
    def __init__(self, input_dim: int, model_type: str = 'mc_dropout'):
        """
        Initialize uncertainty quantifier.
        
        Args:
            input_dim: Number of input features
            model_type: 'mc_dropout', 'evidential', or 'ensemble'
        """
        self.input_dim = input_dim
        self.model_type = model_type
        
        if model_type == 'mc_dropout':
            self.model = MCDropoutNet(input_dim)
        elif model_type == 'evidential':
            self.model = EvidentialNet(input_dim)
        elif model_type == 'ensemble':
            # Ensemble of both approaches
            self.mc_model = MCDropoutNet(input_dim)
            self.ev_model = EvidentialNet(input_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.model_type != 'ensemble':
            self.model.to(self.device)
        else:
            self.mc_model.to(self.device)
            self.ev_model.to(self.device)
    
    def train(self, X_train, y_train, X_val, y_val, epochs: int = 100, batch_size: int = 32):
        """
        Train the uncertainty model.
        """
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if self.model_type == 'ensemble':
            self._train_ensemble(train_loader, X_val_t, y_val_t, epochs)
        else:
            self._train_single(train_loader, X_val_t, y_val_t, epochs)
    
    def _train_single(self, train_loader, X_val, y_val, epochs):
        """Train a single model (MC Dropout or Evidential)."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        if self.model_type == 'mc_dropout':
            criterion = nn.BCELoss()
        else:  # evidential
            criterion = self._evidential_loss
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                if self.model_type == 'mc_dropout':
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                else:  # evidential
                    alpha = self.model(X_batch)
                    loss = criterion(y_batch, alpha)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                val_loss = self._validate(X_val, y_val)
                print(f"Epoch {epoch}: Train Loss = {total_loss/len(train_loader):.4f}, "
                      f"Val Loss = {val_loss:.4f}")
    
    def _train_ensemble(self, train_loader, X_val, y_val, epochs):
        """Train ensemble of MC Dropout and Evidential models."""
        # Train MC Dropout model
        self.model = self.mc_model
        self.model_type = 'mc_dropout'
        self._train_single(train_loader, X_val, y_val, epochs // 2)
        
        # Train Evidential model
        self.model = self.ev_model
        self.model_type = 'evidential'
        self._train_single(train_loader, X_val, y_val, epochs // 2)
        
        self.model_type = 'ensemble'
    
    def _evidential_loss(self, y, alpha, lambda_reg=0.01):
        """
        Loss function for evidential learning.
        Combines cross-entropy with evidence regularization.
        """
        S = torch.sum(alpha, dim=1, keepdim=True)
        prob = alpha / S
        
        # Cross-entropy loss
        ce_loss = -y * torch.log(prob[:, 1:2] + 1e-7) - (1-y) * torch.log(prob[:, 0:1] + 1e-7)
        
        # KL divergence regularization (encourages lower evidence for wrong predictions)
        # This helps calibrate the uncertainty
        alpha_tilde = y * (alpha[:, 1:2] - 1) + (1-y) * (alpha[:, 0:1] - 1)
        kl_loss = torch.sum(alpha_tilde * (torch.digamma(alpha_tilde) - torch.digamma(S)), dim=1, keepdim=True)
        
        return torch.mean(ce_loss + lambda_reg * kl_loss)
    
    def _validate(self, X_val, y_val):
        """Calculate validation loss."""
        self.model.eval()
        
        with torch.no_grad():
            if self.model_type == 'mc_dropout':
                outputs = self.model(X_val, training=False)
                criterion = nn.BCELoss()
                loss = criterion(outputs, y_val)
            else:  # evidential
                alpha = self.model(X_val)
                loss = self._evidential_loss(y_val, alpha)
        
        return loss.item()
    
    def predict_with_uncertainty(self, X, confidence_level: float = 0.95):
        """
        Generate predictions with comprehensive uncertainty quantification.
        
        Returns:
            predictions: Dict containing:
                - point_estimate: Best guess prediction
                - lower_bound: Lower confidence bound
                - upper_bound: Upper confidence bound
                - epistemic_uncertainty: Model uncertainty
                - aleatoric_uncertainty: Data uncertainty
                - confidence_score: Overall confidence in prediction
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        if self.model_type == 'mc_dropout':
            return self._predict_mc_dropout(X_tensor, confidence_level)
        elif self.model_type == 'evidential':
            return self._predict_evidential(X_tensor, confidence_level)
        else:  # ensemble
            return self._predict_ensemble(X_tensor, confidence_level)
    
    def _predict_mc_dropout(self, X, confidence_level):
        """Predictions using MC Dropout."""
        mean_pred, std_pred, percentiles = self.mc_model.predict_with_uncertainty(X, n_samples=100)
        
        # Calculate confidence interval
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        predictions = {
            'point_estimate': mean_pred,
            'lower_bound': percentiles[0],  # 2.5 percentile
            'upper_bound': percentiles[2],  # 97.5 percentile
            'epistemic_uncertainty': std_pred,
            'aleatoric_uncertainty': np.zeros_like(std_pred),  # MC Dropout doesn't separate
            'confidence_score': 1 - std_pred,  # Simple confidence measure
            'prediction_interval_width': percentiles[2] - percentiles[0]
        }
        
        return predictions
    
    def _predict_evidential(self, X, confidence_level):
        """Predictions using Evidential Learning."""
        prob, epistemic, aleatoric = self.ev_model.predict_with_uncertainty(X)
        
        prob = prob.cpu().numpy()
        epistemic = epistemic.cpu().numpy()
        aleatoric = aleatoric.cpu().numpy()
        
        # Calculate confidence bounds using total uncertainty
        total_uncertainty = np.sqrt(epistemic**2 + aleatoric**2)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        predictions = {
            'point_estimate': prob,
            'lower_bound': np.maximum(0, prob - z_score * total_uncertainty),
            'upper_bound': np.minimum(1, prob + z_score * total_uncertainty),
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'confidence_score': 1 / (1 + total_uncertainty),
            'prediction_interval_width': 2 * z_score * total_uncertainty
        }
        
        return predictions
    
    def _predict_ensemble(self, X, confidence_level):
        """Predictions using ensemble of MC Dropout and Evidential."""
        # Get predictions from both models
        self.model = self.mc_model
        mc_preds = self._predict_mc_dropout(X, confidence_level)
        
        self.model = self.ev_model  
        ev_preds = self._predict_evidential(X, confidence_level)
        
        # Combine predictions (weighted average)
        mc_weight = 0.5
        ev_weight = 0.5
        
        predictions = {
            'point_estimate': (mc_weight * mc_preds['point_estimate'] + 
                             ev_weight * ev_preds['point_estimate']),
            'lower_bound': (mc_weight * mc_preds['lower_bound'] + 
                          ev_weight * ev_preds['lower_bound']),
            'upper_bound': (mc_weight * mc_preds['upper_bound'] + 
                          ev_weight * ev_preds['upper_bound']),
            'epistemic_uncertainty': (mc_weight * mc_preds['epistemic_uncertainty'] + 
                                    ev_weight * ev_preds['epistemic_uncertainty']),
            'aleatoric_uncertainty': ev_preds['aleatoric_uncertainty'],  # Only from evidential
            'confidence_score': (mc_weight * mc_preds['confidence_score'] + 
                               ev_weight * ev_preds['confidence_score']),
            'prediction_interval_width': (mc_weight * mc_preds['prediction_interval_width'] + 
                                        ev_weight * ev_preds['prediction_interval_width'])
        }
        
        self.model_type = 'ensemble'
        return predictions
    
    def calibration_score(self, X, y, n_bins: int = 10):
        """
        Calculate Expected Calibration Error (ECE) for model calibration.
        Lower is better (perfect calibration = 0).
        """
        predictions = self.predict_with_uncertainty(X)
        pred_probs = predictions['point_estimate']
        
        # Bin predictions
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (pred_probs > bin_lower) & (pred_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y[in_bin].mean()
                avg_confidence_in_bin = pred_probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


def demonstrate_uncertainty_predictions():
    """
    Demonstrate uncertainty-aware predictions on PE fund data.
    """
    print("="*60)
    print("UNCERTAINTY-AWARE PE FUND PREDICTIONS")
    print("Providing Confidence Bounds for Investment Decisions")
    print("="*60)
    
    # Load data
    import os
    data_path = os.path.join('data', 'raw', 'pe_funds.csv')
    
    if not os.path.exists(data_path):
        print("Please ensure PE fund data exists at", data_path)
        return
    
    # Prepare data using existing preprocessing
    from data_preprocessing import prepare_data
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(data_path)
    
    # Create validation split
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nDataset: {len(X_train_final)} train, {len(X_val)} val, {len(X_test)} test")
    print(f"Features: {X_train.shape[1]}")
    
    # Train MC Dropout model
    print("\n" + "="*40)
    print("Training MC Dropout Model")
    print("="*40)
    
    mc_model = UncertaintyQuantifier(X_train.shape[1], model_type='mc_dropout')
    mc_model.train(X_train_final, y_train_final, X_val, y_val, epochs=50)
    
    # Train Evidential model
    print("\n" + "="*40)
    print("Training Evidential Deep Learning Model")
    print("="*40)
    
    ev_model = UncertaintyQuantifier(X_train.shape[1], model_type='evidential')
    ev_model.train(X_train_final, y_train_final, X_val, y_val, epochs=50)
    
    # Make predictions with uncertainty
    print("\n" + "="*40)
    print("UNCERTAINTY-AWARE PREDICTIONS")
    print("="*40)
    
    # Select a few test samples for demonstration
    n_demos = 5
    X_demo = X_test[:n_demos]
    y_demo = y_test[:n_demos]
    
    print("\nMC Dropout Predictions:")
    print("-"*40)
    mc_preds = mc_model.predict_with_uncertainty(X_demo)
    
    for i in range(n_demos):
        print(f"\nFund {i+1} (True: {'Top' if y_demo[i] else 'Not Top'}):")
        print(f"  Prediction: {mc_preds['point_estimate'][i]:.1%}")
        print(f"  95% CI: [{mc_preds['lower_bound'][i]:.1%}, {mc_preds['upper_bound'][i]:.1%}]")
        print(f"  Epistemic Uncertainty: {mc_preds['epistemic_uncertainty'][i]:.3f}")
        print(f"  Confidence Score: {mc_preds['confidence_score'][i]:.2%}")
    
    print("\n" + "="*40)
    print("Evidential Deep Learning Predictions:")
    print("-"*40)
    ev_preds = ev_model.predict_with_uncertainty(X_demo)
    
    for i in range(n_demos):
        print(f"\nFund {i+1} (True: {'Top' if y_demo[i] else 'Not Top'}):")
        print(f"  Prediction: {ev_preds['point_estimate'][i]:.1%}")
        print(f"  95% CI: [{ev_preds['lower_bound'][i]:.1%}, {ev_preds['upper_bound'][i]:.1%}]")
        print(f"  Epistemic Uncertainty: {ev_preds['epistemic_uncertainty'][i]:.3f}")
        print(f"  Aleatoric Uncertainty: {ev_preds['aleatoric_uncertainty'][i]:.3f}")
        print(f"  Confidence Score: {ev_preds['confidence_score'][i]:.2%}")
    
    # Calculate calibration
    print("\n" + "="*40)
    print("MODEL CALIBRATION ASSESSMENT")
    print("="*40)
    
    mc_ece = mc_model.calibration_score(X_test, y_test)
    ev_ece = ev_model.calibration_score(X_test, y_test)
    
    print(f"MC Dropout ECE: {mc_ece:.4f} (lower is better)")
    print(f"Evidential ECE: {ev_ece:.4f} (lower is better)")
    
    # Save models
    print("\n" + "="*40)
    print("Saving Uncertainty Models")
    print("="*40)
    
    os.makedirs('models', exist_ok=True)
    
    # Save PyTorch models
    torch.save(mc_model.model.state_dict(), 'models/mc_dropout_model.pth')
    torch.save(ev_model.model.state_dict(), 'models/evidential_model.pth')
    
    # Save model configs
    model_configs = {
        'input_dim': X_train.shape[1],
        'mc_dropout_calibration': float(mc_ece),
        'evidential_calibration': float(ev_ece)
    }
    joblib.dump(model_configs, 'models/uncertainty_configs.pkl')
    
    print("Models saved successfully!")
    
    return mc_model, ev_model, mc_preds, ev_preds


if __name__ == "__main__":
    # Run demonstration
    mc_model, ev_model, mc_preds, ev_preds = demonstrate_uncertainty_predictions()
    
    print("\n" + "="*60)
    print("UNCERTAINTY QUANTIFICATION COMPLETE")
    print("Models provide confidence bounds for risk-aware decisions")
    print("="*60)
