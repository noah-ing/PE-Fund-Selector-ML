"""
Utility functions for PE fund selection model.

This module provides helper functions for input validation
and random seed management.
"""

import numpy as np
import random
from typing import Dict, Any, List, Optional

def validate_fund_input(fund_dict: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate input data for a PE fund.
    
    Args:
        fund_dict: Dictionary containing fund characteristics
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if all validations pass
        - error_message: Description of validation error (None if valid)
    """
    # Define required fields
    required_fields = [
        'vintage_year', 'fund_size_mm', 'sector', 'geography',
        'manager_track_record', 'tvpi', 'dpi', 'fund_age_years'
    ]
    
    # Check for missing fields
    missing_fields = [field for field in required_fields if field not in fund_dict]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Validate vintage year
    if not isinstance(fund_dict['vintage_year'], (int, float)):
        return False, "vintage_year must be a number"
    if fund_dict['vintage_year'] < 1990 or fund_dict['vintage_year'] > 2024:
        return False, "vintage_year must be between 1990 and 2024"
    
    # Validate fund size
    if not isinstance(fund_dict['fund_size_mm'], (int, float)):
        return False, "fund_size_mm must be a number"
    if fund_dict['fund_size_mm'] <= 0:
        return False, "fund_size_mm must be positive"
    if fund_dict['fund_size_mm'] > 10000:
        return False, "fund_size_mm seems unrealistically large (>$10B)"
    
    # Validate sector
    valid_sectors = ['Technology', 'Healthcare', 'Energy', 'Industrials', 
                    'Consumer', 'Financial Services']
    if fund_dict['sector'] not in valid_sectors:
        return False, f"sector must be one of: {', '.join(valid_sectors)}"
    
    # Validate geography
    valid_geographies = ['North America', 'Europe', 'Asia']
    if fund_dict['geography'] not in valid_geographies:
        return False, f"geography must be one of: {', '.join(valid_geographies)}"
    
    # Validate manager track record
    if not isinstance(fund_dict['manager_track_record'], (int, float)):
        return False, "manager_track_record must be a number"
    if fund_dict['manager_track_record'] < 0:
        return False, "manager_track_record cannot be negative"
    if fund_dict['manager_track_record'] > 20:
        return False, "manager_track_record seems unrealistically high (>20 funds)"
    
    # Validate TVPI
    if not isinstance(fund_dict['tvpi'], (int, float)):
        return False, "tvpi must be a number"
    if fund_dict['tvpi'] < 0:
        return False, "tvpi cannot be negative"
    if fund_dict['tvpi'] > 10:
        return False, "tvpi seems unrealistically high (>10x)"
    
    # Validate DPI
    if not isinstance(fund_dict['dpi'], (int, float)):
        return False, "dpi must be a number"
    if fund_dict['dpi'] < 0:
        return False, "dpi cannot be negative"
    if fund_dict['dpi'] > 10:
        return False, "dpi seems unrealistically high (>10x)"
    
    # Validate fund age
    if not isinstance(fund_dict['fund_age_years'], (int, float)):
        return False, "fund_age_years must be a number"
    if fund_dict['fund_age_years'] < 0:
        return False, "fund_age_years cannot be negative"
    if fund_dict['fund_age_years'] > 20:
        return False, "fund_age_years seems unrealistically high (>20 years)"
    
    # Logical validation: DPI shouldn't exceed TVPI
    if fund_dict['dpi'] > fund_dict['tvpi']:
        return False, "DPI cannot exceed TVPI (distributions cannot exceed total value)"
    
    return True, None

def set_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Try to set sklearn seed if available
    try:
        import sklearn
        sklearn.random_state = seed
    except:
        pass
    
    print(f"Random seeds set to {seed}")

def format_currency(amount_mm: float) -> str:
    """
    Format fund size in millions to readable currency string.
    
    Args:
        amount_mm: Amount in millions
        
    Returns:
        Formatted currency string
    """
    if amount_mm >= 1000:
        return f"${amount_mm/1000:.1f}B"
    else:
        return f"${amount_mm:.0f}M"

def calculate_performance_tier(irr: float, thresholds: Dict[str, float] = None) -> str:
    """
    Categorize fund performance based on IRR.
    
    Args:
        irr: Internal Rate of Return percentage
        thresholds: Custom thresholds for tiers (optional)
        
    Returns:
        Performance tier string
    """
    if thresholds is None:
        # Default PE industry benchmarks
        thresholds = {
            'top_decile': 30,
            'top_quartile': 25,
            'above_median': 18,
            'below_median': 12
        }
    
    if irr >= thresholds['top_decile']:
        return "Top Decile (Elite)"
    elif irr >= thresholds['top_quartile']:
        return "Top Quartile (Strong)"
    elif irr >= thresholds['above_median']:
        return "Above Median (Good)"
    elif irr >= thresholds['below_median']:
        return "Below Median (Fair)"
    else:
        return "Bottom Quartile (Weak)"

def print_fund_summary(fund_dict: Dict[str, Any], prediction_prob: float = None) -> None:
    """
    Print a formatted summary of fund characteristics.
    
    Args:
        fund_dict: Dictionary of fund features
        prediction_prob: Predicted probability of top-quartile performance (optional)
    """
    print("\n" + "="*50)
    print("FUND SUMMARY")
    print("="*50)
    
    # Basic Information
    print(f"Vintage Year:        {fund_dict['vintage_year']}")
    print(f"Fund Size:           {format_currency(fund_dict['fund_size_mm'])}")
    print(f"Sector:              {fund_dict['sector']}")
    print(f"Geography:           {fund_dict['geography']}")
    
    # Manager Information
    print(f"Manager Experience:  {fund_dict['manager_track_record']} prior funds")
    
    # Performance Metrics
    print(f"TVPI:                {fund_dict['tvpi']:.2f}x")
    print(f"DPI:                 {fund_dict['dpi']:.2f}x")
    print(f"Fund Age:            {fund_dict['fund_age_years']:.1f} years")
    
    # Prediction (if provided)
    if prediction_prob is not None:
        print("\n" + "-"*50)
        print("MODEL PREDICTION")
        print("-"*50)
        print(f"Top Quartile Probability: {prediction_prob:.1%}")
        
        # Interpretation
        if prediction_prob >= 0.75:
            assessment = "Strong Candidate - High likelihood of top-quartile performance"
            recommendation = "RECOMMEND for detailed due diligence"
        elif prediction_prob >= 0.5:
            assessment = "Moderate Candidate - Above-average performance expected"
            recommendation = "CONSIDER for portfolio inclusion"
        elif prediction_prob >= 0.25:
            assessment = "Weak Candidate - Below-average performance likely"
            recommendation = "PROCEED WITH CAUTION"
        else:
            assessment = "Poor Candidate - Low likelihood of strong performance"
            recommendation = "NOT RECOMMENDED"
            
        print(f"Assessment:          {assessment}")
        print(f"Recommendation:      {recommendation}")
    
    print("="*50 + "\n")

def calculate_portfolio_metrics(fund_probabilities: List[float], 
                               fund_sizes: List[float] = None) -> Dict[str, float]:
    """
    Calculate portfolio-level metrics from multiple fund predictions.
    
    Args:
        fund_probabilities: List of top-quartile probabilities
        fund_sizes: List of fund sizes for weighted calculations (optional)
        
    Returns:
        Dictionary of portfolio metrics
    """
    probabilities = np.array(fund_probabilities)
    
    metrics = {
        'mean_probability': np.mean(probabilities),
        'median_probability': np.median(probabilities),
        'std_probability': np.std(probabilities),
        'min_probability': np.min(probabilities),
        'max_probability': np.max(probabilities),
        'high_confidence_funds': np.sum(probabilities >= 0.7),
        'moderate_confidence_funds': np.sum((probabilities >= 0.4) & (probabilities < 0.7)),
        'low_confidence_funds': np.sum(probabilities < 0.4)
    }
    
    # Weighted metrics if fund sizes provided
    if fund_sizes is not None:
        sizes = np.array(fund_sizes)
        weights = sizes / np.sum(sizes)
        metrics['weighted_mean_probability'] = np.average(probabilities, weights=weights)
    
    return metrics

def main():
    """Test utility functions."""
    print("Testing utility functions...")
    
    # Set seeds
    set_seeds(42)
    
    # Test valid fund
    valid_fund = {
        'vintage_year': 2020,
        'fund_size_mm': 500,
        'sector': 'Technology',
        'geography': 'North America',
        'manager_track_record': 3,
        'tvpi': 2.0,
        'dpi': 1.2,
        'fund_age_years': 3
    }
    
    is_valid, error = validate_fund_input(valid_fund)
    print(f"\nValid fund validation: {is_valid}")
    if error:
        print(f"Error: {error}")
    
    # Test invalid fund
    invalid_fund = {
        'vintage_year': 2020,
        'fund_size_mm': -100,  # Invalid: negative size
        'sector': 'Technology',
        'geography': 'North America',
        'manager_track_record': 3,
        'tvpi': 2.0,
        'dpi': 3.0,  # Invalid: DPI > TVPI
        'fund_age_years': 3
    }
    
    is_valid, error = validate_fund_input(invalid_fund)
    print(f"\nInvalid fund validation: {is_valid}")
    if error:
        print(f"Error: {error}")
    
    # Test fund summary
    print_fund_summary(valid_fund, prediction_prob=0.78)
    
    # Test portfolio metrics
    sample_probabilities = [0.82, 0.65, 0.43, 0.91, 0.38, 0.72]
    sample_sizes = [500, 300, 750, 1000, 200, 450]
    
    portfolio_metrics = calculate_portfolio_metrics(sample_probabilities, sample_sizes)
    print("\nPortfolio Metrics:")
    for key, value in portfolio_metrics.items():
        if 'funds' in key:
            print(f"  {key}: {int(value)}")
        else:
            print(f"  {key}: {value:.3f}")
    
    print("\nUtility functions test complete!")

if __name__ == "__main__":
    main()
