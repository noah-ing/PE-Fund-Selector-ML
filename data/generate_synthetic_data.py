"""
Generate synthetic Private Equity fund data for model training.

This script creates realistic PE fund performance data with features
commonly used in fund selection and due diligence processes.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple

# Set random seed for reproducibility
np.random.seed(42)

def generate_fund_data(n_funds: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic PE fund data with realistic distributions.
    
    Args:
        n_funds: Number of fund records to generate
    
    Returns:
        DataFrame containing synthetic PE fund data
    """
    print(f"Generating {n_funds} synthetic PE fund records...")
    
    # Generate fund IDs
    fund_ids = [f"FUND_{str(i+1).zfill(3)}" for i in range(n_funds)]
    
    # Vintage years (2010-2023)
    vintage_years = np.random.choice(range(2010, 2024), n_funds)
    
    # Fund size in millions (log-normal distribution, $100M-$2B range)
    fund_sizes = np.random.lognormal(mean=5.5, sigma=0.7, size=n_funds)
    fund_sizes = np.clip(fund_sizes, 100, 2000)  # Clip to range
    
    # Sectors (weighted distribution based on PE market)
    sectors = ['Technology', 'Healthcare', 'Energy', 'Industrials', 
               'Consumer', 'Financial Services']
    sector_weights = [0.30, 0.20, 0.10, 0.15, 0.15, 0.10]
    fund_sectors = np.random.choice(sectors, n_funds, p=sector_weights)
    
    # Geography (weighted by PE market activity)
    geographies = ['North America', 'Europe', 'Asia']
    geo_weights = [0.50, 0.35, 0.15]
    fund_geographies = np.random.choice(geographies, n_funds, p=geo_weights)
    
    # Manager track record (0-5 prior funds, exponential decay)
    track_records = np.random.exponential(scale=1.5, size=n_funds)
    track_records = np.clip(track_records, 0, 5).astype(int)
    
    # Fund age in years (1-10)
    fund_ages = np.random.uniform(1, 10, n_funds)
    
    # Generate correlated performance metrics
    # Base IRR with sector and geography adjustments
    base_irr = np.random.normal(15, 8, n_funds)  # Mean 15%, std 8%
    
    # Sector performance adjustments
    sector_adjustments = {
        'Technology': 3,
        'Healthcare': 1,
        'Energy': -2,
        'Industrials': 0,
        'Consumer': 0.5,
        'Financial Services': 1.5
    }
    
    # Geography performance adjustments
    geo_adjustments = {
        'North America': 1,
        'Europe': 0,
        'Asia': 2
    }
    
    # Manager experience bonus (better track record = higher returns)
    experience_bonus = track_records * 1.5
    
    # Calculate final IRR
    irr_adjustments = [sector_adjustments[s] for s in fund_sectors]
    geo_adj = [geo_adjustments[g] for g in fund_geographies]
    
    irr_percent = base_irr + irr_adjustments + geo_adj + experience_bonus
    irr_percent = np.clip(irr_percent, 5, 35)  # Realistic range: 5%-35%
    
    # TVPI (Total Value to Paid-In) - correlated with IRR
    # Higher IRR typically means higher TVPI
    tvpi_base = 1.0 + (irr_percent - 5) / 20  # Maps IRR to 1.0x-2.5x range
    tvpi_noise = np.random.normal(0, 0.2, n_funds)
    tvpi = np.clip(tvpi_base + tvpi_noise, 1.0, 3.5)
    
    # DPI (Distributions to Paid-In) - depends on fund age and performance
    # Older funds have higher DPI, better performing funds distribute more
    dpi_base = (fund_ages / 10) * tvpi * 0.7  # Partial realization based on age
    dpi_noise = np.random.normal(0, 0.1, n_funds)
    dpi = np.clip(dpi_base + dpi_noise, 0.5, 2.5)
    
    # Introduce some missing values (realistic for PE data)
    # About 5% missing for DPI (unrealized funds)
    dpi[np.random.choice(n_funds, size=int(n_funds * 0.05), replace=False)] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame({
        'fund_id': fund_ids,
        'vintage_year': vintage_years.astype(int),
        'fund_size_mm': np.round(fund_sizes, 1),
        'sector': fund_sectors,
        'geography': fund_geographies,
        'manager_track_record': track_records,
        'irr_percent': np.round(irr_percent, 2),
        'tvpi': np.round(tvpi, 2),
        'dpi': np.round(dpi, 2),
        'fund_age_years': np.round(fund_ages, 1)
    })
    
    return df

def save_fund_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save fund data to CSV file.
    
    Args:
        df: DataFrame containing fund data
        output_path: Path to save CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total funds: {len(df)}")
    print(f"Date range: {df['vintage_year'].min()}-{df['vintage_year'].max()}")
    print(f"Fund size range: ${df['fund_size_mm'].min():.1f}M - ${df['fund_size_mm'].max():.1f}M")
    print(f"Mean IRR: {df['irr_percent'].mean():.1f}%")
    print(f"Top quartile IRR threshold: {df['irr_percent'].quantile(0.75):.1f}%")
    print(f"Missing values: {df.isnull().sum().sum()} total")
    
    print("\nSector distribution:")
    print(df['sector'].value_counts())
    
    print("\nGeography distribution:")
    print(df['geography'].value_counts())

def main():
    """Main function to generate and save synthetic PE fund data."""
    # Generate data
    df = generate_fund_data(n_funds=5000)
    
    # Define output path
    output_path = os.path.join('data', 'raw', 'pe_funds.csv')
    
    # Save data
    save_fund_data(df, output_path)
    
    # Display first few records
    print("\nFirst 5 records:")
    print(df.head())
    
    print("\nData generation complete!")
    return df

if __name__ == "__main__":
    main()
