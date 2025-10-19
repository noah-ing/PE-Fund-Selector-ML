# Product Context - PE Fund Selection ML Model

## Project Purpose
Building a machine learning model for Private Equity fund selection that predicts top-performing PE funds based on historical performance data.

## Target Audience
- **Private Equity firms**: For fund selection and portfolio construction
- **Limited Partners (LPs)**: For due diligence and fund screening
- **Quant-focused investment roles**: Demonstrating data-driven decision-making capabilities

## Problem Being Solved
Traditional PE fund selection relies on manual review of 50+ page pitch books and Excel-based analysis. This project automates initial fund screening, reducing analyst review time from 2 weeks to 2 hours for a 100-fund pipeline.

## Key Value Propositions
1. **Quantitative Edge**: Moves beyond Excel-only analysis to data-driven fund selection
2. **Time Efficiency**: Reduces initial screening time by ~50%
3. **Explainable Predictions**: Identifies which fund characteristics drive performance
4. **Actionable Outputs**: Directly supports Investment Committee memos

## How It Should Work
1. **Input**: Historical PE fund data (vintage year, size, sector, geography, track record)
2. **Processing**: Random Forest classifier trained on top-quartile funds (IRR > 75th percentile)
3. **Output**: 
   - Probability of top-quartile performance
   - Feature importance rankings
   - Visual insights for investment decisions

## Success Metrics
- Model accuracy > 80%
- ROC-AUC > 0.85
- Clear feature importance insights
- Working end-to-end pipeline from data to predictions
