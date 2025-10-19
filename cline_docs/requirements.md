# Requirements - PE Fund Selection ML Model

## Functional Requirements
1. **Data Generation**: Create synthetic PE fund dataset with 500 records
2. **Feature Set**: 
   - fund_id (unique identifier)
   - vintage_year (2010-2023)
   - fund_size_mm ($100M-$2B)
   - sector (Technology, Healthcare, Energy, Industrials, Consumer, Financial Services)
   - geography (North America, Europe, Asia)
   - manager_track_record (0-5 prior funds)
   - irr_percent (5%-35%)
   - tvpi (1.0x-3.5x)
   - dpi (0.5x-2.5x)
   - fund_age_years (1-10)

3. **Preprocessing Pipeline**:
   - Handle missing values (drop if <5%, median imputation otherwise)
   - One-hot encode categorical features
   - Create binary target: top_quartile (IRR > 75th percentile)
   - 80/20 train/test split with stratification
   - StandardScaler for numerical features

4. **Model Requirements**:
   - RandomForestClassifier (n_estimators=100, max_depth=10)
   - Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC
   - Feature importance extraction
   - Prediction function for new fund inputs

5. **Visualization Requirements**:
   - Feature importance bar chart (top 10)
   - Confusion matrix heatmap
   - ROC curve with AUC score
   - Prediction probability distribution

6. **Documentation**:
   - Professional README with PE-specific insights
   - Jupyter notebook with full pipeline demonstration
   - Type hints and docstrings in all functions
   - Example predictions and use cases

## Technical Requirements
- Python 3.9+
- Libraries: pandas, scikit-learn, matplotlib, seaborn, numpy, jupyter
- Reproducible results (random_state=42)
- Input validation for prediction function
- MIT License

## Quality Standards
- Model accuracy > 80%
- ROC-AUC > 0.85
- All code must run without errors
- Jupyter notebook executes top-to-bottom
- Clear commenting without over-documentation
- Professional visualizations suitable for portfolio presentation

## Deliverables
- Complete project structure in PE-Fund-Selector-ML/
- 4 visualizations saved in results/
- Working prediction function
- GitHub repository with proper .gitignore
- At least 3 meaningful commits showing progression
