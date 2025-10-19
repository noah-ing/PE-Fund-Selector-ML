# Decisions - PE Fund Selection ML Model

## Key Technical Decisions

### 1. Model Choice: Random Forest Classifier
**Decision**: Use RandomForestClassifier instead of deep learning or XGBoost
**Rationale**: 
- Interpretability is crucial for investment decisions
- Feature importance directly supports IC memos
- Industry standard for financial classification tasks
- Robust to overfitting with small dataset (500 records)
- No need for extensive hyperparameter tuning

### 2. Target Variable: Top Quartile Performance
**Decision**: Binary classification (top quartile vs rest) instead of regression
**Rationale**:
- PE investments are typically binary decisions (invest/pass)
- Top quartile funds (IRR > 75th percentile) represent success in PE
- Easier to communicate to non-technical stakeholders
- Better alignment with LP investment criteria

### 3. Synthetic Data Generation
**Decision**: Generate synthetic data instead of using real PE data
**Rationale**:
- Real PE fund data is proprietary and difficult to obtain
- Allows controlled demonstration of model capabilities
- Realistic distributions based on industry knowledge
- Portfolio project doesn't require actual fund data

### 4. Feature Engineering Approach
**Decision**: Keep features simple and interpretable
**Rationale**:
- PE professionals need to understand feature impact
- Avoid black-box complexity
- Focus on features available in standard pitch books
- One-hot encoding for categorical variables maintains interpretability

### 5. Visualization Focus
**Decision**: Prioritize feature importance and ROC curves
**Rationale**:
- Feature importance directly answers "what drives returns?"
- ROC curve demonstrates model's discrimination ability
- Confusion matrix shows practical classification performance
- All visualizations must be presentation-ready for IC meetings

### 6. Technology Stack
**Decision**: Python with scikit-learn (not TensorFlow/PyTorch)
**Rationale**:
- Standard in quantitative finance
- Easy deployment and maintenance
- Extensive documentation and community support
- Faster development time for portfolio project

### 7. Project Structure
**Decision**: Modular design with separate scripts for each component
**Rationale**:
- Demonstrates software engineering best practices
- Easy to test individual components
- Clear separation of concerns
- Professional portfolio presentation

### 8. Evaluation Metrics
**Decision**: Focus on ROC-AUC as primary metric
**Rationale**:
- Standard metric for imbalanced classification in finance
- Threshold-independent evaluation
- Captures model's ranking ability
- Easily explained to investment professionals

### 9. Documentation Approach
**Decision**: Heavy emphasis on business context in README
**Rationale**:
- Portfolio targets PE/investment roles, not just tech roles
- Must demonstrate understanding of PE industry
- Code quality matters less than business impact
- Focus on time savings and decision support

### 10. Testing Strategy
**Decision**: Basic input validation only, not comprehensive unit tests
**Rationale**:
- Portfolio project, not production software
- Time better spent on features and documentation
- Focus on end-to-end functionality
- Demonstrates practical prioritization skills
