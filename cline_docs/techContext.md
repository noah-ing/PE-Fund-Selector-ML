# Tech Context - PE Fund Selection ML Model

## Technology Stack

### Core Languages & Frameworks
- **Python 3.9+**: Primary development language
- **Jupyter Notebook**: Interactive development and demonstration

### Data Science Libraries
- **pandas 1.3.0+**: Data manipulation and analysis
- **numpy 1.21.0+**: Numerical computing
- **scikit-learn 1.0.0+**: Machine learning algorithms and utilities

### Visualization Libraries
- **matplotlib 3.4.0+**: Base plotting functionality
- **seaborn 0.11.0+**: Statistical data visualization

## Development Environment
- **OS**: Windows 10
- **IDE**: Visual Studio Code
- **Shell**: Git Bash (MINGW64)
- **Version Control**: Git + GitHub

## Technical Constraints
1. **Dataset Size**: 500 synthetic records (appropriate for portfolio project)
2. **Memory**: No constraints for this scale
3. **Processing**: Single-threaded execution sufficient
4. **Storage**: < 50MB total project size

## Key Technical Specifications

### Data Schema
```python
{
    'fund_id': str,  # Unique identifier (FUND_001, etc.)
    'vintage_year': int,  # 2010-2023
    'fund_size_mm': float,  # 100-2000 (millions)
    'sector': str,  # Categorical: Technology, Healthcare, etc.
    'geography': str,  # Categorical: North America, Europe, Asia
    'manager_track_record': int,  # 0-5 prior funds
    'irr_percent': float,  # 5-35%
    'tvpi': float,  # 1.0-3.5x
    'dpi': float,  # 0.5-2.5x
    'fund_age_years': int  # 1-10
}
```

### Model Configuration
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,  # Use all cores
    class_weight='balanced'  # Handle imbalanced classes
)
```

### Preprocessing Pipeline
```python
Pipeline([
    ('missing', SimpleImputer(strategy='median')),
    ('encoding', OneHotEncoder(sparse=False)),
    ('scaling', StandardScaler()),
    ('splitting', train_test_split(test_size=0.2, random_state=42))
])
```

## Dependencies Management

### requirements.txt Structure
```
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
numpy>=1.21.0
jupyter>=1.0.0
```

### Installation Command
```bash
pip install -r requirements.txt
```

## File Formats
- **Data**: CSV format for portability
- **Models**: Pickle format (joblib) for serialization
- **Visualizations**: PNG format for quality
- **Documentation**: Markdown for GitHub rendering

## Performance Benchmarks
- Data generation: < 2 seconds
- Preprocessing: < 1 second
- Model training: < 5 seconds
- Prediction: < 100ms per fund
- Visualization generation: < 3 seconds total

## Security Considerations
- No sensitive data (using synthetic data)
- No API keys or credentials
- No external data connections
- MIT License for open source sharing

## Deployment Context
- Portfolio demonstration only
- Local execution via Jupyter notebook
- GitHub Pages potential for README showcase
- No production deployment planned

## Browser Requirements (for Jupyter)
- Modern browser (Chrome, Firefox, Safari, Edge)
- JavaScript enabled
- Cookies enabled for notebook state

## Version Control Strategy
- Single main branch (simple portfolio project)
- Meaningful commit messages
- At least 3 commits showing progression
- .gitignore for Python projects

## Testing Approach
- No CI/CD pipeline (portfolio project)
- Manual testing via notebook execution
- Basic input validation only
- Focus on end-to-end functionality

## Known Limitations
- Synthetic data may not capture all real-world patterns
- Model not optimized for production scale
- No real-time data updates
- No API endpoints
- Single-user local execution only

## Future Enhancement Possibilities
- Streamlit web interface
- Real PE data integration
- Advanced models (XGBoost, Neural Networks)
- Time series analysis for vintage trends
- Multi-class classification (quartiles)
- API deployment with FastAPI
