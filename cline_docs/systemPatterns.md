# System Patterns - PE Fund Selection ML Model

## Architecture Overview
```
PE-Fund-Selector-ML/
├── Data Layer (data/)
│   ├── Raw data generation
│   └── Processed data storage
├── Processing Layer (src/)
│   ├── Data preprocessing pipeline
│   ├── Model training and evaluation
│   ├── Visualization generation
│   └── Utility functions
├── Presentation Layer
│   ├── Jupyter notebook (notebooks/)
│   └── Results visualizations (results/)
└── Documentation (README, requirements.txt)
```

## Key Design Patterns

### 1. Pipeline Pattern
**Implementation**: Sequential data flow from raw → processed → model → predictions
```python
data → preprocess() → train_model() → evaluate() → visualize()
```

### 2. Factory Pattern for Data Generation
**Purpose**: Create consistent synthetic data with configurable parameters
```python
def generate_fund_data(n_funds=500, random_state=42):
    # Centralized data generation with consistent distributions
```

### 3. Separation of Concerns
- **data_preprocessing.py**: Data transformation only
- **model.py**: Model training and prediction only
- **visualizations.py**: Plotting functions only
- **utils.py**: Shared utilities and validation

### 4. Configuration Management
**Approach**: Hardcoded constants for portfolio project
```python
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 10
```

### 5. Type Hinting Pattern
**All functions use type hints for clarity**:
```python
def prepare_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:
```

## Data Flow Patterns

### Input → Processing → Output
1. **Raw Data Generation**:
   - CSV file with 500 synthetic funds
   - Realistic distributions for PE metrics

2. **Preprocessing Pipeline**:
   - Load → Clean → Encode → Split → Scale
   - Returns ready-to-train arrays

3. **Model Pipeline**:
   - Train → Evaluate → Extract importance
   - Save model state for predictions

4. **Visualization Pipeline**:
   - Generate → Format → Save to files
   - All charts presentation-ready

## Error Handling Patterns
- Input validation at entry points
- Graceful failures with informative messages
- No complex try-catch blocks (portfolio project)

## Coding Standards

### Function Structure
```python
def function_name(param: type) -> return_type:
    """
    Brief description.
    
    Args:
        param: Description
    
    Returns:
        Description of return value
    """
    # Implementation
```

### Import Organization
```python
# Standard library
import os
from typing import Dict, List, Tuple

# Third-party
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Local
from src.utils import validate_input
```

## Testing Patterns
- Simple assertions for critical paths
- Input validation tests only
- Focus on end-to-end functionality
- Reproducibility through fixed random states

## Performance Considerations
- Use pandas for data manipulation (optimized C backend)
- Vectorized operations over loops
- Appropriate data types (int8 for binary, float32 for features)
- Early stopping not needed (small dataset)

## Scalability Notes
While this is a portfolio project, patterns allow for:
- Easy swap to real data (same CSV format)
- Model replacement (same interface)
- Additional features (extend preprocessing)
- Production deployment (modular structure)

## Anti-Patterns to Avoid
- ❌ Over-engineering for portfolio project
- ❌ Complex inheritance hierarchies
- ❌ Extensive configuration files
- ❌ Abstract base classes
- ❌ Dependency injection
- ❌ Async operations (not needed)
