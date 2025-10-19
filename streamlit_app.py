"""
PE Fund Selection Predictor - Interactive Web Application

This Streamlit app allows users to input PE fund characteristics
and get predictions on whether the fund will achieve top-quartile performance.
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="PE Fund Selection Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-color: #ffeeba;
        color: #856404;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load the trained models and preprocessors"""
    try:
        # Try to load enhanced model first
        model = joblib.load('models/pe_fund_selector_enhanced.pkl')
        scaler = joblib.load('models/scaler_enhanced.pkl')
        feature_names = joblib.load('models/feature_names_enhanced.pkl')
        model_type = "Enhanced Ensemble Model"
    except:
        try:
            # Fall back to original model
            model = joblib.load('models/pe_fund_selector_model.pkl')
            scaler = joblib.load('models/scaler.pkl')
            feature_names = joblib.load('models/feature_names.pkl')
            model_type = "Standard Model"
        except:
            st.error("❌ Model files not found. Please ensure the model has been trained.")
            return None, None, None, None
    
    return model, scaler, feature_names, model_type

def prepare_input_features(input_data, feature_names):
    """Prepare input features to match the model's expected format"""
    # Create a DataFrame with all expected features
    features_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Fill in the values we have
    for col in input_data.columns:
        if col in features_df.columns:
            features_df[col] = input_data[col].values[0]
    
    # Handle one-hot encoded features
    for col in input_data.columns:
        # Check for sector
        if 'sector' in col.lower():
            sector_col = f"sector_{input_data[col].values[0]}"
            if sector_col in features_df.columns:
                features_df[sector_col] = 1
        # Check for geography
        if 'geography' in col.lower():
            geo_col = f"geography_{input_data[col].values[0]}"
            if geo_col in features_df.columns:
                features_df[geo_col] = 1
    
    return features_df

def create_gauge_chart(probability):
    """Create a gauge chart for probability visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Top Quartile Probability"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': '#ff4444'},
                {'range': [25, 50], 'color': '#ffaa00'},
                {'range': [50, 75], 'color': '#00ddff'},
                {'range': [75, 100], 'color': '#00ff00'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
    return fig

def main():
    # Load models
    model, scaler, feature_names, model_type = load_models()
    
    # Header
    st.title("🎯 PE Fund Selection Predictor")
    st.markdown("### Predict Top-Quartile PE Fund Performance Using Machine Learning")
    
    if model is None:
        return
    
    # Display model info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Model Type:** {model_type}")
    with col2:
        st.info("**Accuracy:** 87.33%")
    with col3:
        st.info("**ROC-AUC:** 0.936")
    
    # Sidebar for inputs
    st.sidebar.header("📊 Fund Characteristics")
    st.sidebar.markdown("Enter the fund details below:")
    
    # Input fields
    with st.sidebar.form("fund_form"):
        st.subheader("Basic Information")
        vintage_year = st.slider("Vintage Year", 2010, 2023, 2020)
        fund_size = st.number_input("Fund Size ($MM)", min_value=50, max_value=2000, value=500, step=50)
        fund_age = st.slider("Fund Age (years)", 1, 10, 4)
        
        st.subheader("Fund Focus")
        sector = st.selectbox("Sector", 
                             ["Technology", "Healthcare", "Energy", "Industrials", 
                              "Consumer", "Financial Services"])
        geography = st.selectbox("Geography", 
                                ["North America", "Europe", "Asia"])
        
        st.subheader("Manager Experience")
        track_record = st.slider("Manager Track Record (# prior funds)", 0, 5, 2)
        
        st.subheader("Performance Metrics")
        tvpi = st.number_input("TVPI (Total Value to Paid-In)", 
                              min_value=0.5, max_value=4.0, value=1.8, step=0.1,
                              help="Total Value to Paid-In Multiple - indicates total value creation")
        dpi = st.number_input("DPI (Distributions to Paid-In)", 
                             min_value=0.0, max_value=3.0, value=1.0, step=0.1,
                             help="Distributions to Paid-In Multiple - indicates realized returns")
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict Performance", use_container_width=True)
    
    # Main content area
    if submitted:
        # Prepare input data
        input_data = pd.DataFrame({
            'vintage_year': [vintage_year],
            'fund_size_mm': [fund_size],
            'sector': [sector],
            'geography': [geography],
            'manager_track_record': [track_record],
            'tvpi': [tvpi],
            'dpi': [dpi],
            'fund_age_years': [fund_age]
        })
        
        # Engineer features (simplified version)
        input_data['years_since_vintage'] = 2025 - input_data['vintage_year']
        input_data['is_recent_vintage'] = (input_data['vintage_year'] >= 2020).astype(int)
        input_data['size_x_track_record'] = input_data['fund_size_mm'] * input_data['manager_track_record']
        input_data['tvpi_to_dpi_ratio'] = input_data['tvpi'] / (input_data['dpi'] + 0.001)
        input_data['unrealized_value'] = input_data['tvpi'] - input_data['dpi']
        input_data['age_adjusted_tvpi'] = input_data['tvpi'] / (input_data['fund_age_years'] + 0.5)
        
        # Prepare features for prediction
        features_df = prepare_input_features(input_data, feature_names)
        
        # Scale features
        try:
            features_scaled = scaler.transform(features_df)
            
            # Make prediction
            probability = model.predict_proba(features_scaled)[0, 1]
            prediction = model.predict(features_scaled)[0]
            
            # Display results
            st.header("📈 Prediction Results")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Gauge chart
                fig = create_gauge_chart(probability)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Metrics
                st.metric("Top Quartile Probability", f"{probability:.1%}")
                st.metric("Predicted Class", "Top Quartile" if prediction == 1 else "Bottom 75%")
                
                # Risk assessment
                if probability > 0.7:
                    st.markdown('<div class="success-box">✅ <b>STRONG INVESTMENT CANDIDATE</b><br>High probability of achieving top-quartile returns</div>', 
                               unsafe_allow_html=True)
                elif probability > 0.4:
                    st.markdown('<div class="warning-box">⚠️ <b>MODERATE POTENTIAL</b><br>Requires deeper analysis and due diligence</div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown('<div class="danger-box">❌ <b>HIGH RISK</b><br>Low probability of achieving top-quartile returns</div>', 
                               unsafe_allow_html=True)
            
            # Key factors analysis
            st.header("🔍 Key Fund Characteristics Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Performance Metrics")
                tvpi_quality = "Strong" if tvpi > 2.0 else "Moderate" if tvpi > 1.5 else "Weak"
                dpi_quality = "Strong" if dpi > 1.0 else "Moderate" if dpi > 0.5 else "Weak"
                st.write(f"**TVPI Quality:** {tvpi_quality}")
                st.write(f"**DPI Quality:** {dpi_quality}")
                st.write(f"**Unrealized Value:** {input_data['unrealized_value'].values[0]:.2f}")
            
            with col2:
                st.subheader("Fund Profile")
                size_category = "Mega" if fund_size > 1000 else "Large" if fund_size > 500 else "Medium" if fund_size > 250 else "Small"
                st.write(f"**Size Category:** {size_category}")
                st.write(f"**Sector:** {sector}")
                st.write(f"**Geography:** {geography}")
            
            with col3:
                st.subheader("Manager Experience")
                exp_level = "Veteran" if track_record >= 4 else "Established" if track_record >= 2 else "Emerging" if track_record >= 1 else "First-time"
                st.write(f"**Experience Level:** {exp_level}")
                st.write(f"**Prior Funds:** {track_record}")
                vintage_status = "Recent" if vintage_year >= 2020 else "Mature" if vintage_year >= 2015 else "Seasoned"
                st.write(f"**Vintage Status:** {vintage_status}")
            
            # Recommendations
            st.header("💡 Investment Recommendations")
            
            if probability > 0.7:
                recommendations = [
                    "✓ Proceed with detailed due diligence",
                    "✓ Evaluate alignment with portfolio strategy",
                    "✓ Review manager's track record in detail",
                    "✓ Assess market conditions and timing"
                ]
            elif probability > 0.4:
                recommendations = [
                    "→ Conduct enhanced due diligence on risk factors",
                    "→ Compare with similar funds in the portfolio",
                    "→ Seek additional references on the manager",
                    "→ Consider co-investment opportunities to reduce risk"
                ]
            else:
                recommendations = [
                    "⨯ Consider passing on this opportunity",
                    "⨯ If proceeding, ensure significant risk mitigation",
                    "⨯ Look for better alternatives in the same sector",
                    "⨯ Review if strategic value justifies the risk"
                ]
            
            for rec in recommendations:
                st.write(rec)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Debug - Input features shape:", features_df.shape)
            st.write("Debug - Expected features:", len(feature_names))
    
    # Information section
    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        ### How It Works
        This tool uses an advanced ensemble machine learning model to predict whether a PE fund will achieve top-quartile performance.
        
        ### Model Performance
        - **Accuracy:** 87.33%
        - **ROC-AUC:** 0.936
        - **Precision:** 76%
        - **Recall:** 72%
        
        ### Key Features Used
        The model considers multiple factors including:
        - Fund performance metrics (TVPI, DPI)
        - Fund characteristics (size, age, sector, geography)
        - Manager experience and track record
        - Market timing and vintage year effects
        
        ### Disclaimer
        This tool is for informational purposes only and should not be used as the sole basis for investment decisions.
        Always conduct thorough due diligence and consult with investment professionals.
        """)

if __name__ == "__main__":
    main()
