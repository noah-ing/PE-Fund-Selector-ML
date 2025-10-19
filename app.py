"""
Hugging Face Spaces deployment file for PE Fund Selector
This is a wrapper for the Streamlit app with HF Spaces compatibility
"""

import streamlit as st
import subprocess
import sys
import os

# Set page config
st.set_page_config(
    page_title="PE Fund Selector - Quant ML",
    page_icon="📈",
    layout="wide"
)

# Check if running on Hugging Face Spaces
if "SPACE_ID" in os.environ:
    st.info("Running on Hugging Face Spaces 🤗")
    
# Import and run the main streamlit app
if __name__ == "__main__":
    # Import the streamlit app
    exec(open('streamlit_app.py').read())
