"""
Quantum Deal Hunter ML - AI-powered PE deal sourcing tool
Ultra-competitive system for private equity acquisition target identification
"""

__version__ = "1.0.0"
__author__ = "Quantum Deal Hunter Team"

from .scraper import IndustrialScraper
from .ml_pipeline import MLPipeline, FinBERTAnalyzer
from .gnn_model import CompanyGraphNetwork
from .backtest import BacktestEngine

__all__ = [
    'IndustrialScraper',
    'MLPipeline',
    'FinBERTAnalyzer',
    'CompanyGraphNetwork',
    'BacktestEngine'
]
