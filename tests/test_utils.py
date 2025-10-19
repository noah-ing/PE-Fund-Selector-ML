"""
Unit tests for utility functions.

Tests input validation and other utility functions
for the PE fund selection model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
from utils import validate_fund_input, format_currency, calculate_performance_tier

class TestFundValidation(unittest.TestCase):
    """Test cases for fund input validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_fund = {
            'vintage_year': 2020,
            'fund_size_mm': 500,
            'sector': 'Technology',
            'geography': 'North America',
            'manager_track_record': 3,
            'tvpi': 2.0,
            'dpi': 1.2,
            'fund_age_years': 3
        }
    
    def test_valid_fund(self):
        """Test validation of a valid fund."""
        is_valid, error = validate_fund_input(self.valid_fund)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_missing_field(self):
        """Test validation with missing required field."""
        invalid_fund = self.valid_fund.copy()
        del invalid_fund['sector']
        is_valid, error = validate_fund_input(invalid_fund)
        self.assertFalse(is_valid)
        self.assertIn('Missing required fields', error)
    
    def test_negative_fund_size(self):
        """Test validation with negative fund size."""
        invalid_fund = self.valid_fund.copy()
        invalid_fund['fund_size_mm'] = -100
        is_valid, error = validate_fund_input(invalid_fund)
        self.assertFalse(is_valid)
        self.assertIn('must be positive', error)
    
    def test_invalid_sector(self):
        """Test validation with invalid sector."""
        invalid_fund = self.valid_fund.copy()
        invalid_fund['sector'] = 'InvalidSector'
        is_valid, error = validate_fund_input(invalid_fund)
        self.assertFalse(is_valid)
        self.assertIn('sector must be one of', error)
    
    def test_invalid_geography(self):
        """Test validation with invalid geography."""
        invalid_fund = self.valid_fund.copy()
        invalid_fund['geography'] = 'InvalidGeo'
        is_valid, error = validate_fund_input(invalid_fund)
        self.assertFalse(is_valid)
        self.assertIn('geography must be one of', error)
    
    def test_dpi_exceeds_tvpi(self):
        """Test validation when DPI exceeds TVPI."""
        invalid_fund = self.valid_fund.copy()
        invalid_fund['tvpi'] = 2.0
        invalid_fund['dpi'] = 3.0
        is_valid, error = validate_fund_input(invalid_fund)
        self.assertFalse(is_valid)
        self.assertIn('DPI cannot exceed TVPI', error)
    
    def test_unrealistic_values(self):
        """Test validation with unrealistic values."""
        # Test unrealistically large fund size
        invalid_fund = self.valid_fund.copy()
        invalid_fund['fund_size_mm'] = 20000  # $20B
        is_valid, error = validate_fund_input(invalid_fund)
        self.assertFalse(is_valid)
        self.assertIn('unrealistically large', error)
        
        # Test unrealistically old fund
        invalid_fund = self.valid_fund.copy()
        invalid_fund['fund_age_years'] = 25
        is_valid, error = validate_fund_input(invalid_fund)
        self.assertFalse(is_valid)
        self.assertIn('unrealistically high', error)

class TestUtilityFunctions(unittest.TestCase):
    """Test cases for other utility functions."""
    
    def test_format_currency_millions(self):
        """Test currency formatting for millions."""
        self.assertEqual(format_currency(500), "$500M")
        self.assertEqual(format_currency(150.5), "$150M")
    
    def test_format_currency_billions(self):
        """Test currency formatting for billions."""
        self.assertEqual(format_currency(1500), "$1.5B")
        self.assertEqual(format_currency(2000), "$2.0B")
    
    def test_performance_tier_classification(self):
        """Test IRR performance tier classification."""
        self.assertEqual(calculate_performance_tier(35), "Top Decile (Elite)")
        self.assertEqual(calculate_performance_tier(27), "Top Quartile (Strong)")
        self.assertEqual(calculate_performance_tier(20), "Above Median (Good)")
        self.assertEqual(calculate_performance_tier(15), "Below Median (Fair)")
        self.assertEqual(calculate_performance_tier(8), "Bottom Quartile (Weak)")
    
    def test_performance_tier_custom_thresholds(self):
        """Test performance tier with custom thresholds."""
        custom_thresholds = {
            'top_decile': 35,
            'top_quartile': 28,
            'above_median': 20,
            'below_median': 15
        }
        self.assertEqual(
            calculate_performance_tier(30, custom_thresholds), 
            "Top Quartile (Strong)"
        )

def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestFundValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*50)
    if result.wasSuccessful():
        print("ALL TESTS PASSED!")
    else:
        print(f"TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
    print("="*50)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
