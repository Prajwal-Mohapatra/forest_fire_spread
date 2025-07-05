"""
Test runner for the forest fire prediction model.
Runs all unit tests and generates a comprehensive test report.
"""

import unittest
import sys
import os
import time
from io import StringIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from tests.test_preprocessing import TestPreprocessing, TestFireDatasetGenerator
from tests.test_metrics import TestLossFunctions, TestMetrics, TestMetricsIntegration
from tests.test_model import TestASPPBlock, TestResUNetA, TestModelIntegration

try:
    from tests.test_versioning import TestModelVersionManager, TestVersioningFunctions, TestVersioningIntegration
    VERSIONING_AVAILABLE = True
except (ImportError, unittest.SkipTest):
    VERSIONING_AVAILABLE = False
    print("Warning: Versioning tests skipped due to missing dependencies")


def run_test_suite(verbosity=2):
    """
    Run the complete test suite.
    
    Args:
        verbosity (int): Test verbosity level (0=quiet, 1=normal, 2=verbose)
        
    Returns:
        unittest.TestResult: Test results
    """
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add preprocessing tests
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestFireDatasetGenerator))
    
    # Add metrics tests
    suite.addTests(loader.loadTestsFromTestCase(TestLossFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsIntegration))
    
    # Add model tests
    suite.addTests(loader.loadTestsFromTestCase(TestASPPBlock))
    suite.addTests(loader.loadTestsFromTestCase(TestResUNetA))
    suite.addTests(loader.loadTestsFromTestCase(TestModelIntegration))
    
    # Add versioning tests if available
    if VERSIONING_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestModelVersionManager))
        suite.addTests(loader.loadTestsFromTestCase(TestVersioningFunctions))
        suite.addTests(loader.loadTestsFromTestCase(TestVersioningIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    # Print failure details
    if result.failures:
        print(f"\n{'='*70}")
        print("FAILURES")
        print(f"{'='*70}")
        for test, traceback in result.failures:
            print(f"\nFAILED: {test}")
            print(f"{'-'*50}")
            print(traceback)
    
    # Print error details
    if result.errors:
        print(f"\n{'='*70}")
        print("ERRORS")
        print(f"{'='*70}")
        for test, traceback in result.errors:
            print(f"\nERROR: {test}")
            print(f"{'-'*50}")
            print(traceback)
    
    return result


def run_specific_test(test_class_name, test_method_name=None, verbosity=2):
    """
    Run a specific test class or method.
    
    Args:
        test_class_name (str): Name of the test class
        test_method_name (str, optional): Name of specific test method
        verbosity (int): Test verbosity level
        
    Returns:
        unittest.TestResult: Test results
    """
    # Get test class
    test_classes = {
        'TestPreprocessing': TestPreprocessing,
        'TestFireDatasetGenerator': TestFireDatasetGenerator,
        'TestLossFunctions': TestLossFunctions,
        'TestMetrics': TestMetrics,
        'TestMetricsIntegration': TestMetricsIntegration,
        'TestASPPBlock': TestASPPBlock,
        'TestResUNetA': TestResUNetA,
        'TestModelIntegration': TestModelIntegration,
    }
    
    if VERSIONING_AVAILABLE:
        test_classes.update({
            'TestModelVersionManager': TestModelVersionManager,
            'TestVersioningFunctions': TestVersioningFunctions,
            'TestVersioningIntegration': TestVersioningIntegration,
        })
    
    if test_class_name not in test_classes:
        print(f"Error: Test class '{test_class_name}' not found.")
        print(f"Available test classes: {list(test_classes.keys())}")
        return None
    
    test_class = test_classes[test_class_name]
    
    # Create test suite
    if test_method_name:
        suite = unittest.TestSuite()
        suite.addTest(test_class(test_method_name))
    else:
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


def check_dependencies():
    """
    Check if all required dependencies are available.
    
    Returns:
        dict: Dictionary of dependency status
    """
    dependencies = {}
    
    # Core dependencies
    try:
        import numpy
        dependencies['numpy'] = numpy.__version__
    except ImportError:
        dependencies['numpy'] = "NOT AVAILABLE"
    
    try:
        import tensorflow
        dependencies['tensorflow'] = tensorflow.__version__
    except ImportError:
        dependencies['tensorflow'] = "NOT AVAILABLE"
    
    try:
        import rasterio
        dependencies['rasterio'] = rasterio.__version__
    except ImportError:
        dependencies['rasterio'] = "NOT AVAILABLE"
    
    try:
        import yaml
        dependencies['pyyaml'] = yaml.__version__
    except ImportError:
        dependencies['pyyaml'] = "NOT AVAILABLE"
    
    # Optional dependencies
    try:
        import psutil
        dependencies['psutil'] = psutil.__version__
    except ImportError:
        dependencies['psutil'] = "NOT AVAILABLE (optional)"
    
    try:
        import sklearn
        dependencies['scikit-learn'] = sklearn.__version__
    except ImportError:
        dependencies['scikit-learn'] = "NOT AVAILABLE (optional)"
    
    return dependencies


def main():
    """Main test runner function."""
    print("Forest Fire Prediction Model - Test Suite")
    print(f"{'='*70}")
    
    # Check dependencies
    print("Checking dependencies...")
    deps = check_dependencies()
    for dep, version in deps.items():
        status = "✓" if "NOT AVAILABLE" not in version else "✗"
        print(f"  {status} {dep}: {version}")
    
    print(f"\n{'='*70}")
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("Usage:")
            print("  python run_tests.py                    # Run all tests")
            print("  python run_tests.py <TestClassName>    # Run specific test class")
            print("  python run_tests.py <TestClassName> <test_method>  # Run specific test method")
            print("  python run_tests.py --list             # List available test classes")
            return
        
        elif sys.argv[1] == '--list':
            print("Available test classes:")
            test_classes = [
                'TestPreprocessing',                'TestFireDatasetGenerator',
                'TestLossFunctions', 'TestMetrics', 'TestMetricsIntegration',
                'TestASPPBlock', 'TestResUNetA', 'TestModelIntegration'
            ]
            if VERSIONING_AVAILABLE:
                test_classes.extend([
                    'TestModelVersionManager', 'TestVersioningFunctions',
                    'TestVersioningIntegration'
                ])
            
            for test_class in test_classes:
                print(f"  - {test_class}")
            return
        
        else:
            # Run specific test
            test_class = sys.argv[1]
            test_method = sys.argv[2] if len(sys.argv) > 2 else None
            
            print(f"Running specific test: {test_class}")
            if test_method:
                print(f"Test method: {test_method}")
            
            result = run_specific_test(test_class, test_method)
            
            if result:
                exit_code = 0 if result.wasSuccessful() else 1
                sys.exit(exit_code)
            else:
                sys.exit(1)
    
    else:
        # Run all tests
        print("Running complete test suite...")
        result = run_test_suite()
        
        exit_code = 0 if result.wasSuccessful() else 1
        sys.exit(exit_code)


if __name__ == '__main__':
    main()
