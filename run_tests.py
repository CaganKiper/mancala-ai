#!/usr/bin/env python
"""Script to run tests with coverage reporting.

This script runs pytest with coverage reporting for the Mancala AI project.
"""

import os
import subprocess
import sys


def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    # Check if pytest and pytest-cov are installed
    try:
        import pytest
        import pytest_cov
    except ImportError:
        print("Installing pytest and pytest-cov...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"])

    # Run pytest with coverage
    print("Running tests with coverage reporting...")
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "--cov=mancala_ai",
        "--cov-report=term",
        "--cov-report=html:coverage_html",
        "tests/"
    ], capture_output=True, text=True)

    # Print the test results
    print(result.stdout)
    if result.stderr:
        print("Errors:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)

    # Print a message about the coverage report
    if os.path.exists("coverage_html"):
        print("\nCoverage report generated in 'coverage_html' directory.")
        print("Open 'coverage_html/index.html' in a browser to view the report.")

    return result.returncode


if __name__ == "__main__":
    sys.exit(run_tests_with_coverage())