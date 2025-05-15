#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Runner script to test AlphaFold integration with AlphaPeptDeep
"""

import os
import sys

def main():
    """
    Run the AlphaFold integration test
    """
    print("Starting AlphaFold integration test...")
    
    # Import and run the test script
    try:
        import test_alphafold_training
        test_alphafold_training.main()
        print("Test completed successfully!")
    except Exception as e:
        print(f"Error running test: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())