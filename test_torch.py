#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script to test PyTorch installation and CUDA compatibility
"""

import sys
import platform

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU only.")
        
    # Test creating a simple tensor
    print("\nCreating a test tensor...")
    x = torch.rand(5, 3)
    print(x)
    
    # Try to move tensor to CUDA if available
    if torch.cuda.is_available():
        print("\nMoving tensor to CUDA...")
        try:
            x = x.cuda()
            print(x)
            print("Successfully moved tensor to CUDA!")
        except Exception as e:
            print(f"Error moving tensor to CUDA: {str(e)}")
    
except ImportError as e:
    print(f"Error importing PyTorch: {str(e)}")
except Exception as e:
    print(f"Unexpected error: {str(e)}")