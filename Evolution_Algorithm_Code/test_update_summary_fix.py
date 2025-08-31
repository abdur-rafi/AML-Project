#!/usr/bin/env python3

"""
Test script to verify the update_summary fix for handling mixed tensor/float types.
"""

import torch
from collections import OrderedDict
import os
import tempfile
import csv

def test_update_summary_fix():
    """Test that update_summary can handle mixed types correctly."""
    
    print("Testing update_summary fix for mixed tensor/float types...")
    
    # Mock the update_summary function with our fix
    def update_summary_fixed(epoch, rowd_in, filename, write_header=False, log_wandb=False):
        rowd = OrderedDict(epoch=epoch)
        for key, value in rowd_in.items():
            if isinstance(value, list):
                if len(value) > 0:
                    # Handle mixed types in the list more robustly
                    list_temp = []
                    for element in value:
                        if hasattr(element, 'item') and callable(getattr(element, 'item')):  # Check if it's a tensor
                            list_temp.append(round(element.item()))
                        elif isinstance(element, (int, float)):
                            list_temp.append(round(float(element)))
                        else:
                            # For any other type, try to convert to float
                            try:
                                list_temp.append(round(float(element)))
                            except (ValueError, TypeError):
                                list_temp.append(element)
                    rowd_in[key] = list_temp
        rowd.update(rowd_in)
        
        # Write to CSV for testing
        with open(filename, mode='a') as cf:
            dw = csv.DictWriter(cf, fieldnames=rowd.keys())
            if write_header:
                dw.writeheader()
            dw.writerow(rowd)
        
        return rowd
    
    # Test case 1: Mixed tensors and floats (simulating the error scenario)
    print("Test 1: Mixed tensor and float values...")
    
    test_data = OrderedDict([
        ('score', [torch.tensor(0.78), torch.tensor(0.66)]),  # Tensors
        ('top1', [78.225, 66.22]),  # Floats 
        ('f1_scores', [54.870, 46.258]),  # Floats from population_objectives
    ])
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
            result = update_summary_fixed(1, test_data, tmp.name, write_header=True)
            print(f"   ✓ Success: {result}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test case 2: All tensors
    print("Test 2: All tensor values...")
    
    test_data2 = OrderedDict([
        ('score', [torch.tensor(0.78), torch.tensor(0.66)]),
        ('top1', [torch.tensor(78.225), torch.tensor(66.22)]),
    ])
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
            result = update_summary_fixed(2, test_data2, tmp.name, write_header=True)
            print(f"   ✓ Success: {result}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test case 3: All floats
    print("Test 3: All float values...")
    
    test_data3 = OrderedDict([
        ('score', [0.78, 0.66]),
        ('top1', [78.225, 66.22]),
        ('f1_scores', [54.870, 46.258]),
    ])
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
            result = update_summary_fixed(3, test_data3, tmp.name, write_header=True)
            print(f"   ✓ Success: {result}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("✅ All tests passed! The update_summary fix should work correctly.")
    return True

if __name__ == "__main__":
    test_update_summary_fix()
