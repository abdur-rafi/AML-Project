#!/usr/bin/env python3

"""
Test script to verify that the amp_autocast parameter fix is working correctly.
This script tests the function signatures and imports without requiring the full ML environment.
"""

import sys
import importlib.util

def test_function_signatures():
    """Test that our functions have the correct signatures."""
    
    print("Testing function signatures...")
    
    # Test multi_objective.py function signature
    try:
        # Import the function
        spec = importlib.util.spec_from_file_location(
            "multi_objective", 
            "utils/tools/multi_objective.py"
        )
        multi_objective = importlib.util.module_from_spec(spec)
        
        # Check if we can inspect the function signature
        func = getattr(multi_objective, 'print_pareto_front_summary_validation', None)
        if func:
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            print(f"✓ print_pareto_front_summary_validation parameters: {params}")
            
            # Check if amp_autocast is in the parameters
            if 'amp_autocast' in params:
                print("✓ amp_autocast parameter found in function signature")
            else:
                print("✗ amp_autocast parameter NOT found in function signature")
                return False
        else:
            print("✗ Function print_pareto_front_summary_validation not found")
            return False
            
    except Exception as e:
        print(f"✗ Error testing multi_objective.py: {e}")
        return False
    
    # Test de_multi_objective.py function signature
    try:
        spec = importlib.util.spec_from_file_location(
            "de_multi_objective", 
            "utils/tools/de_multi_objective.py"
        )
        de_multi_objective = importlib.util.module_from_spec(spec)
        
        func = getattr(de_multi_objective, 'evaluate_validation_objectives', None)
        if func:
            import inspect
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            print(f"✓ evaluate_validation_objectives parameters: {params}")
            
            if 'amp_autocast' in params:
                print("✓ amp_autocast parameter found in function signature")
            else:
                print("✗ amp_autocast parameter NOT found in function signature")
                return False
        else:
            print("✗ Function evaluate_validation_objectives not found")
            return False
            
    except Exception as e:
        print(f"✗ Error testing de_multi_objective.py: {e}")
        return False
    
    return True

def test_main_cosde_calls():
    """Test that main_cosde.py has the correct function calls."""
    
    print("\nTesting main_cosde.py function calls...")
    
    try:
        with open("main_cosde.py", 'r') as f:
            content = f.read()
        
        # Check for the correct function calls with amp_autocast
        calls_to_check = [
            "print_pareto_front_summary_validation(population, model, loader_eval, args, amp_autocast, generation=0)",
            "print_pareto_front_summary_validation(population, model, loader_eval, args, amp_autocast, generation=epoch)",
            "print_pareto_front_summary_validation(population, model, loader_eval, args, amp_autocast, generation=\"Final\")"
        ]
        
        for call in calls_to_check:
            if call in content:
                print(f"✓ Found correct call: {call[:50]}...")
            else:
                print(f"✗ Missing correct call: {call[:50]}...")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading main_cosde.py: {e}")
        return False

def main():
    """Run all tests."""
    print("="*80)
    print("AMP_AUTOCAST PARAMETER FIX VERIFICATION")
    print("="*80)
    
    success = True
    
    # Test function signatures
    success &= test_function_signatures()
    
    # Test main_cosde.py calls
    success &= test_main_cosde_calls()
    
    print("\n" + "="*80)
    if success:
        print("✅ ALL TESTS PASSED - amp_autocast fix appears to be working correctly!")
        print("The TypeError: 'NoneType' object is not callable should be resolved.")
    else:
        print("❌ SOME TESTS FAILED - Please check the fixes above")
    print("="*80)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
