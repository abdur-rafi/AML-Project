#!/usr/bin/env python3

"""
Simple verification script for the amp_autocast fix.
This script verifies that we've correctly addressed the TypeError.
"""

def test_amp_autocast_fix():
    """Test that the amp_autocast parameter fix is correctly implemented."""
    
    print("="*80)
    print("AMP_AUTOCAST FIX VERIFICATION")
    print("="*80)
    
    success = True
    
    # Test 1: Check function signature in multi_objective.py
    print("1. Checking print_pareto_front_summary_validation function signature...")
    
    try:
        with open("utils/tools/multi_objective.py", 'r') as f:
            content = f.read()
        
        expected_signature = "def print_pareto_front_summary_validation(population, model, loader_eval, args, amp_autocast, generation: int = None):"
        
        if expected_signature in content:
            print("   ✓ Function signature includes amp_autocast parameter")
        else:
            print("   ✗ Function signature missing amp_autocast parameter")
            success = False
            
    except Exception as e:
        print(f"   ✗ Error checking multi_objective.py: {e}")
        success = False
    
    # Test 2: Check function signature in de_multi_objective.py
    print("2. Checking evaluate_validation_objectives function signature...")
    
    try:
        with open("utils/tools/de_multi_objective.py", 'r') as f:
            content = f.read()
        
        expected_signature = "def evaluate_validation_objectives(population, model, loader_eval, args, amp_autocast):"
        
        if expected_signature in content:
            print("   ✓ Function signature includes amp_autocast parameter")
        else:
            print("   ✗ Function signature missing amp_autocast parameter")
            success = False
            
        # Also check that val.validate is called with amp_autocast
        if "val.validate(model, loader_eval, args, amp_autocast=amp_autocast)" in content:
            print("   ✓ val.validate called with amp_autocast parameter")
        else:
            print("   ✗ val.validate not called with amp_autocast parameter")
            success = False
            
    except Exception as e:
        print(f"   ✗ Error checking de_multi_objective.py: {e}")
        success = False
    
    # Test 3: Check function calls in main_cosde.py
    print("3. Checking function calls in main_cosde.py...")
    
    try:
        with open("main_cosde.py", 'r') as f:
            content = f.read()
        
        # Count correct calls
        correct_calls = content.count("print_pareto_front_summary_validation(population, model, loader_eval, args, amp_autocast")
        
        if correct_calls >= 3:
            print(f"   ✓ Found {correct_calls} correct function calls with amp_autocast")
        else:
            print(f"   ✗ Expected at least 3 correct calls, found {correct_calls}")
            success = False
            
    except Exception as e:
        print(f"   ✗ Error checking main_cosde.py: {e}")
        success = False
    
    # Test 4: Check that the original error pattern is fixed
    print("4. Checking that original error pattern is fixed...")
    
    try:
        with open("utils/tools/de_multi_objective.py", 'r') as f:
            content = f.read()
        
        # Check that we're NOT passing None to amp_autocast
        if "amp_autocast=None" in content:
            print("   ✗ Found amp_autocast=None (should be fixed)")
            success = False
        else:
            print("   ✓ No amp_autocast=None found")
            
    except Exception as e:
        print(f"   ✗ Error checking for original error pattern: {e}")
        success = False
    
    print("="*80)
    if success:
        print("✅ ALL CHECKS PASSED!")
        print("The original error 'TypeError: 'NoneType' object is not callable' should be fixed.")
        print("The amp_autocast parameter is now properly passed through the call chain:")
        print("  main_cosde.py → print_pareto_front_summary_validation → evaluate_validation_objectives → val.validate")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("Please review the issues above.")
    print("="*80)
    
    return success

if __name__ == "__main__":
    test_amp_autocast_fix()
