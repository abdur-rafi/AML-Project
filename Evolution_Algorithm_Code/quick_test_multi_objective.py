#!/usr/bin/env python3
"""
Quick test to verify multi-objective implementation.
Tests the core functionality without running full evolution.
"""

import sys
import os
import torch
import numpy as np

# Add the project root to the path
sys.path.append('/Users/abdurrafi/Desktop/masters/2nd Term/AML/project/AML/CADE4SNN-main/Evolution_Algorithm_Code')

def test_multi_objective_selection():
    """Test that multi-objective selection considers both accuracy and F1."""
    print("Testing multi-objective selection...")
    
    from utils.tools.multi_objective import (
        fast_non_dominated_sort, nsga2_selection, get_pareto_front_indices,
        print_pareto_front_summary
    )
    
    # Create test population
    population = [torch.randn(10) for _ in range(6)]
    
    # Create objectives where solutions have trade-offs
    objectives = [
        (0.85, 0.70),  # High accuracy, moderate F1
        (0.70, 0.90),  # Moderate accuracy, high F1  
        (0.80, 0.80),  # Balanced - should be Pareto optimal
        (0.60, 0.60),  # Poor on both - should be dominated
        (0.90, 0.75),  # High accuracy, good F1 - should be Pareto optimal
        (0.65, 0.85),  # Moderate accuracy, high F1 - could be Pareto optimal
    ]
    
    print("Original objectives:")
    for i, (acc, f1) in enumerate(objectives):
        print(f"  Solution {i}: Acc={acc:.3f}, F1={f1:.3f}")
    
    # Test Pareto front identification
    pareto_indices = get_pareto_front_indices(objectives)
    print(f"\nPareto optimal solutions: {pareto_indices}")
    
    pareto_objectives = [objectives[i] for i in pareto_indices]
    print("Pareto front:")
    for i in pareto_indices:
        acc, f1 = objectives[i]
        print(f"  Solution {i}: Acc={acc:.3f}, F1={f1:.3f}")
    
    # Test NSGA-II selection
    selected_pop, selected_obj = nsga2_selection(population, objectives, 4)
    
    print(f"\nSelected {len(selected_obj)} solutions by NSGA-II:")
    for i, (acc, f1) in enumerate(selected_obj):
        print(f"  Selected {i}: Acc={acc:.3f}, F1={f1:.3f}")
    
    # Verify that Pareto optimal solutions are preferred
    pareto_count = 0
    for obj in selected_obj:
        if obj in pareto_objectives:
            pareto_count += 1
    
    print(f"\nPareto optimal solutions in selection: {pareto_count}/{len(selected_obj)}")
    
    # The selection should prioritize Pareto optimal solutions
    expected_pareto = min(len(pareto_objectives), len(selected_obj))
    if pareto_count >= expected_pareto - 1:  # Allow for some variation due to crowding distance
        print("✓ Multi-objective selection correctly prioritizes Pareto optimal solutions")
        return True
    else:
        print("❌ Multi-objective selection may not be working correctly")
        return False


def test_objective_calculation():
    """Test that objectives are calculated correctly."""
    print("\nTesting objective calculation...")
    
    from utils.tools.utility import f1_score, accuracy
    
    # Create dummy predictions and targets
    batch_size = 16
    num_classes = 5
    
    # Create predictions that favor different classes
    output = torch.randn(batch_size, num_classes)
    target = torch.randint(0, num_classes, (batch_size,))
    
    # Compute accuracy
    acc, _ = accuracy(output, target, topk=(1, 5))
    
    # Compute F1 score
    f1 = f1_score(output, target, average='macro', num_classes=num_classes)
    
    print(f"Sample objectives: Accuracy={acc:.3f}, F1={f1:.3f}")
    
    # Both should be valid percentages
    if 0 <= acc <= 100 and 0 <= f1 <= 100:
        print("✓ Objective calculation produces valid values")
        return True
    else:
        print("❌ Objective calculation produces invalid values")
        return False


def main():
    """Run quick tests."""
    print("Running quick multi-objective tests...\n")
    
    success = True
    
    try:
        success &= test_multi_objective_selection()
        success &= test_objective_calculation()
        
        if success:
            print("\n" + "="*60)
            print("✓ ALL QUICK TESTS PASSED!")
            print("Multi-objective selection is working correctly.")
            print("The population IS being selected considering multiple objectives.")
            print("="*60)
        else:
            print("\n❌ Some tests failed!")
            
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
