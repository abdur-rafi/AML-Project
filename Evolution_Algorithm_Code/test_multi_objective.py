#!/usr/bin/env python3
"""
Test script for multi-objective differential evolution implementation.
This script tests the multi-objective optimization components separately.
"""

import torch
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append('/Users/abdurrafi/Desktop/masters/2nd Term/AML/project/AML/CADE4SNN-main/Evolution_Algorithm_Code')

from utils.tools.multi_objective import (
    dominates, fast_non_dominated_sort, calculate_crowding_distance,
    nsga2_selection, calculate_hypervolume, get_pareto_front_indices,
    print_pareto_front_summary, select_diverse_solutions, calculate_spacing_metric
)
from utils.tools.utility import f1_score, accuracy


def test_dominance():
    """Test Pareto dominance function."""
    print("Testing Pareto dominance...")
    
    obj1 = (0.8, 0.75)  # accuracy=0.8, f1=0.75
    obj2 = (0.7, 0.7)   # accuracy=0.7, f1=0.7
    obj3 = (0.9, 0.6)   # accuracy=0.9, f1=0.6
    
    assert dominates(obj1, obj2), "obj1 should dominate obj2"
    assert not dominates(obj2, obj1), "obj2 should not dominate obj1"
    assert not dominates(obj1, obj3), "obj1 should not dominate obj3 (trade-off)"
    assert not dominates(obj3, obj1), "obj3 should not dominate obj1 (trade-off)"
    
    print("✓ Dominance tests passed")


def test_non_dominated_sort():
    """Test non-dominated sorting."""
    print("Testing non-dominated sorting...")
    
    objectives = [
        (0.8, 0.75),  # Pareto optimal
        (0.7, 0.7),   # Dominated by 0
        (0.9, 0.8),   # Pareto optimal (best)
        (0.6, 0.9),   # Pareto optimal
        (0.5, 0.6),   # Dominated
        (0.85, 0.85), # Pareto optimal
    ]
    
    fronts = fast_non_dominated_sort(objectives)
    
    print(f"Number of fronts: {len(fronts)}")
    print(f"Front 0 (Pareto optimal): {fronts[0]}")
    
    # Check that the best solutions are in front 0
    assert 2 in fronts[0], "Solution 2 (0.9, 0.8) should be in front 0"
    assert 0 in fronts[0], "Solution 0 (0.8, 0.75) should be in front 0"
    assert 3 in fronts[0], "Solution 3 (0.6, 0.9) should be in front 0"
    assert 5 in fronts[0], "Solution 5 (0.85, 0.85) should be in front 0"
    
    print("✓ Non-dominated sorting tests passed")


def test_crowding_distance():
    """Test crowding distance calculation."""
    print("Testing crowding distance...")
    
    objectives = [
        (0.8, 0.75),
        (0.9, 0.8),
        (0.6, 0.9),
        (0.85, 0.85),
    ]
    
    front = [0, 1, 2, 3]
    distances = calculate_crowding_distance(objectives, front)
    
    print(f"Crowding distances: {distances}")
    
    # Boundary points should have infinite distance
    assert distances[0] == float('inf') or distances[1] == float('inf') or distances[2] == float('inf') or distances[3] == float('inf'), \
        "At least one boundary point should have infinite distance"
    
    print("✓ Crowding distance tests passed")


def test_nsga2_selection():
    """Test NSGA-II selection."""
    print("Testing NSGA-II selection...")
    
    # Create dummy population
    population = [torch.randn(10) for _ in range(6)]
    objectives = [
        (0.8, 0.75),
        (0.7, 0.7),
        (0.9, 0.8),
        (0.6, 0.9),
        (0.5, 0.6),
        (0.85, 0.85),
    ]
    
    # Select top 4
    selected_pop, selected_obj = nsga2_selection(population, objectives, 4)
    
    assert len(selected_pop) == 4, f"Should select 4 individuals, got {len(selected_pop)}"
    assert len(selected_obj) == 4, f"Should select 4 objectives, got {len(selected_obj)}"
    
    print(f"Selected objectives: {selected_obj}")
    print("✓ NSGA-II selection tests passed")


def test_hypervolume():
    """Test hypervolume calculation."""
    print("Testing hypervolume calculation...")
    
    objectives = [
        (0.8, 0.75),
        (0.9, 0.8),
        (0.6, 0.9),
        (0.85, 0.85),
    ]
    
    hv = calculate_hypervolume(objectives)
    print(f"Hypervolume: {hv}")
    
    assert hv > 0, "Hypervolume should be positive"
    
    print("✓ Hypervolume tests passed")


def test_f1_accuracy_integration():
    """Test F1 score and accuracy computation."""
    print("Testing F1 score and accuracy integration...")
    
    # Create dummy predictions and targets
    batch_size = 32
    num_classes = 10
    
    # Random logits
    output = torch.randn(batch_size, num_classes)
    target = torch.randint(0, num_classes, (batch_size,))
    
    # Compute accuracy
    acc, _ = accuracy(output, target, topk=(1, 5))
    
    # Compute F1 score
    f1 = f1_score(output, target, average='macro', num_classes=num_classes)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    assert 0 <= acc <= 100, f"Accuracy should be between 0 and 100, got {acc}"
    assert 0 <= f1 <= 100, f"F1 score should be between 0 and 100, got {f1}"
    
    print("✓ F1 score and accuracy integration tests passed")


def test_diverse_solution_selection():
    """Test diverse solution selection."""
    print("Testing diverse solution selection...")
    
    population = [torch.randn(10) for _ in range(6)]
    objectives = [
        (0.8, 0.75),
        (0.7, 0.7),
        (0.9, 0.8),
        (0.6, 0.9),
        (0.5, 0.6),
        (0.85, 0.85),
    ]
    
    diverse_solutions = select_diverse_solutions(population, objectives, 3)
    
    assert len(diverse_solutions) <= 3, f"Should select at most 3 solutions, got {len(diverse_solutions)}"
    
    print(f"Selected diverse solutions: {[obj for _, obj in diverse_solutions]}")
    print("✓ Diverse solution selection tests passed")


def test_complete_workflow():
    """Test the complete multi-objective workflow."""
    print("Testing complete multi-objective workflow...")
    
    # Simulate a generation of evolution
    population_size = 10
    population = [torch.randn(20) for _ in range(population_size)]
    
    # Simulate objectives (accuracy, f1_score)
    objectives = []
    for i in range(population_size):
        acc = 0.5 + 0.4 * torch.rand(1).item()  # Random accuracy between 0.5 and 0.9
        f1 = 0.4 + 0.5 * torch.rand(1).item()   # Random F1 between 0.4 and 0.9
        objectives.append((acc, f1))
    
    print(f"Generated {len(objectives)} objectives")
    
    # Test Pareto front analysis
    pareto_indices = get_pareto_front_indices(objectives)
    print(f"Pareto front contains {len(pareto_indices)} solutions")
    
    # Test summary printing
    print_pareto_front_summary(objectives, generation=1)
    
    # Test spacing metric
    spacing = calculate_spacing_metric(objectives)
    print(f"Spacing metric: {spacing:.6f}")
    
    print("✓ Complete workflow test passed")


def main():
    """Run all tests."""
    print("Running multi-objective optimization tests...\n")
    
    try:
        test_dominance()
        test_non_dominated_sort()
        test_crowding_distance()
        test_nsga2_selection()
        test_hypervolume()
        test_f1_accuracy_integration()
        test_diverse_solution_selection()
        test_complete_workflow()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("Multi-objective optimization implementation is working correctly.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
