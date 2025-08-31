"""
Multi-objective optimization utilities for evolutionary algorithms.
Implements Pareto dominance, NSGA-II operations, and multi-objective metrics.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Set
from collections import defaultdict


def dominates(obj1: Tuple[float, float], obj2: Tuple[float, float]) -> bool:
    """
    Check if obj1 Pareto dominates obj2.
    For maximization problems (both accuracy and F1 score should be maximized).
    
    Args:
        obj1: First objective tuple (accuracy, f1_score)
        obj2: Second objective tuple (accuracy, f1_score)
        
    Returns:
        True if obj1 dominates obj2, False otherwise
    """
    # obj1 dominates obj2 if it's at least as good in all objectives and strictly better in at least one
    better_in_all = obj1[0] >= obj2[0] and obj1[1] >= obj2[1]
    better_in_one = obj1[0] > obj2[0] or obj1[1] > obj2[1]
    return better_in_all and better_in_one


def fast_non_dominated_sort(objectives: List[Tuple[float, float]]) -> List[List[int]]:
    """
    NSGA-II fast non-dominated sorting algorithm.
    
    Args:
        objectives: List of objective tuples (accuracy, f1_score) for each individual
        
    Returns:
        List of fronts, where each front is a list of individual indices
    """
    n = len(objectives)
    domination_count = [0] * n  # Number of individuals that dominate individual i
    dominated_solutions = [[] for _ in range(n)]  # Individuals dominated by individual i
    fronts = [[]]
    
    # For each individual
    for i in range(n):
        for j in range(n):
            if i != j:
                if dominates(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                elif dominates(objectives[j], objectives[i]):
                    domination_count[i] += 1
        
        # If no one dominates this individual, it belongs to first front
        if domination_count[i] == 0:
            fronts[0].append(i)
    
    # Generate subsequent fronts
    current_front = 0
    while current_front < len(fronts) and len(fronts[current_front]) > 0:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        current_front += 1
    
    # Remove empty fronts
    fronts = [front for front in fronts if front]
    return fronts


def calculate_crowding_distance(objectives: List[Tuple[float, float]], front: List[int]) -> Dict[int, float]:
    """
    Calculate crowding distance for individuals in a front.
    
    Args:
        objectives: List of objective tuples for all individuals
        front: List of individual indices in this front
        
    Returns:
        Dictionary mapping individual index to crowding distance
    """
    distances = {i: 0.0 for i in front}
    
    if len(front) <= 2:
        for i in front:
            distances[i] = float('inf')
        return distances
    
    # For each objective
    for obj_idx in range(2):  # 2 objectives: accuracy and F1
        # Sort front by this objective
        front_sorted = sorted(front, key=lambda x: objectives[x][obj_idx])
        
        # Boundary points get infinite distance
        distances[front_sorted[0]] = float('inf')
        distances[front_sorted[-1]] = float('inf')
        
        # Calculate objective range
        obj_range = objectives[front_sorted[-1]][obj_idx] - objectives[front_sorted[0]][obj_idx]
        
        if obj_range == 0:
            continue
            
        # Calculate distances for intermediate points
        for i in range(1, len(front_sorted) - 1):
            distance = (objectives[front_sorted[i + 1]][obj_idx] - 
                       objectives[front_sorted[i - 1]][obj_idx]) / obj_range
            distances[front_sorted[i]] += distance
    
    return distances


def nsga2_selection(population: List[torch.Tensor], objectives: List[Tuple[float, float]], 
                   pop_size: int) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
    """
    NSGA-II selection based on Pareto fronts and crowding distance.
    
    Args:
        population: List of individuals (tensors)
        objectives: List of objective tuples for each individual
        pop_size: Target population size
        
    Returns:
        Tuple of (selected_population, selected_objectives)
    """
    if len(population) <= pop_size:
        return population, objectives
    
    # Perform non-dominated sorting
    fronts = fast_non_dominated_sort(objectives)
    
    selected_pop = []
    selected_obj = []
    
    # Add fronts until we exceed pop_size
    for front in fronts:
        if len(selected_pop) + len(front) <= pop_size:
            # Add entire front
            for idx in front:
                selected_pop.append(population[idx])
                selected_obj.append(objectives[idx])
        else:
            # Partially add front based on crowding distance
            remaining_slots = pop_size - len(selected_pop)
            
            # Calculate crowding distances for this front
            distances = calculate_crowding_distance(objectives, front)
            
            # Sort by crowding distance (descending)
            front_sorted = sorted(front, key=lambda x: distances[x], reverse=True)
            
            # Add individuals with highest crowding distance
            for i in range(remaining_slots):
                idx = front_sorted[i]
                selected_pop.append(population[idx])
                selected_obj.append(objectives[idx])
            break
    
    return selected_pop, selected_obj


def calculate_hypervolume(objectives: List[Tuple[float, float]], 
                         reference_point: Tuple[float, float] = (0.0, 0.0)) -> float:
    """
    Calculate hypervolume indicator for a set of objectives.
    Simplified 2D implementation.
    
    Args:
        objectives: List of objective tuples
        reference_point: Reference point for hypervolume calculation
        
    Returns:
        Hypervolume value
    """
    if not objectives:
        return 0.0
    
    # Get Pareto front
    fronts = fast_non_dominated_sort(objectives)
    if not fronts or not fronts[0]:
        return 0.0
    
    pareto_front = [objectives[i] for i in fronts[0]]
    
    # Sort by first objective
    pareto_front.sort()
    
    # Calculate hypervolume using sweep line algorithm
    hv = 0.0
    prev_y = reference_point[1]
    
    for acc, f1 in pareto_front:
        if acc > reference_point[0] and f1 > reference_point[1]:
            hv += (acc - reference_point[0]) * (f1 - prev_y)
            prev_y = max(prev_y, f1)
    
    return hv


def get_pareto_front_indices(objectives: List[Tuple[float, float]]) -> List[int]:
    """
    Get indices of individuals on the Pareto front.
    
    Args:
        objectives: List of objective tuples
        
    Returns:
        List of indices of Pareto optimal individuals
    """
    fronts = fast_non_dominated_sort(objectives)
    return fronts[0] if fronts else []


def calculate_spacing_metric(objectives: List[Tuple[float, float]]) -> float:
    """
    Calculate spacing metric for solution distribution.
    Lower values indicate more uniform distribution.
    
    Args:
        objectives: List of objective tuples
        
    Returns:
        Spacing metric value
    """
    if len(objectives) < 2:
        return 0.0
    
    distances = []
    for i, obj1 in enumerate(objectives):
        min_dist = float('inf')
        for j, obj2 in enumerate(objectives):
            if i != j:
                # Euclidean distance in objective space
                dist = np.sqrt((obj1[0] - obj2[0])**2 + (obj1[1] - obj2[1])**2)
                min_dist = min(min_dist, dist)
        distances.append(min_dist)
    
    mean_dist = np.mean(distances)
    variance = np.mean([(d - mean_dist)**2 for d in distances])
    return np.sqrt(variance)


def select_diverse_solutions(population: List[torch.Tensor], 
                           objectives: List[Tuple[float, float]], 
                           num_solutions: int = 5) -> List[Tuple[torch.Tensor, Tuple[float, float]]]:
    """
    Select diverse solutions from the Pareto front for final evaluation.
    
    Args:
        population: List of individuals
        objectives: List of objective tuples
        num_solutions: Number of solutions to select
        
    Returns:
        List of (individual, objectives) tuples representing diverse solutions
    """
    # Get Pareto front
    pareto_indices = get_pareto_front_indices(objectives)
    
    if len(pareto_indices) <= num_solutions:
        return [(population[i], objectives[i]) for i in pareto_indices]
    
    # Select diverse solutions using crowding distance
    pareto_objectives = [objectives[i] for i in pareto_indices]
    distances = calculate_crowding_distance(objectives, pareto_indices)
    
    # Sort by crowding distance (descending)
    sorted_indices = sorted(pareto_indices, key=lambda x: distances[x], reverse=True)
    
    # Select top num_solutions
    selected_indices = sorted_indices[:num_solutions]
    return [(population[i], objectives[i]) for i in selected_indices]


def print_pareto_front_summary(objectives: List[Tuple[float, float]], generation: int = None):
    """
    Print summary statistics for the current Pareto front.
    
    Args:
        objectives: List of objective tuples
        generation: Current generation number (optional)
    """
    if not objectives:
        print("No objectives to summarize")
        return
    
    pareto_indices = get_pareto_front_indices(objectives)
    pareto_objectives = [objectives[i] for i in pareto_indices]
    
    if not pareto_objectives:
        print("No Pareto optimal solutions found")
        return
    
    accuracies = [obj[0] for obj in pareto_objectives]
    f1_scores = [obj[1] for obj in pareto_objectives]
    
    gen_prefix = f"Generation {generation}: " if generation is not None else ""
    
    print(f"\n{gen_prefix}Pareto Front Summary:")
    print(f"  Number of Pareto optimal solutions: {len(pareto_objectives)}")
    print(f"  Accuracy range: [{min(accuracies):.3f}, {max(accuracies):.3f}]")
    print(f"  F1 score range: [{min(f1_scores):.3f}, {max(f1_scores):.3f}]")
    print(f"  Hypervolume: {calculate_hypervolume(pareto_objectives):.6f}")
    print(f"  Spacing metric: {calculate_spacing_metric(pareto_objectives):.6f}")
    
    # Print top 3 solutions
    print("  Top solutions (by crowding distance):")
    diverse_solutions = select_diverse_solutions(
        list(range(len(pareto_objectives))), pareto_objectives, min(3, len(pareto_objectives))
    )
    for i, (_, (acc, f1)) in enumerate(diverse_solutions):
        print(f"    {i+1}. Accuracy: {acc:.3f}, F1: {f1:.3f}")
