"""
Multi-objective Differential Evolution (MODE) implementation.
Extends the existing DE algorithm to handle multiple objectives using NSGA-II principles.
"""

from random import random, sample, uniform
import numpy as np
import torch
from utils.tools.utility import *
import time
from itertools import islice
from utils.tools.spe import model_dict_to_vector, model_vector_to_dict
from utils.tools.option_de import amp_autocast
from utils.tools.multi_objective import (
    dominates, fast_non_dominated_sort, calculate_crowding_distance, 
    nsga2_selection, print_pareto_front_summary, get_pareto_front_indices,
    calculate_hypervolume
)
from spikingjelly.clock_driven import functional
from typing import List, Tuple, Dict
import os
import pickle
from . import val


def score_func_de_multi_objective(model, indi1, indi2, loader_de, args):
    """
    Multi-objective score function for DE that computes both accuracy and F1 score.
    
    Args:
        model: Neural network model
        indi1: First individual (trial vector)
        indi2: Second individual (target vector)
        loader_de: Data loader for evaluation
        args: Arguments containing configuration
        
    Returns:
        List of tuples [(trial_acc, trial_f1), (target_acc, target_f1)]
    """
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc1_all = torch.zeros(2).tolist()
    f1_all = torch.zeros(2).tolist()
    end = time.time()
    population = [indi1, indi2]
    model.eval()
    torch.set_grad_enabled(False)
    slice_len = args.de_slice_len or len(loader_de)
    
    for batch_idx, (input, target) in enumerate(islice(loader_de, slice_len)):
        data_time_m.update(time.time() - end)
        for i in range(0, 2):
            solution = population[i]
            model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
            model.load_state_dict(model_weights_dict)
            input, target = input.cuda(), target.cuda()
            input = input.contiguous(memory_format=torch.channels_last)
            with amp_autocast():
                output, _ = model(input)
            
            functional.reset_net(model)
            
            # Compute accuracy
            acc1, _ = accuracy(output, target, topk=(1, 5))
            if args.distributed:
                acc1 = reduce_tensor(acc1, args.world_size)
            acc1_all[i] += acc1
            
            # Compute F1 score
            f1 = f1_score(output, target, average='macro', num_classes=args.num_classes)
            if args.distributed:
                f1 = reduce_tensor(f1, args.world_size)
            f1_all[i] += f1
            
        batch_time_m.update(time.time() - end)
        end = time.time()

    if args.local_rank == 0:
        print('data_time: {time1.val:.3f} ({time1.avg:.3f})  '
            'batch_time: {time2.val:.3f} ({time2.avg:.3f})  '.format(time1=data_time_m, time2=batch_time_m)) 

    # Return objectives as tuples (accuracy, f1_score)
    scores = [(acc1_all[i].cpu() / slice_len, f1_all[i].cpu() / slice_len) for i in range(2)]
    return scores


def de_multi_objective(popsize, mutate, recombination, population, population_objectives, 
                      model, loader_de, args):
    """
    Multi-objective Differential Evolution using NSGA-II selection.
    
    Args:
        popsize: Population size
        mutate: Mutation factor (F)
        recombination: Crossover probability (CR)
        population: Current population (list of tensors)
        population_objectives: Current population objectives (list of tuples)
        model: Neural network model
        loader_de: Data loader for evaluation
        args: Arguments containing configuration
        
    Returns:
        Tuple of (new_population, new_objectives, update_labels, generation_stats)
    """
    update_label = [0 for _ in range(popsize)]
    device = population[0].device
    dim = len(model_dict_to_vector(model))
    
    # Store trial population and their objectives
    trial_population = []
    trial_objectives = []
    
    # Generate trial vectors for each individual
    for j in range(popsize):
        candidates = list(range(popsize))
        candidates.remove(j)
        k = min(3, len(candidates))
        random_index = sample(candidates, k)

        # DE mutation: x_new = x_r1 + F * (x_r2 - x_r3)
        x_new = population[random_index[2]] + mutate * (population[random_index[0]] - population[random_index[1]])
        
        # DE crossover: binomial crossover
        v_trial = torch.where(
            torch.rand(dim).to(device) < torch.ones(dim).to(device) * recombination, 
            x_new, 
            population[j]
        )
        
        # Evaluate trial vector against target vector
        trial_obj, target_obj = score_func_de_multi_objective(model, v_trial, population[j], loader_de, args)
        
        trial_population.append(v_trial)
        trial_objectives.append(trial_obj)

    # Combine current and trial populations
    combined_population = population + trial_population
    combined_objectives = population_objectives + trial_objectives
    
    # NSGA-II selection to get the next generation
    next_population, next_objectives = nsga2_selection(
        combined_population, combined_objectives, popsize
    )
    
    # Determine which individuals were updated by comparing solution vectors
    for j in range(popsize):
        update_label[j] = 0  # Default: not updated
        if j < len(next_population):
            # Check if the selected individual is from the trial population
            # by comparing it with the original population
            is_from_trial = False
            for k, trial_solution in enumerate(trial_population):
                if torch.allclose(next_population[j], trial_solution, rtol=1e-5, atol=1e-8):
                    is_from_trial = True
                    break
            
            if is_from_trial:
                update_label[j] = 1
    
    # Calculate generation statistics
    gen_stats = calculate_generation_stats(next_objectives)
    
    return next_population, next_objectives, update_label, gen_stats


def calculate_generation_stats(objectives: List[Tuple[float, float]]) -> Dict:
    """
    Calculate statistics for the current generation.
    
    Args:
        objectives: List of objective tuples for all individuals
        
    Returns:
        Dictionary containing generation statistics
    """
    if not objectives:
        return {}
    
    accuracies = [obj[0] for obj in objectives]
    f1_scores = [obj[1] for obj in objectives]
    
    pareto_indices = get_pareto_front_indices(objectives)
    pareto_objectives = [objectives[i] for i in pareto_indices]
    
    stats = {
        'population_size': len(objectives),
        'pareto_front_size': len(pareto_objectives),
        'best_accuracy': max(accuracies),
        'best_f1': max(f1_scores),
        'mean_accuracy': np.mean(accuracies),
        'mean_f1': np.mean(f1_scores),
        'std_accuracy': np.std(accuracies),
        'std_f1': np.std(f1_scores),
        'hypervolume': calculate_hypervolume(pareto_objectives) if pareto_objectives else 0.0,
        'pareto_front': pareto_objectives
    }
    
    return stats


def initialize_population_objectives(population, model, loader_de, args):
    """
    Initialize objectives for the population using the multi-objective score function.
    
    Args:
        population: List of individuals (tensors)
        model: Neural network model
        loader_de: Data loader for evaluation
        args: Arguments containing configuration
        
    Returns:
        List of objective tuples for each individual
    """
    # Import inside function to avoid circular import (main_cosde imports this module)
    from main_cosde import score_func_multi_objective
    
    # Use the efficient batch evaluation function
    objectives = score_func_multi_objective(model, population, loader_de, args)
    
    return objectives


def save_pareto_front(pareto_population, pareto_objectives, generation, output_dir):
    """
    Save the Pareto front to disk.
    
    Args:
        pareto_population: List of Pareto optimal individuals
        pareto_objectives: List of Pareto optimal objectives
        generation: Current generation number
        output_dir: Output directory path
    """
    pareto_data = {
        'generation': generation,
        'population': [ind.cpu().numpy() for ind in pareto_population],
        'objectives': pareto_objectives,
        'hypervolume': calculate_hypervolume(pareto_objectives)
    }
    
    # Save as pickle file
    pareto_file = os.path.join(output_dir, f'pareto_front_gen_{generation}.pkl')
    with open(pareto_file, 'wb') as f:
        pickle.dump(pareto_data, f)
    
    # Save as human-readable text
    text_file = os.path.join(output_dir, f'pareto_front_gen_{generation}.txt')
    with open(text_file, 'w') as f:
        f.write(f"Generation {generation} Pareto Front\n")
        f.write(f"Number of solutions: {len(pareto_objectives)}\n")
        f.write(f"Hypervolume: {calculate_hypervolume(pareto_objectives):.6f}\n\n")
        f.write("Objectives (Accuracy, F1 Score):\n")
        for i, (acc, f1) in enumerate(pareto_objectives):
            f.write(f"  Solution {i+1}: ({acc:.4f}, {f1:.4f})\n")


def log_multi_objective_progress(generation, gen_stats, output_dir):
    """
    Log multi-objective optimization progress.
    
    Args:
        generation: Current generation number
        gen_stats: Generation statistics dictionary
        output_dir: Output directory path
    """
    log_file = os.path.join(output_dir, 'multi_objective_log.txt')
    mode = 'a' if os.path.exists(log_file) else 'w'
    
    with open(log_file, mode) as f:
        if mode == 'w':
            f.write("Generation,Pareto_Front_Size,Best_Accuracy,Best_F1,Mean_Accuracy,Mean_F1,Hypervolume\n")
        
        f.write(f"{generation},{gen_stats['pareto_front_size']},{gen_stats['best_accuracy']:.4f},"
                f"{gen_stats['best_f1']:.4f},{gen_stats['mean_accuracy']:.4f},"
                f"{gen_stats['mean_f1']:.4f},{gen_stats['hypervolume']:.6f}\n")


def evaluate_validation_objectives(population, model, loader_eval, args, amp_autocast):
    """
    Evaluate validation objectives for the population.
    
    Args:
        population: List of individuals (tensors)
        model: Neural network model
        loader_eval: Validation data loader
        args: Arguments containing configuration
        amp_autocast: Automatic mixed precision context manager
        
    Returns:
        List of objective tuples for each individual (validation accuracy, validation F1)
    """
    objectives = []
    
    for individual in population:
        # Apply individual to model
        model_weights_dict = model_vector_to_dict(model=model, weights_vector=individual)
        model.load_state_dict(model_weights_dict)
        
        # Validate on validation data
        with torch.no_grad():
            val_metrics = val.validate(model, loader_eval, args, amp_autocast=amp_autocast)
            
        # Extract accuracy and F1 score
        accuracy = val_metrics.get('top1', 0.0)
        f1_score = val_metrics.get('f1', 0.0)
        
        objectives.append((accuracy, f1_score))
    
    return objectives


def apply_individual_to_model(model, individual):
    """
    Apply an individual (parameter vector) to the model.
    
    Args:
        model: Neural network model
        individual: Parameter tensor to apply to model
    """
    model_weights_dict = model_vector_to_dict(model=model, weights_vector=individual)
    model.load_state_dict(model_weights_dict)
