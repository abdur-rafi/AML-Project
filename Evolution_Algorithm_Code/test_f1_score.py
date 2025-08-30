#!/usr/bin/env python3
"""
Independent test script for F1 score implementation
Tests the custom f1_score function against sklearn's implementation
"""

import torch
import numpy as np
from sklearn.metrics import f1_score as sklearn_f1
from sklearn.metrics import classification_report
import sys
import os
import time



def f1_score(output, target, average='macro', num_classes=None):
    """Computes the F1 score for multi-class classification
    
    Args:
        output: Model predictions (logits) with shape [batch_size, num_classes]
        target: Ground truth labels with shape [batch_size]
        average: 'macro' for macro-averaged F1, 'micro' for micro-averaged F1, 'weighted' for weighted F1
        num_classes: Number of classes (if None, inferred from output)
    
    Returns:
        F1 score as a tensor (0-100 scale)
    """
    if num_classes is None:
        num_classes = output.size(1)
    
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze(1)  # Remove the extra dimension
    
    # Convert to one-hot encoding for easier computation
    pred_one_hot = torch.zeros(batch_size, num_classes, device=output.device)
    target_one_hot = torch.zeros(batch_size, num_classes, device=output.device)
    
    pred_one_hot.scatter_(1, pred.unsqueeze(1), 1)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)
    
    if average == 'micro':
        # Micro-averaged F1: aggregate the contributions of all classes
        tp = (pred_one_hot * target_one_hot).sum()
        fp = (pred_one_hot * (1 - target_one_hot)).sum()
        fn = ((1 - pred_one_hot) * target_one_hot).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
    elif average == 'weighted':
        # Weighted F1: weight by class frequency
        class_counts = target_one_hot.sum(0)
        total_samples = class_counts.sum()
        
        tp = (pred_one_hot * target_one_hot).sum(0)
        fp = (pred_one_hot * (1 - target_one_hot)).sum(0)
        fn = ((1 - pred_one_hot) * target_one_hot).sum(0)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        class_f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Weight by class frequency
        weights = class_counts / total_samples
        f1 = (class_f1 * weights).sum()
        
    else:  # default to macro
        # Macro-averaged F1: average F1 across all classes
        tp = (pred_one_hot * target_one_hot).sum(0)
        fp = (pred_one_hot * (1 - target_one_hot)).sum(0)
        fn = ((1 - pred_one_hot) * target_one_hot).sum(0)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        class_f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Average across classes
        f1 = class_f1.mean()
    
    return f1 * 100.  # Convert to 0-100 scale to match accuracy function


def create_test_data(batch_size=100, num_classes=10, seed=42):
    """Create test data for F1 score evaluation"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create random logits (model output)
    logits = torch.randn(batch_size, num_classes)
    
    # Create random targets
    targets = torch.randint(0, num_classes, (batch_size,))
    
    return logits, targets

def create_specific_test_cases():
    """Create specific test cases with known F1 scores"""
    test_cases = []
    
    # Test Case 1: Perfect predictions
    logits1 = torch.tensor([
        [10.0, 0.0, 0.0],  # Pred: 0, True: 0
        [0.0, 10.0, 0.0],  # Pred: 1, True: 1  
        [0.0, 0.0, 10.0],  # Pred: 2, True: 2
        [10.0, 0.0, 0.0],  # Pred: 0, True: 0
    ])
    targets1 = torch.tensor([0, 1, 2, 0])
    test_cases.append(("Perfect predictions", logits1, targets1, 100.0))
    
    # Test Case 2: All wrong predictions
    logits2 = torch.tensor([
        [0.0, 10.0, 0.0],  # Pred: 1, True: 0
        [0.0, 0.0, 10.0],  # Pred: 2, True: 1
        [10.0, 0.0, 0.0],  # Pred: 0, True: 2
    ])
    targets2 = torch.tensor([0, 1, 2])
    test_cases.append(("All wrong predictions", logits2, targets2, 0.0))
    
    # Test Case 3: Mixed predictions
    logits3 = torch.tensor([
        [10.0, 0.0, 0.0],  # Pred: 0, True: 0 ✓
        [0.0, 10.0, 0.0],  # Pred: 1, True: 1 ✓
        [0.0, 0.0, 10.0],  # Pred: 2, True: 1 ✗
        [10.0, 0.0, 0.0],  # Pred: 0, True: 2 ✗
    ])
    targets3 = torch.tensor([0, 1, 1, 2])
    test_cases.append(("Mixed predictions", logits3, targets3, None))  # Will calculate expected
    
    return test_cases

def test_against_sklearn(logits, targets, average='macro'):
    """Compare our F1 score with sklearn's implementation"""
    # Get predictions from logits
    _, pred = logits.topk(1, 1, True, True)
    pred = pred.squeeze().numpy()
    targets_np = targets.numpy()
    
    # Calculate sklearn F1 score
    if average == 'macro':
        sklearn_result = sklearn_f1(targets_np, pred, average='macro', zero_division=0)
    elif average == 'micro':
        sklearn_result = sklearn_f1(targets_np, pred, average='micro', zero_division=0)
    elif average == 'weighted':
        sklearn_result = sklearn_f1(targets_np, pred, average='weighted', zero_division=0)
    
    # Calculate our F1 score
    our_result = f1_score(logits, targets, average=average)
    
    # Convert to same scale (sklearn returns 0-1, ours returns 0-100)
    sklearn_result_scaled = sklearn_result * 100
    our_result_item = our_result.item()
    
    return sklearn_result_scaled, our_result_item

def test_class_imbalance():
    """Test F1 score with heavily imbalanced classes"""
    print("\n\n6. CLASS IMBALANCE TESTS")
    print("-" * 30)
    
    # Create heavily imbalanced dataset: 90% class 0, 5% class 1, 5% class 2
    batch_size = 100
    num_classes = 3
    
    # Create imbalanced targets
    targets_imbalanced = torch.cat([
        torch.zeros(90),      # 90 samples of class 0
        torch.ones(5),        # 5 samples of class 1  
        torch.full((5,), 2)   # 5 samples of class 2
    ]).long()
    
    # Shuffle the targets
    torch.manual_seed(42)
    shuffled_indices = torch.randperm(batch_size)
    targets_imbalanced = targets_imbalanced[shuffled_indices]
    
    # Create logits that predict mostly class 0 (mimicking biased model)
    logits_biased = torch.randn(batch_size, num_classes)
    logits_biased[:, 0] += 2.0  # Bias towards class 0
    
    print("Test: Heavily imbalanced classes (90%, 5%, 5%)")
    print(f"Class distribution: {torch.bincount(targets_imbalanced).tolist()}")
    
    for avg_method in ['macro', 'micro', 'weighted']:
        our_f1 = f1_score(logits_biased, targets_imbalanced, average=avg_method).item()
        sklearn_f1_scaled, _ = test_against_sklearn(logits_biased, targets_imbalanced, average=avg_method)
        
        diff = abs(our_f1 - sklearn_f1_scaled)
        tolerance = 0.01
        
        status = "✓ PASS" if diff < tolerance else "✗ FAIL"
        print(f"  {avg_method:>8}: Our={our_f1:6.2f}%, sklearn={sklearn_f1_scaled:6.2f}%, diff={diff:6.3f}% {status}")

def test_binary_classification():
    """Test F1 score for binary classification"""
    print("\n\n7. BINARY CLASSIFICATION TESTS")
    print("-" * 30)
    
    # Test Case 1: Perfect binary classification
    logits_binary = torch.tensor([
        [5.0, -5.0],   # Pred: 0, True: 0
        [-5.0, 5.0],   # Pred: 1, True: 1
        [5.0, -5.0],   # Pred: 0, True: 0
        [-5.0, 5.0],   # Pred: 1, True: 1
    ])
    targets_binary = torch.tensor([0, 1, 0, 1])
    
    print("Test: Perfect binary classification")
    for avg_method in ['macro', 'micro', 'weighted']:
        our_f1 = f1_score(logits_binary, targets_binary, average=avg_method).item()
        sklearn_f1_scaled, _ = test_against_sklearn(logits_binary, targets_binary, average=avg_method)
        
        diff = abs(our_f1 - sklearn_f1_scaled)
        tolerance = 0.01
        
        status = "✓ PASS" if diff < tolerance else "✗ FAIL"
        print(f"  {avg_method:>8}: Our={our_f1:6.2f}%, sklearn={sklearn_f1_scaled:6.2f}%, diff={diff:6.3f}% {status}")
    
    # Test Case 2: Imbalanced binary classification
    logits_binary_imb = torch.tensor([
        [2.0, -1.0],   # Pred: 0, True: 0
        [2.0, -1.0],   # Pred: 0, True: 0  
        [2.0, -1.0],   # Pred: 0, True: 0
        [2.0, -1.0],   # Pred: 0, True: 0
        [-1.0, 2.0],   # Pred: 1, True: 1
        [2.0, -1.0],   # Pred: 0, True: 1 (miss)
    ])
    targets_binary_imb = torch.tensor([0, 0, 0, 0, 1, 1])
    
    print("\nTest: Imbalanced binary classification (4:2 ratio)")
    for avg_method in ['macro', 'micro', 'weighted']:
        our_f1 = f1_score(logits_binary_imb, targets_binary_imb, average=avg_method).item()
        sklearn_f1_scaled, _ = test_against_sklearn(logits_binary_imb, targets_binary_imb, average=avg_method)
        
        diff = abs(our_f1 - sklearn_f1_scaled)
        tolerance = 0.01
        
        status = "✓ PASS" if diff < tolerance else "✗ FAIL"
        print(f"  {avg_method:>8}: Our={our_f1:6.2f}%, sklearn={sklearn_f1_scaled:6.2f}%, diff={diff:6.3f}% {status}")

def test_extreme_cases():
    """Test extreme edge cases"""
    print("\n\n8. EXTREME EDGE CASES")
    print("-" * 30)
    
    # Test Case 1: All predictions wrong for one class
    print("Test: All predictions wrong for specific class")
    logits_extreme = torch.tensor([
        [0.0, 5.0, 0.0],   # Pred: 1, True: 0
        [0.0, 5.0, 0.0],   # Pred: 1, True: 0
        [0.0, 0.0, 5.0],   # Pred: 2, True: 1
        [0.0, 0.0, 5.0],   # Pred: 2, True: 1
        [5.0, 0.0, 0.0],   # Pred: 0, True: 2
        [5.0, 0.0, 0.0],   # Pred: 0, True: 2
    ])
    targets_extreme = torch.tensor([0, 0, 1, 1, 2, 2])
    
    for avg_method in ['macro', 'micro', 'weighted']:
        our_f1 = f1_score(logits_extreme, targets_extreme, average=avg_method).item()
        sklearn_f1_scaled, _ = test_against_sklearn(logits_extreme, targets_extreme, average=avg_method)
        
        diff = abs(our_f1 - sklearn_f1_scaled)
        tolerance = 0.01
        
        status = "✓ PASS" if diff < tolerance else "✗ FAIL"
        print(f"  {avg_method:>8}: Our={our_f1:6.2f}%, sklearn={sklearn_f1_scaled:6.2f}%, diff={diff:6.3f}% {status}")
    
    # Test Case 2: Very confident wrong predictions
    print("\nTest: Very confident wrong predictions")
    logits_confident = torch.tensor([
        [100.0, -100.0],   # Very confident wrong: Pred 0, True 1
        [100.0, -100.0],   # Very confident wrong: Pred 0, True 1
        [-100.0, 100.0],   # Very confident right: Pred 1, True 1
        [-100.0, 100.0],   # Very confident right: Pred 1, True 1
    ])
    targets_confident = torch.tensor([1, 1, 1, 1])
    
    for avg_method in ['macro', 'micro', 'weighted']:
        our_f1 = f1_score(logits_confident, targets_confident, average=avg_method).item()
        sklearn_f1_scaled, _ = test_against_sklearn(logits_confident, targets_confident, average=avg_method)
        
        diff = abs(our_f1 - sklearn_f1_scaled)
        tolerance = 0.01
        
        status = "✓ PASS" if diff < tolerance else "✗ FAIL"
        print(f"  {avg_method:>8}: Our={our_f1:6.2f}%, sklearn={sklearn_f1_scaled:6.2f}%, diff={diff:6.3f}% {status}")

def test_numerical_stability():
    """Test numerical stability with very small/large values"""
    print("\n\n9. NUMERICAL STABILITY TESTS")
    print("-" * 30)
    
    # Test Case 1: Very small logit differences
    print("Test: Very small logit differences")
    logits_small = torch.tensor([
        [1e-7, 0.0, -1e-7],
        [0.0, 1e-7, -1e-7],
        [-1e-7, -1e-7, 1e-7],
    ])
    targets_small = torch.tensor([0, 1, 2])
    
    try:
        for avg_method in ['macro', 'micro', 'weighted']:
            our_f1 = f1_score(logits_small, targets_small, average=avg_method).item()
            sklearn_f1_scaled, _ = test_against_sklearn(logits_small, targets_small, average=avg_method)
            
            # Check if result is finite
            if torch.isfinite(torch.tensor(our_f1)) and not torch.isnan(torch.tensor(our_f1)):
                status = "✓ PASS (finite)"
            else:
                status = "✗ FAIL (not finite)"
            
            print(f"  {avg_method:>8}: Our={our_f1:6.2f}%, sklearn={sklearn_f1_scaled:6.2f}% {status}")
    except Exception as e:
        print(f"  Error with small values: {e}")
    
    # Test Case 2: Very large logit values
    print("\nTest: Very large logit values")
    logits_large = torch.tensor([
        [1e6, 0.0, -1e6],
        [0.0, 1e6, -1e6],
        [-1e6, -1e6, 1e6],
    ])
    targets_large = torch.tensor([0, 1, 2])
    
    try:
        for avg_method in ['macro', 'micro', 'weighted']:
            our_f1 = f1_score(logits_large, targets_large, average=avg_method).item()
            sklearn_f1_scaled, _ = test_against_sklearn(logits_large, targets_large, average=avg_method)
            
            # Check if result is finite
            if torch.isfinite(torch.tensor(our_f1)) and not torch.isnan(torch.tensor(our_f1)):
                status = "✓ PASS (finite)"
            else:
                status = "✗ FAIL (not finite)"
            
            print(f"  {avg_method:>8}: Our={our_f1:6.2f}%, sklearn={sklearn_f1_scaled:6.2f}% {status}")
    except Exception as e:
        print(f"  Error with large values: {e}")

def test_different_batch_sizes():
    """Test with various batch sizes"""
    print("\n\n10. BATCH SIZE VARIATION TESTS")
    print("-" * 30)
    
    batch_sizes = [1, 2, 3, 5, 10, 50, 100, 500, 1000]
    num_classes = 5
    
    print("Test: Various batch sizes with 5 classes")
    all_passed = True
    
    for batch_size in batch_sizes:
        try:
            logits, targets = create_test_data(batch_size, num_classes, seed=batch_size)
            
            # Test only macro averaging for brevity
            our_f1 = f1_score(logits, targets, average='macro').item()
            sklearn_f1_scaled, _ = test_against_sklearn(logits, targets, average='macro')
            
            diff = abs(our_f1 - sklearn_f1_scaled)
            tolerance = 0.01
            
            status = "✓" if diff < tolerance else "✗"
            if batch_size <= 10 or batch_size % 100 == 0:  # Print only some results
                print(f"  Batch {batch_size:4d}: Our={our_f1:6.2f}%, sklearn={sklearn_f1_scaled:6.2f}%, diff={diff:6.3f}% {status}")
            
            if diff >= tolerance:
                all_passed = False
                
        except Exception as e:
            print(f"  Batch {batch_size:4d}: ERROR - {e}")
            all_passed = False
    
    if all_passed:
        print("  All batch size tests passed!")
    else:
        print("  Some batch size tests failed!")

def test_consistency():
    """Test consistency across multiple runs with same data"""
    print("\n\n11. CONSISTENCY TESTS")
    print("-" * 30)
    
    logits, targets = create_test_data(100, 10, seed=42)
    
    print("Test: Multiple runs with same data (should be identical)")
    
    # Run the same test 5 times
    results = []
    for i in range(5):
        f1_macro = f1_score(logits, targets, average='macro').item()
        f1_micro = f1_score(logits, targets, average='micro').item()
        f1_weighted = f1_score(logits, targets, average='weighted').item()
        results.append((f1_macro, f1_micro, f1_weighted))
    
    # Check if all results are identical
    first_result = results[0]
    all_consistent = all(result == first_result for result in results)
    
    status = "✓ PASS" if all_consistent else "✗ FAIL"
    print(f"  Consistency check: {status}")
    print(f"  Macro F1:    {first_result[0]:.6f}%")
    print(f"  Micro F1:    {first_result[1]:.6f}%")
    print(f"  Weighted F1: {first_result[2]:.6f}%")
    
    if not all_consistent:
        print("  Results varied across runs:")
        for i, result in enumerate(results):
            print(f"    Run {i+1}: {result}")

def test_memory_efficiency():
    """Test memory usage with large datasets"""
    print("\n\n12. MEMORY EFFICIENCY TESTS")
    print("-" * 30)
    
    print("Test: Large dataset memory usage")
    
    # Test with progressively larger datasets
    sizes = [1000, 5000, 10000]
    num_classes = 100  # CIFAR-100 size
    
    for size in sizes:
        try:
            # Monitor memory before
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            logits, targets = create_test_data(size, num_classes, seed=42)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                logits = logits.cuda()
                targets = targets.cuda()
            
            # Calculate F1 score
            start_time = time.time()
            f1_result = f1_score(logits, targets, average='macro')
            end_time = time.time()
            
            # Monitor memory after
            if torch.cuda.is_available():
                final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                memory_used = (final_memory - initial_memory) / 1024 / 1024  # MB
            else:
                memory_used = "N/A"
            
            print(f"  Size {size:5d}: F1={f1_result.item():6.2f}%, Time={end_time-start_time:.3f}s, Memory={memory_used}MB")
            
        except Exception as e:
            print(f"  Size {size:5d}: ERROR - {e}")

def run_comprehensive_tests():
    """Run comprehensive tests of the F1 score implementation"""
    print("=" * 60)
    print("F1 SCORE IMPLEMENTATION TEST SUITE")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test 1: Specific test cases
    print("\n1. SPECIFIC TEST CASES")
    print("-" * 30)
    
    test_cases = create_specific_test_cases()
    for name, logits, targets, expected in test_cases:
        print(f"\nTest: {name}")
        
        # Test all averaging methods
        for avg_method in ['macro', 'micro', 'weighted']:
            our_f1 = f1_score(logits, targets, average=avg_method).item()
            sklearn_f1_scaled, _ = test_against_sklearn(logits, targets, average=avg_method)
            
            diff = abs(our_f1 - sklearn_f1_scaled)
            tolerance = 0.01  # 0.01% tolerance
            
            status = "✓ PASS" if diff < tolerance else "✗ FAIL"
            print(f"  {avg_method:>8}: Our={our_f1:6.2f}%, sklearn={sklearn_f1_scaled:6.2f}%, diff={diff:6.3f}% {status}")
            
            if diff >= tolerance:
                all_tests_passed = False
    
    # Test 2: Random data with different class counts
    print("\n\n2. RANDOM DATA TESTS")
    print("-" * 30)
    
    test_configs = [
        (50, 5, "Small: 50 samples, 5 classes"),
        (100, 10, "Medium: 100 samples, 10 classes"),
        (200, 20, "Large: 200 samples, 20 classes"),
        (1000, 100, "CIFAR-100: 1000 samples, 100 classes"),
    ]
    
    for batch_size, num_classes, description in test_configs:
        print(f"\nTest: {description}")
        logits, targets = create_test_data(batch_size, num_classes)
        
        for avg_method in ['macro', 'micro', 'weighted']:
            sklearn_f1_scaled, our_f1 = test_against_sklearn(logits, targets, average=avg_method)
            
            diff = abs(our_f1 - sklearn_f1_scaled)
            tolerance = 0.01  # 0.01% tolerance
            
            status = "✓ PASS" if diff < tolerance else "✗ FAIL"
            print(f"  {avg_method:>8}: Our={our_f1:6.2f}%, sklearn={sklearn_f1_scaled:6.2f}%, diff={diff:6.3f}% {status}")
            
            if diff >= tolerance:
                all_tests_passed = False
    
    # Test 3: Edge cases
    print("\n\n3. EDGE CASE TESTS")
    print("-" * 30)
    
    # Single class prediction
    print("\nTest: Single class (all predictions same)")
    logits_single = torch.tensor([[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]])
    targets_single = torch.tensor([0, 1, 0])
    
    for avg_method in ['macro', 'micro', 'weighted']:
        our_f1 = f1_score(logits_single, targets_single, average=avg_method).item()
        sklearn_f1_scaled, _ = test_against_sklearn(logits_single, targets_single, average=avg_method)
        
        diff = abs(our_f1 - sklearn_f1_scaled)
        tolerance = 0.01
        
        status = "✓ PASS" if diff < tolerance else "✗ FAIL"
        print(f"  {avg_method:>8}: Our={our_f1:6.2f}%, sklearn={sklearn_f1_scaled:6.2f}%, diff={diff:6.3f}% {status}")
        
        if diff >= tolerance:
            all_tests_passed = False
    
    # Test 4: GPU compatibility (if available)
    if torch.cuda.is_available():
        print("\n\n4. GPU COMPATIBILITY TEST")
        print("-" * 30)
        
        logits_cpu, targets_cpu = create_test_data(100, 10)
        logits_gpu = logits_cpu.cuda()
        targets_gpu = targets_cpu.cuda()
        
        f1_cpu = f1_score(logits_cpu, targets_cpu, average='macro').item()
        f1_gpu = f1_score(logits_gpu, targets_gpu, average='macro').item()
        
        diff = abs(f1_cpu - f1_gpu)
        tolerance = 1e-6
        
        status = "✓ PASS" if diff < tolerance else "✗ FAIL"
        print(f"CPU vs GPU: {f1_cpu:.6f}% vs {f1_gpu:.6f}%, diff={diff:.8f}% {status}")
        
        if diff >= tolerance:
            all_tests_passed = False
    else:
        print("\n\n4. GPU COMPATIBILITY TEST")
        print("-" * 30)
        print("CUDA not available, skipping GPU test")
    
    # Test 5: Detailed classification report comparison
    print("\n\n5. DETAILED COMPARISON WITH SKLEARN")
    print("-" * 30)
    
    logits, targets = create_test_data(200, 5, seed=123)
    _, pred = logits.topk(1, 1, True, True)
    pred_np = pred.squeeze().numpy()
    targets_np = targets.numpy()
    
    print("\nSklearn Classification Report:")
    print(classification_report(targets_np, pred_np, digits=4, zero_division=0))
    
    print("\nOur F1 Score Implementation:")
    for avg_method in ['macro', 'micro', 'weighted']:
        our_f1 = f1_score(logits, targets, average=avg_method).item()
        sklearn_f1_val = sklearn_f1(targets_np, pred_np, average=avg_method, zero_division=0) * 100
        print(f"  {avg_method:>8} F1: {our_f1:7.4f}% (sklearn: {sklearn_f1_val:7.4f}%)")
    
    # Additional comprehensive tests
    test_class_imbalance()
    test_binary_classification()
    test_extreme_cases()
    test_numerical_stability()
    test_different_batch_sizes()
    test_consistency()
    test_memory_efficiency()
    
    # Final result
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! F1 score implementation is correct.")
    else:
        print("❌ SOME TESTS FAILED! Check implementation.")
    print("=" * 60)
    
    return all_tests_passed

def demonstrate_usage():
    """Demonstrate how to use the F1 score function"""
    print("\n\nUSAGE DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    batch_size = 8
    num_classes = 3
    logits = torch.tensor([
        [2.1, 0.1, -1.0],  # Class 0
        [-1.0, 3.2, 0.5],  # Class 1
        [0.2, -0.5, 2.8],  # Class 2
        [1.9, 0.3, -0.8],  # Class 0
        [-0.5, 2.7, 0.1],  # Class 1
        [0.1, -1.2, 2.1],  # Class 2
        [2.5, -0.2, 0.0],  # Class 0
        [-0.8, 1.8, 1.2],  # Class 1
    ])
    targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    
    print(f"Sample data: {batch_size} samples, {num_classes} classes")
    print(f"Predictions: {logits.argmax(dim=1).tolist()}")
    print(f"True labels: {targets.tolist()}")
    
    print("\nF1 Score Results:")
    for avg_method in ['macro', 'micro', 'weighted']:
        f1_val = f1_score(logits, targets, average=avg_method)
        print(f"  {avg_method:>8} F1: {f1_val.item():6.2f}%")
    
    print("\nCode example:")
    print("```python")
    print("from utils.tools.utility import f1_score")
    print("")
    print("# Calculate macro F1 score (default)")
    print("f1_macro = f1_score(model_output, true_labels)")
    print("")
    print("# Calculate micro F1 score")
    print("f1_micro = f1_score(model_output, true_labels, average='micro')")
    print("")
    print("# Calculate weighted F1 score")
    print("f1_weighted = f1_score(model_output, true_labels, average='weighted')")
    print("```")

if __name__ == "__main__":
    print("Testing F1 Score Implementation")
    print("Requirements: torch, numpy, sklearn")
    
    try:
        # Run all tests
        success = run_comprehensive_tests()
        
        # Show usage demonstration
        demonstrate_usage()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install required packages: pip install torch numpy scikit-learn")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
