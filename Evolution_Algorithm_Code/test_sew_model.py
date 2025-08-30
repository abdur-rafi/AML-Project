#!/usr/bin/env python3
"""
Test script for SEW ResNet34 model on CIFAR-100.
Creates a SEW ResNet34 model, loads parameters from a state dictionary,
handles CIFAR-100 dataset download/loading, and evaluates test accuracy.
"""

import os
import argparse
import torch
import torch.nn as nn
from contextlib import suppress
from spikingjelly.clock_driven import functional

# Import necessary modules from the existing codebase
from utils.snn_model import SEW
from utils.data.cifar_loader import create_loader_cifar
from utils.tools.utility import AverageMeter, accuracy


def validate_checkpoint_file(file_path):
    """
    Validate if a file is a valid PyTorch checkpoint.
    
    Args:
        file_path (str): Path to the checkpoint file
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return False, f"File is empty: {file_path}"
    
    if file_size < 1024:  # Less than 1KB is suspicious for a model checkpoint
        return False, f"File is too small ({file_size} bytes) to be a valid checkpoint"
    
    # Check file extension
    valid_extensions = ['.pt', '.pth', '.tar', '.ckpt']
    if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
        print(f"Warning: File doesn't have a standard PyTorch extension: {file_path}")
    
    # Try to read first few bytes to check if it's a valid zip/tar file
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            # PyTorch files are typically ZIP files, so check for ZIP signature
            if header[:2] == b'PK':  # ZIP file signature
                return True, "Valid ZIP-based checkpoint file"
            elif header == b'}\x93\x8b\x00':  # Some PyTorch files start with this
                return True, "Valid PyTorch checkpoint file"
            else:
                return False, f"File doesn't appear to be a valid PyTorch checkpoint (header: {header})"
    except Exception as e:
        return False, f"Cannot read file: {str(e)}"


def try_alternative_loading_methods(file_path):
    """
    Try alternative methods to load a potentially corrupted checkpoint.
    
    Args:
        file_path (str): Path to the checkpoint file
        
    Returns:
        dict or None: Loaded checkpoint if successful, None otherwise
    """
    print("Trying alternative loading methods...")
    
    # Method 1: Try loading with weights_only=True (PyTorch 1.13+)
    try:
        print("Trying weights_only=True...")
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
        print("Success with weights_only=True")
        return checkpoint
    except Exception as e:
        print(f"Failed with weights_only=True: {str(e)}")
    
    # Method 2: Try loading with pickle protocol
    try:
        print("Trying with pickle_module...")
        import pickle
        checkpoint = torch.load(file_path, map_location='cpu', pickle_module=pickle)
        print("Success with pickle_module")
        return checkpoint
    except Exception as e:
        print(f"Failed with pickle_module: {str(e)}")
    
    # Method 3: Try loading as a raw file and inspect
    try:
        print("Inspecting file content...")
        with open(file_path, 'rb') as f:
            content = f.read(1000)  # Read first 1000 bytes
            print(f"File starts with: {content[:50]}")
            
            # Check if it's a text file (sometimes state dicts are saved as text)
            if content.isascii():
                print("File appears to be ASCII text, not a binary PyTorch file")
                return None
    except Exception as e:
        print(f"Cannot inspect file: {str(e)}")
    
    return None


def create_sew_model(num_classes=100, T=4):
    """
    Create a SEW ResNet34 model.
    
    Args:
        num_classes (int): Number of output classes (default: 100 for CIFAR-100)
        T (int): Number of time steps (default: 4)
    
    Returns:
        torch.nn.Module: SEW ResNet34 model
    """
    model = SEW.resnet34(num_classes=num_classes, g="add", down='max', T=T)
    return model


def load_model_from_state_dict(model, state_dict_path):
    """
    Load model parameters from a state dictionary file.
    
    Args:
        model (torch.nn.Module): Model to load parameters into
        state_dict_path (str): Path to the state dictionary file
    
    Returns:
        torch.nn.Module: Model with loaded parameters
    """
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"State dictionary file not found: {state_dict_path}")
    
    # Check file size and basic validity
    file_size = os.path.getsize(state_dict_path)
    if file_size == 0:
        raise ValueError(f"State dictionary file is empty: {state_dict_path}")
    
    print(f"Loading model parameters from: {state_dict_path}")
    print(f"File size: {file_size / (1024*1024):.2f} MB")
    
    try:
        # Try to load the state dictionary with better error handling
        checkpoint = torch.load(state_dict_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint file: {str(e)}")
        print(f"File size: {file_size} bytes")
        
        # Check if file is readable
        try:
            with open(state_dict_path, 'rb') as f:
                first_bytes = f.read(100)
                print(f"First 100 bytes: {first_bytes}")
        except Exception as read_error:
            print(f"Cannot read file: {read_error}")
        
        # Try alternative loading methods
        checkpoint = try_alternative_loading_methods(state_dict_path)
        
        if checkpoint is None:
            # Provide helpful suggestions
            print("\nPossible solutions:")
            print("1. Check if the file was completely downloaded/transferred")
            print("2. Verify the file is a valid PyTorch checkpoint (.pt, .pth, .tar)")
            print("3. Try re-downloading or re-copying the file")
            print("4. Check if the file path is correct")
            print("5. Ensure the file is not corrupted")
            
            raise ValueError(f"Failed to load checkpoint from {state_dict_path}: {str(e)}")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("Found 'state_dict' key in checkpoint")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("Found 'model' key in checkpoint")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Found 'model_state_dict' key in checkpoint")
        else:
            # Assume the entire dict is the state dict
            state_dict = checkpoint
            print("Using entire checkpoint as state dict")
            
        # Print additional checkpoint info if available
        if 'epoch' in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
        if 'best_acc1' in checkpoint:
            print(f"Checkpoint best accuracy: {checkpoint['best_acc1']:.4f}%")
        if 'arch' in checkpoint:
            print(f"Checkpoint architecture: {checkpoint['arch']}")
    else:
        state_dict = checkpoint
        print("Checkpoint is direct state dict")
    
    # Validate state dict
    if not isinstance(state_dict, dict):
        raise ValueError(f"Expected state_dict to be a dictionary, got {type(state_dict)}")
    
    if len(state_dict) == 0:
        raise ValueError("State dictionary is empty")
    
    print(f"State dict contains {len(state_dict)} parameter tensors")
    
    # Try to load the state dictionary into the model
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys in state dict: {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            print(f"Warning: Unexpected keys in state dict: {unexpected_keys[:5]}...")  # Show first 5
            
        print("Model parameters loaded successfully!")
        
    except Exception as e:
        print(f"Error loading state dict into model: {str(e)}")
        print("This might be due to model architecture mismatch")
        raise ValueError(f"Failed to load state dict into model: {str(e)}")
    
    return model


def setup_cifar100_loader(data_path, batch_size=128, num_workers=4):
    """
    Setup CIFAR-100 data loader with automatic download if needed.
    
    Args:
        data_path (str): Path to the data directory
        batch_size (int): Batch size for the data loader
        num_workers (int): Number of worker processes for data loading
    
    Returns:
        torch.utils.data.DataLoader: Test data loader for CIFAR-100
    """
    print(f"Setting up CIFAR-100 dataset at: {data_path}")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)
    
    # Create a simple args object for the loader function
    class Args:
        def __init__(self):
            self.dataset = 'cifar100'
            self.data_dir = data_path
            self.batch_size = batch_size
            self.validation_batch_size = batch_size
            self.workers = num_workers
            self.distributed = False
            self.pin_mem = True
            self.input_size = 32
    
    args = Args()
    
    # Create the data loaders using the existing function
    _, test_loader, _ = create_loader_cifar(args)
    
    print(f"CIFAR-100 test loader created with {len(test_loader)} batches")
    return test_loader


def evaluate_test_accuracy(model, test_loader, device='cuda'):
    """
    Evaluate the test accuracy of the model on CIFAR-100.
    
    Args:
        model (torch.nn.Module): Model to evaluate
        test_loader (torch.utils.data.DataLoader): Test data loader
        device (str): Device to run evaluation on ('cuda' or 'cpu')
    
    Returns:
        dict: Dictionary containing accuracy metrics
    """
    print("Evaluating test accuracy...")
    
    # Set device
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        model = model.to(memory_format=torch.channels_last)
        print("Using CUDA for evaluation")
    else:
        device = 'cpu'
        print("Using CPU for evaluation")
    
    model.eval()
    
    # Metrics tracking
    batch_time_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    
    total_samples = 0
    correct_top1 = 0
    correct_top5 = 0
    
    # Use mixed precision if available
    amp_autocast = suppress
    if device == 'cuda':
        try:
            from torch.cuda.amp import autocast
            amp_autocast = autocast
            print("Using mixed precision (AMP)")
        except ImportError:
            print("Mixed precision not available, using regular precision")
            amp_autocast = suppress
    
    print(f"Starting evaluation on {len(test_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(test_loader):
            # Move data to device
            if device == 'cuda':
                input, target = input.cuda(), target.cuda()
                input = input.contiguous(memory_format=torch.channels_last)
            
            batch_size = input.size(0)
            total_samples += batch_size
            
            # Forward pass with mixed precision
            with amp_autocast():
                output, _ = model(input)
            
            # Reset the spiking neural network state
            functional.reset_net(model)
            
            # Calculate accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            # Update metrics
            top1_m.update(acc1.item(), batch_size)
            top5_m.update(acc5.item(), batch_size)
            
            # Track total correct predictions
            correct_top1 += acc1.item() * batch_size / 100
            correct_top5 += acc5.item() * batch_size / 100
            
            # Print progress every 20 batches
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(test_loader):
                print(f"Batch [{batch_idx + 1}/{len(test_loader)}] - "
                      f"Top1 Acc: {top1_m.avg:.2f}% - Top5 Acc: {top5_m.avg:.2f}%")
    
    # Calculate final metrics
    final_top1_acc = (correct_top1 / total_samples) * 100
    final_top5_acc = (correct_top5 / total_samples) * 100
    
    results = {
        'top1_accuracy': final_top1_acc,
        'top5_accuracy': final_top5_acc,
        'total_samples': total_samples,
        'avg_top1': top1_m.avg,
        'avg_top5': top5_m.avg
    }
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total test samples: {total_samples}")
    print(f"Top-1 Accuracy: {final_top1_acc:.4f}%")
    print(f"Top-5 Accuracy: {final_top5_acc:.4f}%")
    print("="*60)
    
    return results


def main():
    """Main function to run the test evaluation."""
    parser = argparse.ArgumentParser(description='Test SEW ResNet34 on CIFAR-100')
    parser.add_argument('--state_dict_path', type=str, required=True,
                        help='Path to the model state dictionary file')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to the CIFAR-100 data directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use for evaluation')
    parser.add_argument('--num_classes', type=int, default=100,
                        help='Number of classes (default: 100 for CIFAR-100)')
    parser.add_argument('--T', type=int, default=4,
                        help='Number of time steps for the SNN')
    parser.add_argument('--check_file_only', action='store_true',
                        help='Only validate the checkpoint file without running evaluation')
    
    args = parser.parse_args()
    
    # If only checking file, do that and exit
    if args.check_file_only:
        print("Checkpoint File Validation")
        print("="*30)
        is_valid, message = validate_checkpoint_file(args.state_dict_path)
        print(f"Result: {message}")
        
        if is_valid:
            try:
                print("\nTrying to load checkpoint...")
                checkpoint = torch.load(args.state_dict_path, map_location='cpu')
                print("✓ Checkpoint loaded successfully!")
                
                if isinstance(checkpoint, dict):
                    print(f"✓ Checkpoint is a dictionary with {len(checkpoint)} keys")
                    print(f"Keys: {list(checkpoint.keys())}")
                    
                    for key in ['state_dict', 'model', 'model_state_dict']:
                        if key in checkpoint:
                            state_dict = checkpoint[key]
                            print(f"✓ Found '{key}' with {len(state_dict)} parameters")
                            break
                    else:
                        print("✓ Using entire checkpoint as state dict")
                        state_dict = checkpoint
                        print(f"✓ State dict has {len(state_dict)} parameters")
                else:
                    print("✓ Checkpoint is direct state dict")
                    print(f"✓ Has {len(checkpoint)} parameters")
                    
            except Exception as e:
                print(f"✗ Failed to load checkpoint: {str(e)}")
                try_alternative_loading_methods(args.state_dict_path)
        
        return 0 if is_valid else 1
    
    print("SEW ResNet34 CIFAR-100 Test Evaluation")
    print("="*50)
    print(f"State dict path: {args.state_dict_path}")
    print(f"Data path: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Time steps (T): {args.T}")
    print("="*50)
    
    try:
        # Step 0: Validate the checkpoint file
        print("\n0. Validating checkpoint file...")
        is_valid, validation_message = validate_checkpoint_file(args.state_dict_path)
        print(f"Validation result: {validation_message}")
        
        if not is_valid:
            print("ERROR: Invalid checkpoint file!")
            print("\nTroubleshooting steps:")
            print("1. Verify the file path is correct")
            print("2. Check if the file was completely downloaded")
            print("3. Ensure the file is a valid PyTorch checkpoint (.pt, .pth, .tar)")
            print("4. Try re-downloading or re-copying the file")
            return 1
        
        # Step 1: Create the SEW ResNet34 model
        print("\n1. Creating SEW ResNet34 model...")
        model = create_sew_model(num_classes=args.num_classes, T=args.T)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model created with {n_parameters:,} trainable parameters")
        
        # Step 2: Load model parameters from state dictionary
        print("\n2. Loading model parameters...")
        model = load_model_from_state_dict(model, args.state_dict_path)
        
        # Step 3: Setup CIFAR-100 data loader
        print("\n3. Setting up CIFAR-100 data loader...")
        test_loader = setup_cifar100_loader(
            args.data_path, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers
        )
        
        # Step 4: Evaluate test accuracy
        print("\n4. Evaluating test accuracy...")
        results = evaluate_test_accuracy(model, test_loader, device=args.device)
        
        # Step 5: Save results
        print("\n5. Saving results...")
        results_file = os.path.join(os.path.dirname(args.state_dict_path), 'test_results.txt')
        with open(results_file, 'w') as f:
            f.write("SEW ResNet34 CIFAR-100 Test Results\n")
            f.write("="*40 + "\n")
            f.write(f"State dict: {args.state_dict_path}\n")
            f.write(f"Total samples: {results['total_samples']}\n")
            f.write(f"Top-1 Accuracy: {results['top1_accuracy']:.4f}%\n")
            f.write(f"Top-5 Accuracy: {results['top5_accuracy']:.4f}%\n")
            f.write(f"Average Top-1: {results['avg_top1']:.4f}%\n")
            f.write(f"Average Top-5: {results['avg_top5']:.4f}%\n")
        
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\nEvaluation completed successfully!")
    return 0


if __name__ == '__main__':
    exit(main())
