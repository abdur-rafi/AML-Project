# Update Summary AttributeError Fix

## Problem
The error occurred during multi-objective optimization:
```
AttributeError: 'float' object has no attribute 'item'
```

This happened in `utils/tools/utility.py` at line 42 in the `update_summary` function.

## Root Cause
The original code assumed that lists in `rowd_in` contained either:
1. All tensors (which have `.item()` method), OR  
2. All floats/ints

However, in multi-objective mode, we have mixed types:
- `score`: List of tensors (from DE evolution)
- `top1`, `top5`, `val_loss`: Lists of floats (from validation)
- `f1_scores`: List of floats extracted from `population_objectives`

The original code tried to call `.item()` on float values when it detected the first element was a tensor.

## Original Problematic Code
```python
if len(value) > 0 and isinstance(value[0], torch.Tensor):
    list_temp = []
    for element in value:
        list_temp.append(round(element.item()))  # ❌ Fails when element is float
    rowd_in[key] = list_temp
elif len(value) > 0 and isinstance(value[0], (int, float)):
    rowd_in[key] = [round(float(element)) for element in value]
```

## Fix Applied
```python
if len(value) > 0:
    # Handle mixed types in the list more robustly
    list_temp = []
    for element in value:
        if isinstance(element, torch.Tensor):
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
```

## Key Changes
1. **Individual Type Checking**: Instead of checking only the first element type, we now check each element individually
2. **Robust Handling**: Each element is handled based on its actual type (tensor vs float vs other)
3. **Fallback Mechanism**: If an element can't be converted to float, it's kept as-is
4. **Mixed Type Support**: The function now handles lists with mixed tensor and float values

## Result
The `update_summary` function now properly handles:
- ✅ Lists of all tensors
- ✅ Lists of all floats/ints  
- ✅ Lists with mixed tensors and floats
- ✅ Multi-objective optimization logging

The multi-objective optimization should now continue without the AttributeError.
