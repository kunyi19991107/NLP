#!/usr/bin/env python
"""
Split the risk_factors_split_labeled.json dataset into train, validation, and test sets
with a ratio of 8:1:1 (80% train, 10% val, 10% test).

Usage:
    python dataset_split.py
"""

import json
import pathlib
import random
import sys
from typing import List, Dict, Any

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from constant import PRIMARY_CATEGORY_TO_INDEX


def transform_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a record by extracting parent_category and sub_category from
    predicted_risk_categories and removing the predicted_risk_categories field.
    
    Args:
        record: Original record dictionary
    
    Returns:
        Transformed record with parent_category and sub_category as separate fields
    """
    new_record = {k: v for k, v in record.items() if k != "predicted_risk_categories"}
    
    # Extract parent_category and sub_category from predicted_risk_categories
    predicted = record.get("predicted_risk_categories")
    if predicted is not None and isinstance(predicted, dict):
        new_record["parent_category"] = predicted.get("parent_category")
        new_record["sub_category"] = predicted.get("sub_category")
    else:
        # If predicted_risk_categories is None or invalid, set to None
        new_record["parent_category"] = None
        new_record["sub_category"] = None
    
    return new_record


def split_data(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: List of data records
        train_ratio: Proportion for training set (default: 0.8)
        val_ratio: Proportion for validation set (default: 0.1)
        test_ratio: Proportion for test set (default: 0.1)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Shuffle the data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split indices
    total = len(shuffled_data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split the data
    train_data = shuffled_data[:train_end]
    val_data = shuffled_data[train_end:val_end]
    test_data = shuffled_data[val_end:]
    
    return train_data, val_data, test_data


def main():
    """Main function to split the dataset."""
    base_path = pathlib.Path(__file__).parent.parent
    input_file = base_path / "data" / "risk_factors_split_labeled.json"
    
    # Output files
    train_file = base_path / "data" / "train" / "risk_factors_split_labeled_train.json"
    val_file = base_path / "data" / "val" / "risk_factors_split_labeled_val.json"
    test_file = base_path / "data" / "test" / "risk_factors_split_labeled_test.json"
    
    # Load data
    print(f"Loading data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Total records: {len(data)}")
    
    # Split data (8:1:1 ratio)
    print("Splitting data into train (80%), val (10%), test (10%)...")
    train_data, val_data, test_data = split_data(
        data,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
    )
    
    print(f"Train set: {len(train_data)} records ({100*len(train_data)/len(data):.1f}%)")
    print(f"Val set: {len(val_data)} records ({100*len(val_data)/len(data):.1f}%)")
    print(f"Test set: {len(test_data)} records ({100*len(test_data)/len(data):.1f}%)")
    
    # Transform records: extract parent_category and sub_category, remove predicted_risk_categories
    print("\nTransforming records (extracting parent_category and sub_category)...")
    train_data = [transform_record(record) for record in train_data]
    val_data = [transform_record(record) for record in val_data]
    test_data = [transform_record(record) for record in test_data]
    
    # Create output directories if they don't exist
    train_file.parent.mkdir(parents=True, exist_ok=True)
    val_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    print(f"\nSaving train set to {train_file}...")
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saving validation set to {val_file}...")
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saving test set to {test_file}...")
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print("\nDone! Dataset split completed successfully.")

    # add stats of each primary_category in train, val, test sets
    print("\n" + "="*60)
    print("Category distribution statistics:")
    print("="*60)
    for primary_category in PRIMARY_CATEGORY_TO_INDEX.keys():
        train_count = len([d for d in train_data if d.get("parent_category") == primary_category])
        val_count = len([d for d in val_data if d.get("parent_category") == primary_category])
        test_count = len([d for d in test_data if d.get("parent_category") == primary_category])
        
        print(f"\n{primary_category}:")
        print(f"  Train: {train_count} records ({100*train_count/len(train_data):.1f}%)" if len(train_data) > 0 else "  Train: 0 records (0.0%)")
        print(f"  Val:   {val_count} records ({100*val_count/len(val_data):.1f}%)" if len(val_data) > 0 else "  Val:   0 records (0.0%)")
        print(f"  Test:  {test_count} records ({100*test_count/len(test_data):.1f}%)" if len(test_data) > 0 else "  Test:  0 records (0.0%)")


if __name__ == "__main__":
    main()

