#!/usr/bin/env python
"""
Prediction script for fine-tuned FinBERT encoder model.
Loads test data and predicts primary risk categories using the fine-tuned model.

Usage:
    python predict_encoder_model.py --model_path ./model_finetuned
    python predict_encoder_model.py --model_path ./model_finetuned/checkpoint-3
"""

import argparse
import json
import pathlib
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def predict_batch(
    texts: List[str],
    model_path: str,
    device: str,
    batch_size: int = 32,
    max_length: int = 256,
):
    """
    Predict primary categories for a batch of texts.
    
    Args:
        texts: List of texts to classify
        model_path: Path to the fine-tuned model
        device: Device to use
        batch_size: Batch size for processing
        max_length: Maximum sequence length
    
    Returns:
        List of predicted primary category strings
    """
    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Get id2label mapping from model config
    id2label = model.config.id2label
    print(f"Model has {len(id2label)} classes: {list(id2label.values())}")
    
    predictions = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_ids = torch.argmax(probabilities, dim=-1).cpu().tolist()
        
        # Convert IDs to category names
        batch_predictions = [id2label[str(pred_id)] for pred_id in predicted_ids]
        predictions.extend(batch_predictions)
        
        if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(texts):
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} records...")
    
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict primary risk categories using fine-tuned FinBERT"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model_finetuned",
        help="Path to fine-tuned model directory (default: ./model_finetuned)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to test JSON file (default: data/test/risk_factors_split_labeled_test.json)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length (default: 256)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, mps, cpu, or None for auto)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records to process (for testing)",
    )

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Determine input file path
    base_path = pathlib.Path(__file__).parent.parent
    if args.input_file:
        input_file = pathlib.Path(args.input_file)
    else:
        input_file = base_path / "data" / "test" / "risk_factors_split_labeled_test.json"

    # Load test data
    print(f"Loading test data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if args.limit:
        data = data[: args.limit]
        print(f"Limited to {len(data)} records for testing")
    
    print(f"Total records to process: {len(data)}")
    
    # Extract texts
    texts = [record.get("text", "") for record in data]
    
    # Predict
    print("\nPredicting categories...")
    predictions = predict_batch(
        texts,
        args.model_path,
        device,
        args.batch_size,
        args.max_length,
    )
    
    # Add predictions to records
    for record, predicted_category in zip(data, predictions):
        record["predicted_primary_category"] = predicted_category
    
    # Create results directory
    results_dir = base_path / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Generate output filename
    model_path_name = pathlib.Path(args.model_path).name
    output_file = results_dir / f"predictions_finbert_{model_path_name}.json"
    
    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Done! Processed {len(data)} records.")
    print(f"Results saved to {output_file}")
