

#!/usr/bin/env python
"""
Prediction script for decoder-based models (zero-shot).
Loads test data and predicts primary risk categories using pre-trained models.

Usage:
    python predict_decoder_model.py --model_name Qwen/Qwen2.5-7B-Instruct
    python predict_decoder_model.py --model_name mistralai/Mistral-7B-Instruct-v0.2
"""

import argparse
import json
import pathlib
import re
from typing import Dict, Any, Optional, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# Taxonomy from generate_silver_label.py
TAXONOMY = [
    {"parent_category": "Capital & Liquidity Risks", "sub_category": "Capital"},
    {"parent_category": "Capital & Liquidity Risks", "sub_category": "Liquidity"},
    {"parent_category": "Capital & Liquidity Risks", "sub_category": "Banking"},
    {"parent_category": "Capital & Liquidity Risks", "sub_category": "None"},
    {"parent_category": "Company Risks", "sub_category": "Internal"},
    {"parent_category": "Company Risks", "sub_category": "External"},
    {"parent_category": "Company Risks", "sub_category": "Structure"},
    {"parent_category": "Company Risks", "sub_category": "Location"},
    {"parent_category": "Company Risks", "sub_category": "Legal"},
    {"parent_category": "Company Risks", "sub_category": "Managerial"},
    {"parent_category": "Company Risks", "sub_category": "Stakeholders"},
    {"parent_category": "Company Risks", "sub_category": "None"},
    {"parent_category": "Market Risks", "sub_category": "Product Specific"},
    {"parent_category": "Market Risks", "sub_category": "Characteristics"},
    {"parent_category": "Market Risks", "sub_category": "Fluctuation"},
    {"parent_category": "Market Risks", "sub_category": "Model"},
    {"parent_category": "Market Risks", "sub_category": "Distress"},
    {"parent_category": "Market Risks", "sub_category": "None"},
    {"parent_category": "Investment Risks", "sub_category": "Allocation"},
    {"parent_category": "Investment Risks", "sub_category": "Transaction"},
    {"parent_category": "Investment Risks", "sub_category": "None"},
    {"parent_category": "Credit Risks", "sub_category": "Rate"},
    {"parent_category": "Credit Risks", "sub_category": "Contracts"},
    {"parent_category": "Credit Risks", "sub_category": "Terms"},
    {"parent_category": "Credit Risks", "sub_category": "Parties"},
    {"parent_category": "Credit Risks", "sub_category": "None"},
    {"parent_category": "None", "sub_category": "None"},
]

PRIMARY_CATEGORIES = sorted(list(set(item["parent_category"] for item in TAXONOMY)))

# Category definitions (from generate_silver_label.py)
CATEGORY_DEFINITIONS = {
    "Capital & Liquidity Risks": "Risks related to the company's ability to maintain sufficient capital and cash resources to operate and absorb losses. Failures in funding, liquidity, or banking access may threaten solvency or business continuity.",
    "Company Risks": "Risks arising from the company's internal operations, governance, structure, workforce, and external operating environment. These affect daily functioning, compliance, and long-term strategic stability.",
    "Market Risks": "Risks driven by changes in market conditions, prices, competition, or economic cycles. These impact asset values, revenues, and overall business performance.",
    "Investment Risks": "Risks related to how capital is invested, allocated, and executed across assets or strategies. Poor decisions or execution failures may reduce returns or increase losses.",
    "Credit Risks": "Risks arising from counterparties failing to meet financial obligations. These affect repayment, interest income, and recovery of principal.",
    "None": "Not related to any of the other categories."
}


def build_prompt(text: str) -> str:
    """
    Build a prompt for the decoder model with taxonomy and definitions.
    
    Args:
        text: The risk factor text to classify
    
    Returns:
        Formatted prompt string
    """
    categories_str = "\n".join(f"- {cat}" for cat in PRIMARY_CATEGORIES)
    
    definitions_str = "\n".join(
        f"- {cat}: {CATEGORY_DEFINITIONS.get(cat, '')}"
        for cat in PRIMARY_CATEGORIES
    )
    
    prompt = f"""You are a financial risk analyst specializing in SEC 10-K/10-Q risk factors.

Classify the following risk factor paragraph into ONE of these primary categories:

{categories_str}

Definitions of the primary categories:
{definitions_str}

Instructions:
- Use only the categories listed above. Refer to the definitions above for guidance.
- Use the "None" category if the paragraph is not related to any of the other categories.
- You may output only one category at maximum, and it must be one of the categories listed above.
- If the paragraph is meta-text (e.g. "risk factors have not materially changed") or you cannot confidently assign a category, return "None".
- Think in terms of *financial and business risk*, not generic statements.

Risk factor paragraph:
{text}

Primary category:"""
    
    return prompt


def parse_generated_text(generated_text: str) -> Optional[str]:
    """
    Parse the generated text to extract primary category.
    
    Args:
        generated_text: The text generated by the model
    
    Returns:
        Primary category string or None if parsing fails
    """
    generated_text = generated_text.strip()
    
    # Try to find a primary category in the generated text
    # Check for exact matches first (case-insensitive)
    for primary_cat in PRIMARY_CATEGORIES:
        # Check for exact match (case-insensitive)
        if primary_cat.lower() == generated_text.lower():
            return primary_cat
        # Check if category appears in the text
        if primary_cat.lower() in generated_text.lower():
            return primary_cat
    
    # Check for "None" explicitly
    if "none" in generated_text.lower() and "category" not in generated_text.lower():
        return "None"
    
    return None


def predict(
    text: str,
    model_name: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    max_new_tokens: int = 50,
    temperature: float = 0.1,
) -> Optional[str]:
    """
    Predict primary risk category using a decoder model.
    
    Args:
        text: The risk factor text to classify
        model_name: Name/path of the model
        tokenizer: Pre-loaded tokenizer
        model: Pre-loaded model
        device: Device to use
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Primary category string or None if parsing fails
    """
    # Build prompt
    prompt = build_prompt(text)
    
    # Tokenize - handle chat templates for instruction-tuned models
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        # Use chat template for instruction-tuned models
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
    else:
        # Fallback for models without chat template
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Parse the generated text
    result = parse_generated_text(generated_text)
    
    return result


def load_model(model_name: str, device: str):
    """
    Load model and tokenizer.
    
    Args:
        model_name: HuggingFace model name or path
        device: Device to load model on
    
    Returns:
        Tuple of (tokenizer, model)
    """
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    
    if device != "cuda":
        model.to(device)
    
    model.eval()
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully on {device}")
    return tokenizer, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot prediction of primary risk categories using decoder models"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"],
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to test JSON file (default: data/test/risk_factors_split_labeled_test.json)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (0.0 for greedy decoding)",
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
        data = data[:args.limit]
        print(f"Limited to {len(data)} records for testing")
    
    print(f"Total records to process: {len(data)}")
    
    # Load model
    tokenizer, model = load_model(args.model_name, device)
    
    # Process each record
    print("\nProcessing records...")
    for i, record in enumerate(data, 1):
        text = record.get("text", "")
        if not text:
            record["predicted_primary_category"] = None
            continue
        
        predicted_category = predict(
            text,
            args.model_name,
            tokenizer,
            model,
            device,
            args.max_new_tokens,
            args.temperature,
        )
        
        record["predicted_primary_category"] = predicted_category
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(data)} records...")
    
    # Create results directory
    results_dir = base_path / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Generate output filename
    model_safe_name = args.model_name.replace("/", "_").replace("-", "_")
    output_file = results_dir / f"predictions_{model_safe_name}.json"
    
    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Done! Processed {len(data)} records.")
    print(f"Results saved to {output_file}")

