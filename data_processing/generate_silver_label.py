#!/usr/bin/env python
"""
Use gpt-4.1-mini to infer company risk factor categories for
paragraphs from 10-K / 10-Q filings.

Usage:
    python generate_silver_label.py --input input.json --output output.json
"""

import argparse
import json
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


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

TAXONOMY_STR = "\n".join(
    f"- {item['parent_category']} - {item['sub_category']}"
    for item in TAXONOMY
)

SYSTEM_PROMPT = f"""
You are a financial risk analyst specialising in SEC 10-K/10-Q risk factors.

You must map each input paragraph to zero or one risk factor categories
from the following taxonomy. Each category consists of a parent category
and a sub category:

{TAXONOMY_STR}

Definitions of the primary categories:
- Capital & Liquidity Risks: Risks related to the company’s ability to maintain sufficient capital and cash resources to operate and absorb losses. Failures in funding, liquidity, or banking access may threaten solvency or business continuity.
- Company Risks: Risks arising from the company’s internal operations, governance, structure, workforce, and external operating environment. These affect daily functioning, compliance, and long-term strategic stability.
- Market Risks: Risks driven by changes in market conditions, prices, competition, or economic cycles. These impact asset values, revenues, and overall business performance.
- Investment Risks: Risks related to how capital is invested, allocated, and executed across assets or strategies. Poor decisions or execution failures may reduce returns or increase losses.
- Credit Risks: Risks arising from counterparties failing to meet financial obligations. These affect repayment, interest income, and recovery of principal.

Instructions:
- Use only the categories listed above. Refer to the definitions of the primary categories above for guidance.
- Use the "None" category if the paragraph is not related to any of the other categories.
- You may output only one category at maximum, and it must be one of the categories listed above.
- If the paragraph is meta-text (e.g. "risk factors have not materially changed") or you cannot confidently assign a category, return an empty dictionary.
- Think in terms of *financial and business risk*, not generic statements.

Return a single JSON object with a dictionary with the following keys:
- "parent_category": string
- "sub_category": string
- "rationale": short string explaining why this category applies
"""

# --------- OpenAI client setup ---------

def init_client() -> OpenAI:
    """
    Initialize OpenAI client, loading API key from .env if needed.
    Expects OPENAI_API_KEY in environment or .env file.
    """
    load_dotenv()  # loads .env into environment if present
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not found. Please set it in your environment or .env file."
        )
    return OpenAI()


# --------- LLM call ---------


def classify_paragraph(client: OpenAI, paragraph: str) -> Dict[str, Any] | None:
    """
    Call gpt-4.1-mini to classify a single paragraph.
    Returns a dictionary with {parent_category, sub_category, rationale} or None.
    Returns None if parent_category is "None" or if no valid category is found.
    """
    # Short-circuit very empty strings
    if not paragraph or not paragraph.strip():
        return None

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Classify the following 10-K/10-Q risk factor paragraph:\n\n"
                    f"{paragraph}"
                ),
            },
        ],
        # feel free to tweak
        temperature=0.1,
        max_tokens=500,
    )

    content = completion.choices[0].message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Very defensive fallback
        return None

    # The prompt asks for a single JSON object with parent_category, sub_category, rationale
    # Check if it's an empty dict (as per prompt: "return an empty dictionary")
    if not parsed or not isinstance(parsed, dict):
        return None

    # Check if the response has the expected keys
    parent = parsed.get("parent_category")
    sub = parsed.get("sub_category")
    rationale = parsed.get("rationale", "")

    # If either parent or sub is missing (None or empty string), return None
    if parent is None or sub is None or parent == "" or sub == "":
        return None

    # If parent_category is "None", return None
    if parent == "None":
        return None

    # Validate against taxonomy
    valid_pairs = {
        (item["parent_category"], item["sub_category"]) for item in TAXONOMY
    }

    if (parent, sub) in valid_pairs:
        return {
            "parent_category": parent,
            "sub_category": sub,
            "rationale": rationale,
        }

    return None


# --------- IO helpers ---------


def load_input(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of dictionaries.")
    return data


def save_output(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_existing_output(path: str) -> List[Dict[str, Any]]:
    """Load existing output file if it exists, otherwise return empty list."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except (json.JSONDecodeError, IOError):
            pass
    return []


# --------- Main pipeline ---------


def main(input_path: str, output_path: str, input_field: str, limit: int | None = None) -> None:
    client = init_client()
    rows = load_input(input_path)

    if limit is not None:
        rows = rows[:limit]

    # Try to load existing output to resume from checkpoint
    enriched_rows = load_existing_output(output_path)
    start_index = len(enriched_rows)
    
    if start_index > 0:
        print(f"Resuming from checkpoint: {start_index} records already processed")
        rows = rows[start_index:]

    total_rows = start_index + len(rows)
    for i, row in tqdm(enumerate(rows, start=1), total=len(rows)):
        paragraph = row.get(input_field, "")
        labels = classify_paragraph(client, paragraph)
        new_row = dict(row)
        new_row["predicted_risk_categories"] = labels
        enriched_rows.append(new_row)
        current_total = start_index + i
        print(f"[{current_total}/{total_rows}] processed {row.get('file_name', 'N/A')} {row.get('sentence_index', 0)}/{row.get('total_sentences', 0)}")
        
        # Save every 100 records
        if current_total % 100 == 0:
            save_output(output_path, enriched_rows)
            print(f"Checkpoint: Saved {current_total} records to {output_path}")

    # Final save
    save_output(output_path, enriched_rows)
    print(f"Final save: Saved {len(enriched_rows)} records to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify 10-K/10-Q risk factor paragraphs into risk categories."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input JSON file (list of dicts with 'risk_factors' field).",
    )
    parser.add_argument(
        "--input_field",
        "-f",
        required=True,
        help="Name of the field in the input JSON file that contains the text to classify.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output JSON file with added 'predicted_risk_categories'.",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Optional: only process first N records (for testing).",
    )

    args = parser.parse_args()
    main(args.input, args.output, args.input_field, args.limit)
