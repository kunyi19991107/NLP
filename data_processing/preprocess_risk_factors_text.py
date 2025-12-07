import json
import re
import pathlib
from typing import List, Dict, Any

def remove_special_characters(text: str) -> str:
    """
    Remove special characters while preserving basic punctuation and alphanumeric characters.
    Removes unicode bullets, excessive whitespace, and other special symbols.
    """
    # Remove unicode bullets and other special symbols
    text = re.sub(r'[\u2022\u25cf\u2023\u2043\u2219]', ' ', text)  # Various bullet points
    text = re.sub(r'[\u201c\u201d]', '"', text)  # Smart quotes to regular quotes
    text = re.sub(r'[\u2018\u2019]', "'", text)  # Smart apostrophes to regular apostrophes
    text = re.sub(r'[\u2013\u2014]', '-', text)  # En/em dashes to regular dash
    text = re.sub(r'[\u00a0]', ' ', text)  # Non-breaking space to regular space
    
    # Remove other non-printable characters except newlines
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    
    # Normalize whitespace (multiple spaces to single space, but preserve newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()

def join_consecutive_lines(text: str) -> str:
    """
    Join consecutive lines that are part of the same sentence.
    A line is considered part of the previous sentence if:
    - The previous line doesn't end with sentence-ending punctuation (. ! ?)
    - The current line starts with lowercase or is a continuation
    """
    lines = text.split('\n')
    joined_lines = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            if joined_lines and joined_lines[-1] and not joined_lines[-1].endswith((' ', '\n')):
                joined_lines[-1] += ' '
            continue
        
        # If previous line exists and doesn't end with sentence-ending punctuation
        if joined_lines and joined_lines[-1]:
            prev_line = joined_lines[-1].rstrip()
            # Check if previous line ends with sentence-ending punctuation
            if not re.search(r'[.!?]\s*$', prev_line):
                # Check if current line starts with lowercase or is a continuation
                if line and (line[0].islower() or not line[0].isalnum()):
                    # Join with space
                    joined_lines[-1] = prev_line + ' ' + line
                    continue
        
        joined_lines.append(line)
    
    return '\n'.join(joined_lines)

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using a regex-based approach.
    Splits on sentence-ending punctuation followed by whitespace and capital letter or end of text.
    """
    # First, normalize the text
    text = text.strip()
    if not text:
        return []
    
    # Split on sentence endings: . ! ? followed by whitespace and capital letter
    # Also handle end of text
    # Use positive lookbehind to keep punctuation with the sentence
    # Pattern: period/exclamation/question, optional quote, whitespace, then capital letter or end
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?]["\'])\s+(?=[A-Z])|(?<=[.!?])\s*$|(?<=[.!?]["\'])\s*$'
    
    sentences = re.split(sentence_pattern, text)
    
    # Clean up and filter sentences
    result = []
    for sentence in sentences:
        sentence = sentence.strip()
        # Keep sentences that are meaningful (at least a few characters)
        if sentence and len(sentence) > 5:
            result.append(sentence)
    
    # If no splits occurred or all were filtered, treat entire text as one sentence
    if not result:
        result = [text] if text else []
    
    return result

def preprocess_risk_factors(risk_factors_text: str) -> List[str]:
    """
    Preprocess risk factors text: remove special characters, join lines, split into sentences.
    Returns list of sentences.
    """
    # Step 1: Remove special characters
    cleaned_text = remove_special_characters(risk_factors_text)
    
    # Step 2: Join consecutive lines that are part of the same sentence
    joined_text = join_consecutive_lines(cleaned_text)
    
    # Step 3: Split into sentences
    sentences = split_into_sentences(joined_text)
    
    return sentences

def main():
    """
    Main function to process risk_factors_text.json and create risk_factors_text_split.json
    """
    base_path = pathlib.Path(__file__).parent.parent
    input_file = base_path / "data" / "risk_factors_text.json"
    output_file = base_path / "data" / "risk_factors_text_split.json"
    
    # Read input data
    print(f"Reading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} records...")
    
    # Process each record
    processed_data = []
    for record in data:
        risk_factors_text = record.get('risk_factors', '')
        
        if not risk_factors_text:
            # Skip records with empty risk_factors
            continue
        
        # Preprocess and split into sentences
        sentences = preprocess_risk_factors(risk_factors_text)
        
        # Create a new record for each sentence
        for sentence_idx, sentence in enumerate(sentences):
            new_record = {
                **{k: v for k, v in record.items() if k != 'risk_factors'},
                'text': sentence,
                'sentence_index': sentence_idx,
                'total_sentences': len(sentences)
            }
            processed_data.append(new_record)
    
    # Save output
    print(f"Saving {len(processed_data)} processed records to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    
    print(f"Done! Processed {len(data)} original records into {len(processed_data)} sentence records.")

if __name__ == '__main__':
    main()

