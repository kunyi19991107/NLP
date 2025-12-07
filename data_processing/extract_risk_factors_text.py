import os
import json
import re
from bs4 import BeautifulSoup

DATA_DIR = 'data/html'
OUTPUT_FILE = 'data/risk_factors.json'

def extract_metadata(soup, filename):
    metadata = {
        'file_name': filename,
        'company_name': None,
        'file_date': None,
        'filing_type': None,
        'url': os.path.join(DATA_DIR, filename)
    }
    
    # Try to find XBRL tags
    name_tag = soup.find('ix:nonnumeric', {'name': 'dei:EntityRegistrantName'})
    if name_tag:
        metadata['company_name'] = name_tag.get_text(strip=True)
        
    date_tag = soup.find('ix:nonnumeric', {'name': 'dei:DocumentPeriodEndDate'})
    if date_tag:
        metadata['file_date'] = date_tag.get_text(strip=True)
        
    type_tag = soup.find('ix:nonnumeric', {'name': 'dei:DocumentType'})
    if type_tag:
        metadata['filing_type'] = type_tag.get_text(strip=True)

    # Fallback if XBRL tags are not found (basic regex or search)
    if not metadata['company_name']:
        # Try to find it in the first few lines or title
        title = soup.find('title')
        if title:
            metadata['company_name'] = title.get_text(strip=True).split(':')[0] # Heuristic

    return metadata

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def extract_risk_factors(soup):
    # Convert to text to make regex easier across tags
    # But we need to be careful not to lose the order.
    # Strategy: Find the start and end markers in the text content of the body.
    
    text = soup.get_text(separator='\n')
    
    # Pattern for Item 1A. Risk Factors
    # It can be "Item 1A.", "Item 1A", "RISK FACTORS"
    # We need to be careful about the Table of Contents which also has these strings.
    # Usually TOC has page numbers or dots.
    
    # Let's try to find the section headers.
    # A common pattern is "Item 1A." followed by "Risk Factors"
    
    # We will look for the LAST occurrence of "Item 1A" if there are multiple (TOC vs Body), 
    # BUT sometimes TOC is at the end? No, usually TOC is at start.
    # However, in 10-K, Risk Factors is Item 1A. In 10-Q, it is Item 1A in Part II.
    
    # Regex for start:
    # Case insensitive, allowing for some whitespace
    start_pattern = re.compile(r'Item\s+1A\.?\s*(Risk\s+Factors)?', re.IGNORECASE)
    
    # Regex for end:
    # Item 1B (Unresolved Staff Comments) or Item 2 (Properties / Unregistered Sales)
    end_pattern = re.compile(r'Item\s+(1B|2)\.?', re.IGNORECASE)
    
    lines = text.split('\n')
    start_indices = []
    for i, line in enumerate(lines):
        if start_pattern.search(line):
            # Check if it looks like a TOC entry (e.g. has dots and a number at the end)
            if re.search(r'\.{5,}\s*\d+$', line):
                continue
            start_indices.append(i)
            
    if not start_indices:
        return None
        
    # Heuristic: The actual section is likely the last one found, or we need to check context.
    # In the example file, TOC had links. The body had "Item 1A."
    
    # Let's try to extract from the last found start index to the next end index.
    # This is risky if there are multiple valid-looking starts.
    
    # Refined strategy:
    # Iterate through start_indices. For each, look for the nearest subsequent end_pattern.
    # If the text between them is substantial (> 100 chars), assume it's the section.
    
    for start_idx in reversed(start_indices):
        content = []
        found_end = False
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if end_pattern.search(line):
                # Check if it is a TOC entry
                if re.search(r'\.{5,}\s*\d+$', line):
                    continue
                found_end = True
                break
            content.append(line)
        
        if found_end:
            full_text = " ".join(content)
            if len(full_text) > 100: # Arbitrary threshold to avoid empty matches
                return clean_text(full_text)
                
    return None

def main():
    results = []
    
    if not os.path.exists(DATA_DIR):
        print(f"Directory {DATA_DIR} does not exist.")
        return

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith('.htm') and not filename.endswith('.html'):
            continue
            
        filepath = os.path.join(DATA_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                
            metadata = extract_metadata(soup, filename)
            risk_factors = extract_risk_factors(soup)
            
            if risk_factors:
                entry = metadata.copy()
                entry['risk_factors'] = risk_factors
                results.append(entry)
                print(f"Extracted Risk Factors from {filename}")
            else:
                print(f"No Risk Factors found in {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved {len(results)} entries to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
