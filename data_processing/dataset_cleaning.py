# load risk factors labeled, save only ones with non empty risk factors, and separate to 'primary risk factor' and 'secondary risk factor' columns
import pandas as pd
import json
import os
import pathlib

if __name__ == "__main__":
    _base_path = pathlib.Path(__file__).parent.parent
    with open(_base_path / "data/risk_factors_labeled.json", "r") as f:
        data = json.load(f)
    
    processed_data = []
    for record in data:
        if len(record['predicted_risk_categories']):
            record['primary_risk_factor'] = [c['parent_category'] for c in record['predicted_risk_categories']]
            record['secondary_risk_factor'] = [c['sub_category'] for c in record['predicted_risk_categories']]
            processed_data.append(record)
    
    with open(_base_path / "data/risk_factors_labeled_cleaned.json", "w") as f:
        json.dump(processed_data, f, indent=4)



    