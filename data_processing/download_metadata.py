import requests
import os
import json

url = "https://efts.sec.gov/LATEST/search-index"
params = {
    "forms": ["10-K","10-Q"],
    "dateRange": "last-30",
    "from": 0,
    "size": 250  # max returned per page
}

HEADERS = {"User-Agent": "Ian ian.nlp.group.ion220@slmail.me"}
r = requests.get(url, headers=HEADERS, params=params)
data = r.json()

# Each hit contains:
#   - accessionNumber
#   - cik
#   - file_url
#   - filedAt

# save data to json
file_path = os.path.join(os.path.dirname(__file__), "..", "data", "filing_metadata.json")
with open(file_path, "w") as f:
    json.dump(data, f, indent=4)