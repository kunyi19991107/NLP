import requests
from pathlib import Path
import json
import tqdm

HEADERS = {"User-Agent": "Ian ian.nlp.group.ion220@slmail.me"}

DATA_DIR = Path(__file__).parent.parent / "data"

def file_url_from_hit(hit):
    src = hit["_source"]

    cik = int(src["ciks"][0])              # strips leading zeros
    adsh = src["adsh"]                     # e.g. '0001213900-25-113607'
    adsh_nodash = adsh.replace("-", "")
    
    # filename is after ':' in _id
    file_name = hit["_id"].split(":", 1)[1]  # 'ea0266581-10q_westin.htm'
    
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh_nodash}/{file_name}"

# load data
with open(DATA_DIR / "filing_metadata.json", "r") as f:
    data = json.load(f)

output_dir = DATA_DIR / "html"
output_dir.mkdir(parents=True, exist_ok=True)
for hit in tqdm.tqdm(data['hits']['hits']):
    url = file_url_from_hit(hit)
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    (output_dir / hit["_id"]).write_bytes(resp.content)
