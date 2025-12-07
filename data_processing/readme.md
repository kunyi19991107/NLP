# data processing pipeline

The data processing pipeline is a series of scripts that extract risk factors from 10-K and 10-Q filings, and then classify them into risk factor categories using a pre-trained model.

The pipeline is as follows:
1. download_metadata.py: download metadata from SEC EDGAR
2. download_html.py: download html files from SEC EDGAR
3. extract_risk_factors_text.py: extract risk factors from 10-K and 10-Q filings
4. preprocess_risk_factors_text.py: segment to sentences, remove special characters, join consecutive lines supposedly in one sentence
5. generate_silver_label.py: generate silver labels for risk factors using GPT-4.1-mini
6. dataset_split.py: split the labeled dataset into train (80%), validation (10%), and test (10%) sets, extracting parent_category and sub_category as separate fields


# Dataset Statistics

## Distribution by Category

| Category | Train | Validation | Test |
|----------|-------|------------|------|
| Capital & Liquidity Risks | 149 (5.4%) | 23 (6.6%) | 16 (4.6%) |
| Company Risks | 1,555 (55.8%) | 206 (59.2%) | 196 (56.2%) |
| Market Risks | 490 (17.6%) | 43 (12.4%) | 66 (18.9%) |
| Investment Risks | 113 (4.1%) | 16 (4.6%) | 12 (3.4%) |
| Credit Risks | 162 (5.8%) | 21 (6.0%) | 16 (4.6%) |