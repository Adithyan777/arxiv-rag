import pandas as pd
import re
import pprint

csv_filename = 'final_paper_list.csv'

def extract_metadata(filename):
    # Extract ID from filename (assumes format: "id title")
    id_match = re.match(r'(\d{4}\.\d+)', filename)
    if not id_match:
        raise ValueError("Invalid filename format. Expected format: 'XXXX.XXXXX Title'")
    
    arxiv_id = id_match.group(1)
    
    try:
        df = pd.read_csv(csv_filename)

        # print(f"Extracting metadata for ID: {arxiv_id}")
        # print(f"Total papers in dataset: {len(df)}")

        # Convert ID column to string and remove any whitespace
        df['id'] = df['id'].astype(str).str.strip()
        paper_data = df[df['id'].str.contains(arxiv_id, regex=False)]
        
        if paper_data.empty:
            raise ValueError(f"No metadata found for ID: {arxiv_id}")
        
        # Initialize empty metadata dictionary
        metadata = {}
        
        # Define columns to check
        columns = ['id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors']
        
        # Only add non-empty values to metadata
        for col in columns:
            value = paper_data[col].iloc[0]
            # Check if value is not empty/null/NaN
            if pd.notna(value) and str(value).strip():
                metadata[col] = value
        
        return metadata
        
    except FileNotFoundError:
        raise FileNotFoundError(f"{csv_filename} not found")

# pprint.pprint(extract_metadata("2505.00312 aware-net_adaptive_weighted_averaging_for_robust_ensemble_network_in_deepfake_detection"))