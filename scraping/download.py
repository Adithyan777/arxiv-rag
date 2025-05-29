import pandas as pd
import requests
import os
from pathlib import Path

# Create downloads directory if it doesn't exist
DOWNLOAD_DIR = Path(__file__).parent.parent / "papers"
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Read the CSV file
df = pd.read_csv("arxiv_scrape_results_filtered.csv")
downloaded_count = 0

def download_paper(paper_id):
    url = f"http://arxiv.org/pdf/{paper_id}"
    response = requests.get(url)
    if response.status_code == 200:
        filepath = DOWNLOAD_DIR / f"{paper_id}.pdf"
        with open(filepath, "wb") as f:
            f.write(response.content)
        return True
    return False

# Iterate through papers
for _, row in df.iterrows():
    if downloaded_count >= 40:
        print("\nReached maximum limit of 40 downloads. Exiting...")
        break
        
    print("\n" + "="*80)
    print(f"\nTitle: {row['title']}\n")
    print("Abstract:")
    print("-"*80)
    print(row['abstract'])
    print("-"*80)
    
    while True:
        choice = input("\nWould you like to download this paper? (y/n): ").lower()
        if choice in ['y', 'n']:
            break
        print("Please enter 'y' or 'n'")
    
    if choice == 'y':
        print(f"Downloading paper {row['id']}...")
        if download_paper(row['id']):
            downloaded_count += 1
            print(f"Successfully downloaded! ({downloaded_count}/40)")
        else:
            print("Failed to download paper")
    
    print("\nMoving to next paper...")

print("\nScript completed!")

