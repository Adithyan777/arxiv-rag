import json
import glob
import os
import argparse

def combine_jsonl_files(input_dir):
  # Use input directory for output file
  output_file = os.path.join(input_dir, "combined_output.jsonl")
  
  # Get all jsonl files in the directory
  jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
  
  with open(output_file, 'w') as outfile:
    # Process each input file
    for jsonl_file in jsonl_files:
      # Skip the output file if it exists
      if jsonl_file == output_file:
        continue
      print(f"Processing {jsonl_file}...")
      with open(jsonl_file, 'r') as infile:
        # Read each line and write to output file
        for line in infile:
          outfile.write(line)

if __name__ == "__main__":
  # Define input path (using absolute path)
  INPUT_DIR = "/Users/adithyankrishnan/Downloads/arxiv-rag/datasets"
  
  combine_jsonl_files(INPUT_DIR)
  print("Finished combining JSONL files!")