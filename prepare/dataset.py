import json
import glob
from pathlib import Path

def combine_jsonl_files(input_directory: str, output_file: str) -> dict:
    """
    Combine multiple JSONL files into one and count objects in each file.
    
    Args:
        input_directory: Directory containing JSONL files
        output_file: Path to output combined JSONL file
    
    Returns:
        dict: Statistics about file counts
    """
    # Initialize statistics dictionary
    stats = {
        'per_file_counts': {},
        'total_count': 0
    }
    
    # Get all JSONL files in the directory
    jsonl_files = glob.glob(f"{input_directory}/*.jsonl")
    
    # Open output file for writing
    with open(output_file, 'w') as outfile:
        # Process each JSONL file
        for file_path in jsonl_files:
            file_count = 0
            
            # Read the JSON file
            with open(file_path, 'r') as infile:
                try:
                    # Try to load as a JSON array
                    json_array = json.load(infile)
                    if isinstance(json_array, list):
                        # Write each object as a separate line
                        for obj in json_array:
                            json.dump(obj, outfile)
                            outfile.write('\n')
                            file_count += 1
                    else:
                        print(f"Warning: {file_path} is not a JSON array")
                except json.JSONDecodeError as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue
                    
            print(f"Processing {file_path}: {file_count} objects")
            # Store statistics
            file_name = Path(file_path).name
            stats['per_file_counts'][file_name] = file_count
            stats['total_count'] += file_count
    
    return stats

# Example usage
if __name__ == "__main__":
    input_dir = "final_datasets"
    output_file = "combined_output.jsonl"
    
    stats = combine_jsonl_files(input_dir, output_file)
    
    # Print statistics
    print("\nFile-wise object counts:")
    for file_name, count in stats['per_file_counts'].items():
        print(f"{file_name}: {count} objects")
    print(f"\nTotal objects in combined file: {stats['total_count']}")


