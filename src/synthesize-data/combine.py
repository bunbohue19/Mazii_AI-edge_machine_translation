import json
import glob
from pathlib import Path

def merge_json_files(file_paths, output_file='data.json', mode='array'):
    """
    Merge multiple JSON files into one.
    
    Args:
        file_paths: List of file paths or a glob pattern (e.g., '*.json')
        output_file: Name of the output file
        mode: Merge mode - 'array', 'object', or 'named'
            - 'array': Combine all JSON contents into an array
            - 'object': Merge all objects into one (assumes all files contain objects)
            - 'named': Create object with filenames as keys
    """
    
    # Handle glob pattern
    if isinstance(file_paths, str):
        file_paths = glob.glob(file_paths)
    
    if not file_paths:
        print("No files found!")
        return
    
    data_list = []
    
    # Read all JSON files
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data_list.append({
                    'filename': Path(file_path).name,
                    'data': data
                })
                print(f"✓ Loaded: {file_path}")
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing {file_path}: {e}")
        except Exception as e:
            print(f"✗ Error reading {file_path}: {e}")
    
    if not data_list:
        print("No valid JSON files to merge!")
        return
    
    # Merge based on mode
    if mode == 'array':
        # Combine all data into an array
        merged = [item['data'] for item in data_list]
    
    elif mode == 'object':
        # Merge all objects into one
        merged = {}
        for item in data_list:
            if isinstance(item['data'], dict):
                merged.update(item['data'])
            else:
                # If not a dict, use filename as key
                key = item['filename'].replace('.json', '')
                merged[key] = item['data']
    
    elif mode == 'named':
        # Create object with filenames as keys
        merged = {}
        for item in data_list:
            key = item['filename'].replace('.json', '')
            merged[key] = item['data']
    
    else:
        print(f"Unknown mode: {mode}")
        return
    
    # Write merged data
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Successfully merged {len(data_list)} files into '{output_file}'")
        print(f"Mode: {mode}")
    except Exception as e:
        print(f"✗ Error writing output file: {e}")


# Example usage:
if __name__ == '__main__':
    # 1: Merge all JSON files in current directory into an array
    merge_json_files('../../data/synthetic-data/output/*.json', 
                    output_file='../../data/train/new_data.json', 
                    mode='object'
                )
    
    # 2: Merge specific files into one object
    # merge_json_files(['file1.json', 'file2.json', 'file3.json'], 
    #                  output_file='merged_object.json', 
    #                  mode='object')
    
    # 3: Merge with filenames as keys
    # merge_json_files('data/*.json', output_file='merged_named.json', mode='named')
    
    # 4: Merge files from specific directory
    # merge_json_files('path/to/files/*.json', output_file='output.json', mode='array')