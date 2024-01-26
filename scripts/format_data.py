import os
import json

def format_data(input_path, output_path):
    assert os.path.isdir(input_path), "Data path must be a directory."

    combined_data = []

    for file in os.listdir(input_path):
        filepath = os.path.join(input_path, file)
        
        with open(filepath, "r") as input_file:
            content = json.load(input_file)
        
        combined_data.extend(content)

    with open(output_path, "w") as output_file:
        json.dump(combined_data, output_file, indent=2)

    


if __name__ == "__main__":
    distributed_json_data_folder = "../dataset"
    output_path = "../formatted_data.json"

    format_data(distributed_json_data_folder, output_path)