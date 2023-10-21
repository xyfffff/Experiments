import os
import torch
import numpy as np

def merge_tsp_data(output_file):
    all_data = []
    all_label = []

    for item in os.listdir():
        if os.path.isdir(item) and item.startswith('core_'):
            folder_name = item
            file_path = os.path.join(folder_name, 'data.pth')
            
            if os.path.exists(file_path):
                data, label = torch.load(file_path)
                all_data.extend(data)
                all_label.extend(label)
                print(f"Data from {file_path} has been loaded.")
            else:
                print(f"Warning: {file_path} does not exist and will be skipped.")

    # Convert lists to tensors
    all_data = np.array(all_data)
    all_label = np.array(all_label)

    # Save merged data
    torch.save((all_data, all_label), output_file)
    print(f"Merged data has been saved to {output_file}")

if __name__ == "__main__":
    output_file = 'train_data_50.pth'
    merge_tsp_data(output_file)
