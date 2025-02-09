import os
import json
from collections import Counter

# Define paths to datasets
datasets = ["hold", "test", "tier1", "tier3"]
base_path = "C:/Users/yusuf/Desktop/budaas/geotiffs"

# Initialize counters for each dataset
damage_counts = {dataset: Counter() for dataset in datasets}

# Iterate over each dataset split
for dataset in datasets:
    label_folder = os.path.join(base_path, dataset, "labels")
    
    # Iterate over label files in the folder
    for label_file in os.listdir(label_folder):
        if label_file.endswith(".json"):  # Check if it's a JSON label file
            with open(os.path.join(label_folder, label_file), "r") as file:
                data = json.load(file)
                print(label_file)
                
                features = data["features"]["lng_lat"]
                # Process each building annotation
                for feature in features:
                    if "subtype" in feature["properties"]:
                        damage = feature["properties"]["subtype"]  # Damage level
                        damage_counts[dataset][damage] += 1

# Print results
for dataset, counts in damage_counts.items():
    print(f"Dataset: {dataset}")
    for damage, count in counts.items():
        print(f"  {damage}: {count}")
