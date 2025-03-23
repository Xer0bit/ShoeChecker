import os
import pandas as pd

# Define base paths
base_path = r"E:\UM-Projects\ml\datasets\Damaged_Shoes"
material_base_path = r"E:\UM-Projects\ml\datasets\shoeMaterial"
shoe_type_base_path = r"E:\UM-Projects\ml\datasets\shoeType"

# Define splits
splits = ["train", "test", "valid"]

# Define known damage types to filter out non-damage-type folders
known_damage_types = [
    "hole", "split-off", "sole_replacement", "Cracks", 
    "discoloration_shoes", "dicoloration_shoes", "Scratches", "scuffs_shoes"
]

# Define repair costs for each damage type, varying by material (example values)
repair_costs = {
    "hole": {"leather": (15, 35), "canvas": (10, 25), "suede": (20, 40), "rubber": (10, 30)},
    "split-off": {"leather": (25, 55), "canvas": (20, 45), "suede": (30, 60), "rubber": (20, 50)},
    "sole_replacement": {"leather": (30, 70), "canvas": (25, 60), "suede": (35, 75), "rubber": (25, 65)},
    "Cracks": {"leather": (20, 45), "canvas": (15, 35), "suede": (25, 50), "rubber": (15, 40)},
    "discoloration_shoes": {"leather": (15, 30), "canvas": (10, 20), "suede": (20, 35), "rubber": (10, 25)},
    "dicoloration_shoes": {"leather": (15, 30), "canvas": (10, 20), "suede": (20, 35), "rubber": (10, 25)},  # Adding the typo variant
    "Scratches": {"leather": (10, 20), "canvas": (5, 15), "suede": (15, 25), "rubber": (5, 15)},
    "scuffs_shoes": {"leather": (10, 20), "canvas": (5, 15), "suede": (15, 25), "rubber": (5, 15)}
}

# Build mappings
material_mapping = {}
for material_type in os.listdir(material_base_path):
    material_folder = os.path.join(material_base_path, material_type)
    if os.path.isdir(material_folder):
        for img_name in os.listdir(material_folder):
            if img_name.lower().endswith(".jpg"):
                material_mapping[img_name] = material_type

shoe_type_mapping = {}
for shoe_type in os.listdir(shoe_type_base_path):
    shoe_type_folder = os.path.join(shoe_type_base_path, shoe_type)
    if os.path.isdir(shoe_type_folder):
        for img_name in os.listdir(shoe_type_folder):
            if img_name.lower().endswith(".jpg"):
                shoe_type_mapping[img_name] = shoe_type

# Create output directory for metadata files if it doesn't exist
metadata_dir = os.path.join(base_path, "metadata")
os.makedirs(metadata_dir, exist_ok=True)

# Generate CSV for each split
for split in splits:
    split_path = os.path.join(base_path, split)
    data = []

    # Iterate through each damage type folder in the split
    for folder_name in os.listdir(split_path):
        folder_path = os.path.join(split_path, folder_name)
        if os.path.isdir(folder_path):
            # Check if the folder name is a known damage type or similar to one
            is_damage_type = False
            matched_damage_type = None
            
            # First check exact matches
            if folder_name in known_damage_types:
                is_damage_type = True
                matched_damage_type = folder_name
            else:
                # Then check for similar names (case-insensitive, ignoring underscores)
                for damage_type in known_damage_types:
                    if (damage_type.lower() == folder_name.lower() or 
                        damage_type.replace('_', '').lower() == folder_name.replace('_', '').lower()):
                        is_damage_type = True
                        matched_damage_type = damage_type
                        break
            
            if is_damage_type:
                for img_name in os.listdir(folder_path):
                    if img_name.lower().endswith(".jpg"):
                        img_path = os.path.join(folder_path, img_name)
                        # Match the image to its material type and shoe type
                        material_type = material_mapping.get(img_name, "unknown")
                        shoe_type = shoe_type_mapping.get(img_name, "unknown")
                        
                        # Get repair costs based on material type
                        if material_type in repair_costs[matched_damage_type]:
                            cost_min, cost_max = repair_costs[matched_damage_type][material_type]
                        else:
                            cost_min, cost_max = repair_costs[matched_damage_type]["canvas"]  # Default to canvas
                            
                        data.append([img_path, matched_damage_type, material_type, shoe_type, cost_min, cost_max])
            else:
                print(f"Skipping directory '{folder_name}' - not recognized as a damage type")

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data, columns=["image_path", "damage_type", "material_type", "shoe_type", "repair_cost_min", "repair_cost_max"])
    output_path = os.path.join(metadata_dir, f"{split}_metadata.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path} with {len(df)} entries")

print("Metadata generation complete!")
print(f"CSV files are saved in: {metadata_dir}")
