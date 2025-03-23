import tensorflow as tf
import numpy as np
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import argparse
from PIL import Image
import matplotlib.pyplot as plt

def predict_damage(img_path, material_type=None, shoe_type=None, verbose=True, display_image=False):
    """
    Predict damage type and repair cost for a shoe image
    
    Args:
        img_path: Path to the shoe image file
        material_type: Optional material type (if None, will try to infer from metadata)
        shoe_type: Optional shoe type (if None, will try to infer from metadata)
        verbose: Whether to print detailed information during prediction
        display_image: Whether to display the image with prediction results
        
    Returns:
        Dictionary with prediction results
    """
    if verbose:
        print(f"Analyzing image: {img_path}")
    
    # Verify the image path exists
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}. Please check the path.")

    # Load the trained model
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_paths = [
        os.path.join(base_dir, "models", "shoe_damage_model.h5"),  # Project models directory
        r"E:\UM-Projects\ml\models\shoe_damage_model.h5",          # Original path
        r"E:\UM-Projects\ml\datasets\Damaged_Shoes\shoe_damage_model.h5"  # Alternate path
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        raise FileNotFoundError(
            "Model not found. Please ensure the model file exists in one of these locations:\n" +
            "\n".join(model_paths)
        )
    
    if verbose:
        print(f"Loading model from {model_path}")
    
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    # Define damage types and associated data
    damage_types = ["hole", "split-off", "sole_replacement", "Cracks", "discoloration_shoes", "Scratches", "scuffs_shoes"]
    repair_costs = {
        "hole": {"leather": (15, 35), "canvas": (10, 25), "suede": (20, 40), "rubber": (10, 30)},
        "split-off": {"leather": (25, 55), "canvas": (20, 45), "suede": (30, 60), "rubber": (20, 50)},
        "sole_replacement": {"leather": (30, 70), "canvas": (25, 60), "suede": (35, 75), "rubber": (25, 65)},
        "Cracks": {"leather": (20, 45), "canvas": (15, 35), "suede": (25, 50), "rubber": (15, 40)},
        "discoloration_shoes": {"leather": (15, 30), "canvas": (10, 20), "suede": (20, 35), "rubber": (10, 25)},
        "Scratches": {"leather": (10, 20), "canvas": (5, 15), "suede": (15, 25), "rubber": (5, 15)},
        "scuffs_shoes": {"leather": (10, 20), "canvas": (5, 15), "suede": (15, 25), "rubber": (5, 15)}
    }
    descriptions = {
        "hole": {
            "leather": "A visible hole in the leather upper or sole, approximately 0.5–2 cm.",
            "canvas": "A visible hole in the canvas material, approximately 0.5–2 cm.",
            "suede": "A visible hole in the suede material, approximately 0.5–2 cm.",
            "rubber": "A visible hole in the rubber sole, approximately 0.5–2 cm."
        },
        "split-off": {
            "leather": "A section of the leather shoe has split or detached, exposing inner layers.",
            "canvas": "A section of the canvas shoe has split or detached, exposing inner layers.",
            "suede": "A section of the suede shoe has split or detached, exposing inner layers.",
            "rubber": "A section of the rubber sole has split or detached."
        },
        "sole_replacement": {
            "leather": "The sole of the leather shoe is worn flat, cracked, or detached, reducing support.",
            "canvas": "The sole of the canvas shoe is worn flat, cracked, or detached, reducing support.",
            "suede": "The sole of the suede shoe is worn flat, cracked, or detached, reducing support.",
            "rubber": "The rubber sole is worn flat, cracked, or detached, reducing support."
        },
        "Cracks": {
            "leather": "Visible cracks in the leather material, often in the sole or upper.",
            "canvas": "Visible cracks in the canvas material, often around stress points.",
            "suede": "Visible cracks in the suede material, often in the upper.",
            "rubber": "Visible cracks in the rubber sole."
        },
        "discoloration_shoes": {
            "leather": "The leather material has faded or stained due to exposure.",
            "canvas": "The canvas material has faded or stained due to exposure.",
            "suede": "The suede material has faded or stained due to exposure.",
            "rubber": "The rubber material has faded or stained due to exposure."
        },
        "Scratches": {
            "leather": "Surface-level marks or abrasions on the leather upper.",
            "canvas": "Surface-level marks or abrasions on the canvas material.",
            "suede": "Surface-level marks or abrasions on the suede material.",
            "rubber": "Surface-level marks or abrasions on the rubber sole."
        },
        "scuffs_shoes": {
            "leather": "Scuff marks on the leather surface, often on the toe or heel.",
            "canvas": "Scuff marks on the canvas surface, often on the toe or heel.",
            "suede": "Scuff marks on the suede surface, often on the toe or heel.",
            "rubber": "Scuff marks on the rubber surface, often on the toe or heel."
        }
    }

    # If material_type or shoe_type is not provided, try to find them from metadata
    if material_type is None or shoe_type is None:
        if verbose:
            print("Attempting to infer material type and shoe type from metadata...")
            
        # Try to find relevant metadata files
        metadata_files = [
            r"E:\UM-Projects\ml\datasets\Damaged_Shoes\metadata\train_metadata.csv",
            r"E:\UM-Projects\ml\datasets\Damaged_Shoes\metadata\test_metadata.csv",
            r"E:\UM-Projects\ml\datasets\Damaged_Shoes\metadata\valid_metadata.csv"
        ]
        
        img_basename = os.path.basename(img_path)
        for metadata_file in metadata_files:
            if os.path.exists(metadata_file):
                df = pd.read_csv(metadata_file)
                # Try exact match first
                matches = df[df['image_path'] == img_path]
                if matches.empty:
                    # Try partial match with basename
                    matches = df[df['image_path'].str.contains(img_basename)]
                
                if not matches.empty:
                    if material_type is None:
                        material_type = matches['material_type'].iloc[0]
                        if verbose:
                            print(f"Found material type in metadata: {material_type}")
                    if shoe_type is None:
                        shoe_type = matches['shoe_type'].iloc[0]
                        if verbose:
                            print(f"Found shoe type in metadata: {shoe_type}")
                    break
    
    # Default values if still not found
    if material_type is None:
        material_type = "canvas"
        if verbose:
            print(f"Using default material type: {material_type}")
    
    if shoe_type is None:
        shoe_type = "unknown"
        if verbose:
            print(f"Using default shoe type: {shoe_type}")

    # Load and preprocess the image
    try:
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise RuntimeError(f"Error loading or preprocessing image: {e}")

    # Predict damage type
    try:
        if verbose:
            print("Running prediction...")
        prediction = model.predict(img_array, verbose=0)  # Suppress prediction verbose output
        predicted_class = np.argmax(prediction, axis=1)[0]
        damage_type = damage_types[predicted_class]
        
        # Get all confidence scores for visualization
        confidence_scores = prediction[0] * 100
        confidence = confidence_scores[predicted_class]
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")

    # Get description and cost based on damage type, material, and shoe type
    description = descriptions[damage_type].get(material_type, descriptions[damage_type]["canvas"])
    cost_min, cost_max = repair_costs[damage_type].get(material_type, repair_costs[damage_type]["canvas"])

    # Display the image with prediction results if requested
    if display_image:
        plt.figure(figsize=(12, 6))
        
        # Display image
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Damage Type: {damage_type} ({confidence:.1f}%)")
        plt.axis('off')
        
        # Display confidence bar chart
        plt.subplot(1, 2, 2)
        y_pos = np.arange(len(damage_types))
        plt.barh(y_pos, confidence_scores, align='center')
        plt.yticks(y_pos, damage_types)
        plt.xlabel('Confidence (%)')
        plt.title('Damage Type Confidence Scores')
        plt.tight_layout()
        plt.show()

    # Return results as a dictionary
    result = {
        "image_path": img_path,
        "damage_type": damage_type,
        "material_type": material_type,
        "shoe_type": shoe_type,
        "confidence": confidence,
        "description": description,
        "repair_cost_min": cost_min,
        "repair_cost_max": cost_max,
        "all_confidences": {damage_types[i]: confidence_scores[i] for i in range(len(damage_types))}
    }
    
    return result

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict shoe damage type and repair cost')
    parser.add_argument('--image', required=True, help='Path to the shoe image')
    parser.add_argument('--material', help='Material type (leather, canvas, suede, rubber)')
    parser.add_argument('--shoe_type', help='Type of shoe')
    parser.add_argument('--display', action='store_true', help='Display the image with prediction results')
    parser.add_argument('--batch', help='Path to a text file with multiple image paths to process')
    parser.add_argument('--output', help='Path to save prediction results as CSV (only for batch mode)')
    args = parser.parse_args()
    
    if args.batch:
        # Batch processing mode
        if not os.path.exists(args.batch):
            print(f"Batch file not found: {args.batch}")
            return
            
        with open(args.batch, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
            
        if not image_paths:
            print("No image paths found in the batch file.")
            return
            
        print(f"Processing {len(image_paths)} images in batch mode...")
        results = []
        
        for i, img_path in enumerate(image_paths):
            print(f"\nProcessing image {i+1}/{len(image_paths)}: {img_path}")
            try:
                result = predict_damage(img_path, args.material, args.shoe_type, verbose=True, display_image=False)
                results.append(result)
                print(f"Prediction: {result['damage_type']} (Confidence: {result['confidence']:.2f}%)")
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
        
        # Save results to CSV if output path is provided
        if args.output:
            results_df = pd.DataFrame(results)
            results_df.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
    else:
        # Single image processing mode
        try:
            # Make prediction
            result = predict_damage(args.image, args.material, args.shoe_type, verbose=True, display_image=args.display)
            
            # Output results
            print("\n===== Shoe Damage Assessment =====")
            print(f"Image Path: {result['image_path']}")
            print(f"Damage Type: {result['damage_type']}")
            print(f"Material Type: {result['material_type']}")
            print(f"Shoe Type: {result['shoe_type']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print(f"Description: {result['description']}")
            print(f"Repair Cost: ${result['repair_cost_min']}–${result['repair_cost_max']}")
            
            # Show top 3 probable damage types
            print("\nTop 3 probable damage types:")
            sorted_confidences = sorted(result['all_confidences'].items(), key=lambda x: x[1], reverse=True)
            for i, (damage_type, conf) in enumerate(sorted_confidences[:3]):
                print(f"  {i+1}. {damage_type}: {conf:.2f}%")
                
            print("==================================\n")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
