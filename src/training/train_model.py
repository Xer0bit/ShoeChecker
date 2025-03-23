import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.optimizers import Adam
import os
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras.callbacks as callbacks

def train_model(csv_path, epochs=20, batch_size=32, image_size=(224, 224), validation_split=0.2, output_dir=None, learning_rate=0.001, model_type='mobilenet'):
    """
    Train a CNN model to classify shoe damage types using transfer learning
    
    Args:
        csv_path: Path to CSV file containing image paths and damage types
        epochs: Number of training epochs
        batch_size: Batch size for training
        image_size: Target size for images (width, height)
        validation_split: Fraction of data to use for validation
        output_dir: Directory to save the model and results
        learning_rate: Initial learning rate for the optimizer
        model_type: Type of model to use ('custom', 'mobilenet', or 'resnet')
    """
    print(f"Loading data from {csv_path}...")
    
    # Step 1: Load the CSV data
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} records from the CSV file.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        
        # Try to use the metadata files instead
        metadata_files = [
            r"E:\UM-Projects\ml\datasets\Damaged_Shoes\metadata\train_metadata.csv",
            r"E:\UM-Projects\ml\datasets\Damaged_Shoes\metadata\test_metadata.csv",
            r"E:\UM-Projects\ml\datasets\Damaged_Shoes\metadata\valid_metadata.csv"
        ]
        
        # Combine metadata files
        df_list = []
        for file in metadata_files:
            if os.path.exists(file):
                print(f"Loading alternative metadata from {file}...")
                df_temp = pd.read_csv(file)
                df_list.append(df_temp)
        
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
            print(f"Loaded {len(df)} records from alternative metadata files.")
        else:
            raise Exception("Could not find any metadata files.")

    # Step 2: Define damage types and map labels to indices
    damage_types = ["hole", "split-off", "sole_replacement", "Cracks", "discoloration_shoes", "Scratches", "scuffs_shoes"]
    label_map = {label: idx for idx, label in enumerate(damage_types)}
    
    # Ensure all image files exist
    before_count = len(df)
    df = df[df['image_path'].apply(os.path.exists)]
    after_count = len(df)
    
    if before_count != after_count:
        print(f"WARNING: {before_count - after_count} image files were not found and have been removed from the dataset.")
    
    print(f"Found {after_count} valid image files.")

    # Check if we have any valid images
    if after_count == 0:
        raise ValueError("No valid image files found. Please check your dataset paths and ensure images exist.")

    # Step 3: Replace "dicoloration_shoes" with "discoloration_shoes" to match the damage_types list
    df['damage_type'] = df['damage_type'].replace("dicoloration_shoes", "discoloration_shoes")
    
    # Print class distribution
    print("\nClass distribution:")
    for damage_type in damage_types:
        count = len(df[df['damage_type'] == damage_type])
        print(f"  {damage_type}: {count} samples")
        
        # If any class has fewer than 2 samples, we can't stratify
        if count < 2:
            print(f"WARNING: '{damage_type}' has fewer than 2 samples. Disabling stratification.")
            stratify = None
            break
    else:
        # This will only execute if the loop completes without breaking
        stratify = df['damage_type']

    # Step 4: Split the data into training and validation sets directly
    # This avoids the "ran out of data" warning with flow_from_dataframe
    train_df, val_df = train_test_split(
        df, 
        test_size=validation_split, 
        stratify=stratify,  # Use None if we don't have enough samples
        random_state=42
    )
    
    print(f"\nSplit data into {len(train_df)} training samples and {len(val_df)} validation samples")

    # Step 5: Create separate ImageDataGenerators for training and validation
    train_datagen = ImageDataGenerator(
        rescale=1./255,          # Normalize pixel values to [0, 1]
        rotation_range=20,       # Data augmentation: rotate images up to 20 degrees
        width_shift_range=0.2,   # Data augmentation: shift width
        height_shift_range=0.2,  # Data augmentation: shift height
        horizontal_flip=True,    # Data augmentation: flip horizontally
        zoom_range=0.1,          # Data augmentation: random zoom
        shear_range=0.1,         # Data augmentation: shear
        brightness_range=[0.8, 1.2]  # Data augmentation: brightness
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255  # Only normalize validation images, no augmentation
    )

    # Step 6: Create generators
    print("\nCreating data generators...")
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="image_path",       # Column with image paths
        y_col="damage_type",      # Column with labels
        target_size=image_size,   # Resize images to target size
        batch_size=batch_size,    # Number of images per batch
        class_mode="categorical", # Multi-class classification
        classes=damage_types,     # Explicitly specify the class names
        shuffle=True              # Shuffle the training data
    )

    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="image_path",
        y_col="damage_type",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=damage_types,
        shuffle=False  # No need to shuffle validation data
    )

    # Setup callbacks - ADDING THIS BEFORE BUILDING THE MODEL
    if output_dir is None:
        output_dir = r"E:\UM-Projects\ml\models"
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"model_checkpoint_{timestamp}_{{epoch:02d}}_{{val_accuracy:.4f}}.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, f"logs_{timestamp}"),
            histogram_freq=1
        )
    ]

    # Step 7: Build model with transfer learning
    print(f"\nBuilding {model_type} model...")
    
    if model_type == 'mobilenet':
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*image_size, 3)
        )
    elif model_type == 'resnet':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*image_size, 3)
        )
    else:
        # Use the existing custom model architecture
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(*image_size, 3)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(len(damage_types), activation='softmax')  # Output layer with 7 classes
        ])
        # Skip transfer learning steps for custom model
        print("\nUsing custom model without transfer learning...")
        return train_custom_model(model, train_generator, validation_generator, epochs, 
                                callbacks_list, learning_rate, output_dir, val_df, damage_types, image_size, batch_size)

    # Initially freeze the base model
    base_model.trainable = False

    # Build the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(len(damage_types), activation='softmax')
    ])

    # Step 8: Train the top layers first
    print("\nStage 1: Training the top layers...")
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Step 10: Calculate steps per epoch properly
    # This prevents the "ran out of data" warning
    steps_per_epoch = len(train_generator)
    validation_steps = len(validation_generator)

    history_initial = model.fit(
        train_generator,
        epochs=min(10, epochs // 2),  # Train top layers for half the epochs or 10, whichever is smaller
        validation_data=validation_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=1
    )

    # Step 9: Fine-tune the model
    print("\nStage 2: Fine-tuning the model...")
    base_model.trainable = True

    # Freeze earlier layers
    if model_type == 'mobilenet':
        for layer in base_model.layers[:-20]:  # Unfreeze last 20 layers of MobileNetV2
            layer.trainable = False
    else:  # ResNet50
        for layer in base_model.layers[:-30]:  # Unfreeze last 30 layers of ResNet50
            layer.trainable = False

    # Recompile with a lower learning rate
    optimizer = Adam(learning_rate=learning_rate / 10)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Continue training
    remaining_epochs = epochs - min(10, epochs // 2)
    if remaining_epochs > 0:
        history_fine = model.fit(
            train_generator,
            epochs=remaining_epochs,
            validation_data=validation_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks_list,
            verbose=1
        )

        # Combine histories
        history = {
            'accuracy': history_initial.history['accuracy'] + history_fine.history['accuracy'],
            'val_accuracy': history_initial.history['val_accuracy'] + history_fine.history['val_accuracy'],
            'loss': history_initial.history['loss'] + history_fine.history['loss'],
            'val_loss': history_initial.history['val_loss'] + history_fine.history['val_loss']
        }
    else:
        history = history_initial.history

    # Step 12: Save the trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"shoe_damage_model_{timestamp}.h5")
    model.save(model_path)
    print(f"\nModel saved as {model_path}")
    
    # Also save a copy to the standard location
    standard_path = r"E:\UM-Projects\ml\models\shoe_damage_model.h5"
    model.save(standard_path)
    print(f"Model also saved as {standard_path}")

    # Step 13: Plot training and validation accuracy and loss
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save the plots
    plot_path = os.path.join(output_dir, f"training_history_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()
    print(f"Training plots saved to {plot_path}")
    
    # Step 14: Evaluate the model on the validation set
    print("\nEvaluating model on validation data...")
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="image_path",
        y_col="damage_type",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=damage_types,
        shuffle=False  # Don't shuffle for evaluation
    )
    
    evaluation = model.evaluate(val_generator)
    print(f"Validation Loss: {evaluation[0]:.4f}")
    print(f"Validation Accuracy: {evaluation[1]:.4f}")
    
    return model, history

def train_custom_model(model, train_generator, validation_generator, epochs, callbacks_list, 
                      learning_rate, output_dir, val_df, damage_types, image_size, batch_size):
    """Helper function to train the custom model without transfer learning"""
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        steps_per_epoch=len(train_generator),
        validation_steps=len(validation_generator),
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Step 12: Save the trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"shoe_damage_model_{timestamp}.h5")
    model.save(model_path)
    print(f"\nModel saved as {model_path}")
    
    # Also save a copy to the standard location
    standard_path = r"E:\UM-Projects\ml\models\shoe_damage_model.h5"
    model.save(standard_path)
    print(f"Model also saved as {standard_path}")

    # Step 13: Plot training and validation accuracy and loss
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save the plots
    plot_path = os.path.join(output_dir, f"training_history_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()
    print(f"Training plots saved to {plot_path}")
    
    # Step 14: Evaluate the model on the validation set
    print("\nEvaluating model on validation data...")
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="image_path",
        y_col="damage_type",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=damage_types,
        shuffle=False  # Don't shuffle for evaluation
    )
    
    evaluation = model.evaluate(val_generator)
    print(f"Validation Loss: {evaluation[0]:.4f}")
    print(f"Validation Accuracy: {evaluation[1]:.4f}")
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description='Train a shoe damage classification model')
    parser.add_argument('--csv_path', default=r"E:\UM-Projects\ml\datasets\Damaged_Shoes\metadata\train_metadata.csv",
                        help='Path to CSV file containing image paths and damage types')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                        help='Image size for training (width height)')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    parser.add_argument('--output_dir', default=r"E:\UM-Projects\ml\models",
                        help='Directory to save the model and results')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate for the optimizer')
    parser.add_argument('--model_type', choices=['custom', 'mobilenet', 'resnet'], default='custom',
                        help='Type of model to use (custom, mobilenet, or resnet)')
                        
    args = parser.parse_args()
    
    train_model(
        args.csv_path,
        args.epochs,
        args.batch_size,
        tuple(args.image_size),
        args.validation_split,
        args.output_dir,
        args.learning_rate,
        args.model_type
    )

if __name__ == "__main__":
    main()