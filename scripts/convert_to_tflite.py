#!/usr/bin/env python3
"""
TFLite Conversion Script for LSTM Models
Fixes the TensorListReserve error by using SELECT_TF_OPS and disabling tensor list lowering.
"""

import tensorflow as tf
from tensorflow import keras
import os
import sys

def convert_to_tflite(model_path, output_path):
    """
    Convert Keras model to TFLite format with LSTM support.
    
    Args:
        model_path: Path to the saved Keras model (.h5)
        output_path: Path where the TFLite model will be saved (.tflite)
    """
    print(f"📱 Converting {model_path} to TFLite...")
    
    # Load the Keras model
    try:
        model = keras.models.load_model(model_path)
        print(f"✓ Loaded Keras model from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        sys.exit(1)
    
    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # CRITICAL FIX for LSTM models:
    # Use SELECT_TF_OPS to support TensorFlow operations in TFLite
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable built-in TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS      # Enable select TensorFlow ops (needed for LSTM)
    ]
    
    # Disable experimental tensor list ops lowering
    # This prevents the TensorListReserve error
    converter._experimental_lower_tensor_list_ops = False
    
    # Convert the model
    try:
        tflite_model = converter.convert()
        print("✓ Model converted successfully")
    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        sys.exit(1)
    
    # Save the TFLite model
    try:
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get file size
        size_bytes = os.path.getsize(output_path)
        size_kb = size_bytes / 1024
        size_mb = size_kb / 1024
        
        print(f"✅ TFLite model saved: {output_path}")
        print(f"   Size: {size_kb:.2f} KB ({size_mb:.2f} MB)")
        
    except Exception as e:
        print(f"❌ Failed to save model: {str(e)}")
        sys.exit(1)
    
    return tflite_model


def main():
    """Main execution function"""
    
    # Define paths
    keras_model_path = "models/cinnamon_grades_model.h5"
    tflite_output_path = "cinnamon_grades_model.tflite"
    
    # Check if Keras model exists
    if not os.path.exists(keras_model_path):
        print(f"❌ Keras model not found: {keras_model_path}")
        print("   Please train the model first using train_model.py")
        sys.exit(1)
    
    # Convert the model
    convert_to_tflite(keras_model_path, tflite_output_path)
    
    print("\n" + "="*50)
    print("🎉 TFLite conversion completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()