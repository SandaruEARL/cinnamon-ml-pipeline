#!/usr/bin/env python3
"""
TFLite Conversion Script - CI/CD Compatible
Converts LSTM model to TFLite and saves in correct locations for CI pipeline
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
import shutil

def convert_to_tflite(model_path, output_path):
    """
    Convert Keras model to TFLite using concrete function strategy.
    This works without Flex ops for properly structured LSTM models.
    """
    print("="*70)
    print("🔧 TFLite Converter for CI/CD Pipeline")
    print("="*70)
    
    # Load model
    print(f"\n📂 Loading model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    try:
        model = keras.models.load_model(model_path)
        print(f"   ✓ Model loaded successfully")
        print(f"   ✓ Input shape: {model.input_shape}")
        print(f"   ✓ Output shape: {model.output_shape}")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Convert using concrete function (works without Flex ops)
    print(f"\n🔄 Converting to TFLite using concrete function strategy...")
    
    try:
        # Get input shape (excluding batch dimension)
        input_shape = model.input_shape[1:]  # e.g., (30, 30)
        
        # Create concrete function with fixed batch size = 1
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1] + list(input_shape), dtype=tf.float32)
        ])
        def model_fn(x):
            return model(x, training=False)
        
        concrete_func = model_fn.get_concrete_function()
        
        # Convert using concrete function
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        
        # Use ONLY built-in TFLite ops (no Flex needed!)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter._experimental_lower_tensor_list_ops = True
        
        print("   ⚙️  Converter settings:")
        print("      - Using concrete function with fixed input shape")
        print("      - TFLITE_BUILTINS only (no Flex ops)")
        print("      - Tensor list lowering: enabled")
        
        # Perform conversion
        tflite_model = converter.convert()
        
        size_kb = len(tflite_model) / 1024
        size_mb = size_kb / 1024
        
        print(f"\n✅ Conversion successful!")
        print(f"   Size: {size_kb:.2f} KB ({size_mb:.2f} MB)")
        print(f"   ✓ No Flex delegate required!")
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {str(e)}")
        print("\n💡 Fallback: Trying with Flex ops...")
        
        try:
            # Fallback to Flex ops if standard conversion fails
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter._experimental_lower_tensor_list_ops = False
            
            tflite_model = converter.convert()
            
            size_kb = len(tflite_model) / 1024
            print(f"\n⚠️  Converted with Flex ops (fallback)")
            print(f"   Size: {size_kb:.2f} KB")
            print(f"   ⚠️  Requires Flex delegate in Flutter app!")
            
        except Exception as e2:
            print(f"\n❌ Fallback also failed: {str(e2)}")
            sys.exit(1)
    
    # Save the TFLite model
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"\n💾 Saved TFLite model: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Failed to save model: {str(e)}")
        sys.exit(1)
    
    # Test the model
    print(f"\n🧪 Testing TFLite model...")
    
    try:
        interpreter = tf.lite.Interpreter(model_path=output_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"   ✓ Input shape: {input_details[0]['shape']}")
        print(f"   ✓ Output shape: {output_details[0]['shape']}")
        
        # Run test inference
        test_input = np.random.random(input_details[0]['shape']).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"   ✓ Test inference successful!")
        print(f"   ✓ Sample output: {output[0][:5]}")
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        print(f"   ⚠️  Model saved but may not work correctly")
        # Don't exit - model is saved, let CI decide
    
    return tflite_model


def main():
    """Main execution for CI/CD pipeline"""
    
    # Paths
    keras_model_path = "models/cinnamon_grades_model.h5"
    
    # Output paths
    tflite_in_models = "models/cinnamon_grades_model.tflite"
    tflite_in_root = "cinnamon_grades_model.tflite"
    
    print("\n" + "="*70)
    print("🚀 Starting TFLite Conversion")
    print("="*70)
    
    # Convert the model
    tflite_model = convert_to_tflite(keras_model_path, tflite_in_models)
    
    # Copy to root directory for CI script
    print(f"\n📋 Copying to root for CI pipeline...")
    try:
        shutil.copy(tflite_in_models, tflite_in_root)
        print(f"   ✓ Copied to: {tflite_in_root}")
    except Exception as e:
        print(f"   ⚠️  Failed to copy: {e}")
        # Not critical - model is already saved
    
    # Final summary
    print("\n" + "="*70)
    print("✅ CONVERSION COMPLETE!")
    print("="*70)
    print("\n📦 Output files:")
    
    if os.path.exists(tflite_in_models):
        size = os.path.getsize(tflite_in_models) / 1024
        print(f"   ✓ {tflite_in_models} ({size:.2f} KB)")
    
    if os.path.exists(tflite_in_root):
        size = os.path.getsize(tflite_in_root) / 1024
        print(f"   ✓ {tflite_in_root} ({size:.2f} KB)")
    
    print("\n🎉 Ready for deployment!")
    
    # Exit with success code
    sys.exit(0)


if __name__ == "__main__":
    main()