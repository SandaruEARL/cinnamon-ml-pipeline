#!/usr/bin/env python3
"""
FIXED TFLite Conversion Script
Multiple strategies to convert LSTM models to TFLite
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys

def convert_strategy_1_concrete_function(model, output_path):
    """
    Strategy 1: Use concrete function with fixed batch size
    This often works when standard conversion fails
    """
    print("\n🔄 Strategy 1: Concrete function with fixed shape...")
    
    try:
        # Get model's input shape (excluding batch dimension)
        input_shape = model.input_shape[1:]  # e.g., (30, 30)
        
        # Create a concrete function with batch size = 1
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1] + list(input_shape), dtype=tf.float32)
        ])
        def model_fn(x):
            return model(x, training=False)
        
        concrete_func = model_fn.get_concrete_function()
        
        # Convert using concrete function
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        
        # Try with ONLY built-in ops first
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter._experimental_lower_tensor_list_ops = True
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_kb = len(tflite_model) / 1024
        print(f"✅ SUCCESS with Strategy 1!")
        print(f"   File: {output_path}")
        print(f"   Size: {size_kb:.2f} KB")
        print(f"   ✓ No Flex ops needed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy 1 failed: {str(e)[:150]}")
        return False


def convert_strategy_2_unroll_lstm(model, output_path):
    """
    Strategy 2: Rebuild model with unrolled LSTM
    This forces static graph execution
    """
    print("\n🔄 Strategy 2: Unrolled LSTM (static execution)...")
    
    try:
        # Clone model architecture
        config = model.get_config()
        
        # Modify LSTM layers to use unroll=True
        for i, layer_config in enumerate(config['layers']):
            if layer_config['class_name'] == 'LSTM':
                layer_config['config']['unroll'] = True
                print(f"   ✓ Set unroll=True for LSTM layer {i}")
        
        # Rebuild model
        new_model = keras.Model.from_config(config)
        new_model.set_weights(model.get_weights())
        
        # Convert
        converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter._experimental_lower_tensor_list_ops = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_kb = len(tflite_model) / 1024
        print(f"✅ SUCCESS with Strategy 2!")
        print(f"   File: {output_path}")
        print(f"   Size: {size_kb:.2f} KB")
        print(f"   ✓ No Flex ops needed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy 2 failed: {str(e)[:150]}")
        return False


def convert_strategy_3_savedmodel(model, output_path):
    """
    Strategy 3: Convert via SavedModel format
    Sometimes this path has better LSTM support
    """
    print("\n🔄 Strategy 3: Via SavedModel format...")
    
    try:
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save as SavedModel
            saved_model_dir = os.path.join(tmpdir, 'saved_model')
            tf.saved_model.save(model, saved_model_dir)
            
            # Convert from SavedModel
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            converter._experimental_lower_tensor_list_ops = True
            
            tflite_model = converter.convert()
            
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            size_kb = len(tflite_model) / 1024
            print(f"✅ SUCCESS with Strategy 3!")
            print(f"   File: {output_path}")
            print(f"   Size: {size_kb:.2f} KB")
            print(f"   ✓ No Flex ops needed!")
            
            return True
            
    except Exception as e:
        print(f"❌ Strategy 3 failed: {str(e)[:150]}")
        return False


def convert_strategy_4_flex_ops(model, output_path):
    """
    Strategy 4: Accept Flex ops (fallback)
    Works reliably but increases app size
    """
    print("\n🔄 Strategy 4: Using Flex delegate (fallback)...")
    
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable both built-in and TensorFlow ops
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        # Disable tensor list lowering (required for Flex)
        converter._experimental_lower_tensor_list_ops = False
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_kb = len(tflite_model) / 1024
        print(f"✅ SUCCESS with Strategy 4!")
        print(f"   File: {output_path}")
        print(f"   Size: {size_kb:.2f} KB")
        print(f"   ⚠️  Requires Flex delegate in Flutter app")
        print(f"   ⚠️  Add to android/app/build.gradle:")
        print(f"       implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:2.11.0'")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy 4 failed: {str(e)[:150]}")
        return False


def test_tflite_model(tflite_path, input_shape):
    """Test the converted TFLite model"""
    print(f"\n🧪 Testing TFLite model...")
    
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"   ✓ Input shape: {input_details[0]['shape']}")
        print(f"   ✓ Output shape: {output_details[0]['shape']}")
        
        # Create test input
        test_input = np.random.random(input_details[0]['shape']).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"   ✓ Test inference successful!")
        print(f"   ✓ Output sample: {output[0][:5]}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def convert_to_tflite_multi_strategy(model_path, output_path):
    """
    Try multiple conversion strategies in order of preference:
    1. Concrete function (no Flex, optimized)
    2. Unrolled LSTM (no Flex, static)
    3. SavedModel path (no Flex, alternative)
    4. Flex delegate (works but larger)
    """
    print("="*70)
    print("🔧 TFLite Multi-Strategy Converter")
    print("="*70)
    
    # Load model
    print(f"\n📂 Loading model: {model_path}")
    try:
        model = keras.models.load_model(model_path)
        print(f"   ✓ Model loaded successfully")
        print(f"   ✓ Input shape: {model.input_shape}")
        print(f"   ✓ Output shape: {model.output_shape}")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return False
    
    # Try strategies in order
    strategies = [
        ("Concrete Function (Best)", convert_strategy_1_concrete_function),
        ("Unrolled LSTM", convert_strategy_2_unroll_lstm),
        ("SavedModel Path", convert_strategy_3_savedmodel),
        ("Flex Delegate (Fallback)", convert_strategy_4_flex_ops),
    ]
    
    for strategy_name, strategy_func in strategies:
        print(f"\n{'='*70}")
        print(f"Trying: {strategy_name}")
        print(f"{'='*70}")
        
        success = strategy_func(model, output_path)
        
        if success:
            # Test the model
            if test_tflite_model(output_path, model.input_shape):
                print("\n" + "="*70)
                print(f"✅ CONVERSION SUCCESSFUL using {strategy_name}!")
                print("="*70)
                return True
            else:
                print(f"⚠️  Model converted but failed testing, trying next strategy...")
    
    # All strategies failed
    print("\n" + "="*70)
    print("❌ ALL CONVERSION STRATEGIES FAILED")
    print("="*70)
    print("\nPossible solutions:")
    print("1. Check your TensorFlow version (recommend 2.11+)")
    print("2. Try simplifying the model architecture")
    print("3. Use the .h5 model directly (not TFLite)")
    print("4. Contact support with the error logs above")
    
    return False


def main():
    """Main execution"""
    
    # Paths
    model_path = "models/cinnamon_grades_model.h5"
    output_path = "models/cinnamon_grades_model.tflite"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Please train the model first using the training script")
        sys.exit(1)
    
    # Convert
    success = convert_to_tflite_multi_strategy(model_path, output_path)
    
    if success:
        print("\n🎉 You can now use the TFLite model in your Flutter app!")
        sys.exit(0)
    else:
        print("\n⚠️  TFLite conversion failed, but you can still use the .h5 model")
        sys.exit(1)


if __name__ == "__main__":
    main()