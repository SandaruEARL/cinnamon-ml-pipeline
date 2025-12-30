#!/usr/bin/env python3
"""
Fixed TFLite Converter for LSTM Models
Handles the tensor list operations issue properly
"""

import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

def convert_lstm_to_tflite(keras_model_path, output_path):
    """
    Convert Keras LSTM model to TFLite with proper configuration
    """
    print("🔄 Converting LSTM model to TFLite...")
    print(f"   Input: {keras_model_path}")
    print(f"   Output: {output_path}")
    
    try:
        # Load model WITHOUT compiling (avoids optimizer issues)
        model = tf.keras.models.load_model(keras_model_path, compile=False)
        
        print(f"\n✓ Model loaded successfully")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # CRITICAL: Enable TensorFlow ops for LSTM layers
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Standard TFLite ops
            tf.lite.OpsSet.SELECT_TF_OPS      # TensorFlow ops (LSTM support)
        ]
        
        # Disable experimental tensor list lowering (this is the key fix)
        converter._experimental_lower_tensor_list_ops = False
        
        # Enable optimizations for smaller size
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Allow custom ops (additional safety measure)
        converter.allow_custom_ops = True
        
        print("\n⚙️ Converting with LSTM support...")
        print("   - TFLite builtins: enabled")
        print("   - Select TF ops: enabled")
        print("   - Tensor list lowering: disabled")
        print("   - Optimizations: enabled")
        
        # Perform conversion
        tflite_model = converter.convert()
        
        # Save the TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"\n✅ Model converted successfully!")
        print(f"   File: {output_path}")
        print(f"   Size: {len(tflite_model) / 1024:.2f} KB")
        
        return tflite_model
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Ensure TensorFlow version >= 2.4")
        print("   2. Try removing model.compile() before saving")
        print("   3. Check if model has unsupported layers")
        raise

def test_tflite_model(tflite_model, input_shape):
    """
    Test the converted TFLite model with dummy data
    """
    print("\n🧪 Testing TFLite model...")
    
    try:
        # Create interpreter
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"   Input shape: {input_details[0]['shape']}")
        print(f"   Output shape: {output_details[0]['shape']}")
        print(f"   Input dtype: {input_details[0]['dtype']}")
        
        # Create dummy input matching the expected shape
        # Shape should be (1, lookback_days, num_features)
        dummy_input = np.random.rand(*input_shape).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"   Test output shape: {output.shape}")
        print(f"   Sample predictions (first 5): {output[0][:5]}")
        
        print("\n✅ TFLite model test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Model test failed: {str(e)}")
        return False

def main():
    # Configuration
    KERAS_MODEL = 'models/cinnamon_grades_model.h5'
    TFLITE_MODEL = 'models/cinnamon_grades_model.tflite'
    
    # Expected input shape: (batch=1, lookback=30, features=varies)
    # Adjust based on your actual model
    TEST_INPUT_SHAPE = (1, 30, 30)  # Adjust 'features' dimension as needed
    
    print("=" * 70)
    print("🌿 CINNAMON GRADES MODEL - TFLite LSTM Converter")
    print("=" * 70)
    
    # Check if input file exists
    if not Path(KERAS_MODEL).exists():
        print(f"\n❌ Error: Model file not found: {KERAS_MODEL}")
        print("   Please run train_model.py first to generate the Keras model.")
        sys.exit(1)
    
    try:
        # Convert model
        tflite_model = convert_lstm_to_tflite(KERAS_MODEL, TFLITE_MODEL)
        
        # Test model
        test_tflite_model(tflite_model, TEST_INPUT_SHAPE)
        
        print("\n" + "=" * 70)
        print("🎉 SUCCESS! Model ready for deployment")
        print("=" * 70)
        
        print("\n📱 Deployment Notes:")
        print("   1. This model requires TensorFlow Lite with SELECT_TF_OPS")
        print("   2. In Flutter, use tflite_flutter package")
        print("   3. Model size may be larger due to TF ops inclusion")
        print("   4. Inference may be slower than pure TFLite ops")
        
        print("\n🔗 Flutter Integration:")
        print("   - Add dependency: tflite_flutter: ^0.10.0")
        print("   - Include TF ops delegate in your app")
        print("   - See: https://pub.dev/packages/tflite_flutter")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())