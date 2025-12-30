import tensorflow as tf
import numpy as np

print("🔄 Converting LSTM model to TFLite...")

# Load model WITHOUT compiling
model = tf.keras.models.load_model('cinnamon_grades_model.h5', compile=False)

print(f"✓ Model loaded successfully")
print(f"  Input shape: {model.input_shape}")
print(f"  Output shape: {model.output_shape}")

# Convert to TFLite with LSTM support
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# CRITICAL: Enable TensorFlow ops for LSTM layers
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Standard TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS      # TensorFlow ops (needed for LSTM)
]

# Disable tensor list ops lowering (fixes LSTM conversion)
converter._experimental_lower_tensor_list_ops = False

# Enable optimizations for smaller size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("\n⚙️ Converting with LSTM support...")
tflite_model = converter.convert()

# Save the TFLite model
with open('cinnamon_grades_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"\n✅ Model converted to TFLite!")
print(f"   File: cinnamon_grades_model.tflite")
print(f"   Size: {len(tflite_model) / 1024:.2f} KB")

# Test the TFLite model
print("\n🧪 Testing TFLite model...")

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"   Input shape: {input_details[0]['shape']}")
print(f"   Output shape: {output_details[0]['shape']}")

# Test with dummy data (30x30 input based on your model)
dummy_input = np.random.rand(1, 30, 30).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print(f"   Test output shape: {output.shape}")
print(f"   Sample predictions: {output[0][:5]}")

print("\n🎉 Success! LSTM model is ready for deployment!")
print("\n⚠️  NOTE: This model requires TensorFlow Lite with SELECT_TF_OPS support.")
print("   In Flutter, you'll need to use tflite_flutter with TF ops enabled.")
