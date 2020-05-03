import tensorflow as tf
import numpy as np

# Export the model to a SavedModel(tf)
resnet50 = tf.keras.applications.ResNet50()
resnet50.save('resnet50', save_format='tf')

input_data = np.array(np.random.random_sample([1, 224, 224, 3]), dtype=np.float32)

# tensorflow inference
tf_output = resnet50.predict(input_data)

#  Convert the model to tflite
converter = tf.lite.TFLiteConverter.from_saved_model('resnet50')
tflite_resnet50 = converter.convert()
open("resnet50.tflite", "wb").write(tflite_resnet50)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="resnet50.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

tf_lite_output = interpreter.get_tensor(output_details[0]['index'])


# compare  results
np.testing.assert_allclose(tf_output, tf_lite_output, rtol=1e-03, atol=1e-05)
print("PASS")