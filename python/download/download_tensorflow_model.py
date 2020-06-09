import tensorflow as tf
import numpy as np
import os
models = {
    'resnet50': tf.keras.applications.ResNet50,
    'mobilenet': tf.keras.applications.MobileNet,
    'mobilenetv2': tf.keras.applications.MobileNetV2,
}

if __name__ == "__main__":

    for i in models:
        # Export the model to a SavedModel(tf)
        model = models.get(i)()
        model.save("models/{}".format(i), save_format='tf')

        input_data = np.array(np.random.random_sample([1, 224, 224, 3]), dtype=np.float32)

        # tensorflow inference
        tf_output = model.predict(input_data)

        #  Convert the model to tflite
        converter = tf.lite.TFLiteConverter.from_saved_model("models/{}".format(i))
        tflite_model = converter.convert()
        os.makedirs("tflite_models", exist_ok=True)
        open("tflite_models/{}.tflite".format(i), "wb").write(tflite_model)

        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path="tflite_models/{}.tflite".format(i))
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
        print("{} PASS".format(i))