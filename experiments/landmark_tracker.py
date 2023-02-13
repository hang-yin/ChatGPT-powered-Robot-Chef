import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="hand_landmark_lite.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
print(input_shape)

# load a png image to a numpy array of size (1, 224, 224, 3)
# and convert it to a float32 tensor
img = tf.io.read_file('hand_example.png')
img = tf.image.decode_png(img, channels=3)
img = tf.image.resize(img, (224, 224))
img = tf.cast(img, tf.float32)
img = img[tf.newaxis, :]
print(img.shape)

# run the model
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

# get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
# reshape the output tensor to (21, 3)
output_data = np.reshape(output_data, (21, 3))
print(output_data)

# plot the 21 3D landmarks
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(output_data[:, 0], output_data[:, 1], output_data[:, 2])
plt.show()


"""
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
"""