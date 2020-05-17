import tensorflow as tf

print(tf.__version__)

print(tf.config.experimental.list_physical_devices('GPU'))

assert len(tf.config.experimental.list_physical_devices('GPU')) > 0
print('GPU Detected')