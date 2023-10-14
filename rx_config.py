import tensorflow as tf


def init_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # enable "Eager execution" for GRU usage and improve performance by GPU optimization
            tf.compat.v1.enable_eager_execution()
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as e:
            print(e)

# References:
# Limiting GPU memory usage by Keras, https://stackoverflow.com/a/57992246
