import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))

import tensorflow as tf
print(tf.__version__)
