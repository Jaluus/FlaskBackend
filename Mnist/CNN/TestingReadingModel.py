import os
from mnist_dataset import *
import numpy as np
import tensorflow as tf

model_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(model_dir ,"tb logs" , str(time.time()))
mnist_model_path = os.path.join(model_dir, "mnist_CNN_model.h5")

mnist = MNIST()
mnist.to_binary()
mnist.data_augmentation(augment_size=60000)

x_test, y_test = mnist.get_test_set()

model = tf.keras.models.load_model(mnist_model_path)
Array = np.expand_dims(x_test[0],axis=0) # Cause the model expectes (samples, rows,cols,channels) and not just (rows,cols,channels)
pred = model.predict(Array)
print(pred)