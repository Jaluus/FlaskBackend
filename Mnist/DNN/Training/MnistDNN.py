import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from mnist_dataset import *


model_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(model_dir ,"tb logs" , str(time.time()))
mnist_model_path = os.path.join(model_dir, "mnist_model.h5")

mnist = MNIST()
mnist.data_augmentation(augment_size=60000)
mnist.to_binary()

(x_train,y_train) = mnist.get_train_set()
(x_test,y_test) = mnist.get_test_set()
x_train = x_train.reshape((-1,784))
x_test = x_test.reshape((-1,784))

num_features = 784
num_classes = 10

epochs = 5
batch_size = 128

def model_NN(optimizer,learning_rate,num_classes=num_classes):

    init_w = TruncatedNormal(mean=0.0, stddev=0.01)
    init_b = Constant(value=0.0)
    opti  = optimizer(learning_rate = learning_rate)


########################################################################################### 
########################## NETZWERK #######################################################
    model = Sequential()
    model.add(Dense(100, activation="relu", input_shape = (num_features,)))
    model.add(Dense(250, activation="relu"))
    model.add(Dense(10, activation="softmax"))
########################################################################################### 
########################## NETZWERK #######################################################
    
    model.compile(
        loss = "categorical_crossentropy",  #categorical_crossentropy / mse
        optimizer = opti,
        metrics = ["accuracy"]
        )
    
    return model

def Create_callbacks():

    tb = tf.keras.callbacks.TensorBoard(
        log_dir = log_dir,
        histogram_freq=1
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        mnist_model_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max',
        save_freq="epoch"
    )

    return [tb,checkpoint]


if __name__ == "__main__":

    model = model_NN(Adam,0.0005)

    model.fit(x= x_train,
            y = y_train,
            epochs = epochs ,
            batch_size = batch_size,
            validation_data=(x_test,y_test),
            #callbacks = Create_callbacks(),
    )
    model.save(mnist_model_path)
    model.load_weights(mnist_model_path)
    score1 = model.evaluate(x = x_test, y = y_test)
    print(score1)