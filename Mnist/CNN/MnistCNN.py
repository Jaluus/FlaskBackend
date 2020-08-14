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
mnist.to_binary()
mnist.data_augmentation(augment_size=60000)

x_train_splitted, x_val, y_train_splitted, y_val = mnist.get_splitted_train_and_validation_set()
x_test, y_test = mnist.get_test_set()
x_train,y_train = mnist.get_train_set()
num_classes = mnist.num_classes


epochs = 30
batch_size = 128

def model_NN(optimizer,learning_rate,num_classes=num_classes):

    init_w = TruncatedNormal(mean=0.0, stddev=0.01)
    init_b = Constant(value=0.0)


########################################################################################### 
########################## NETZWERK #######################################################
    input_img = Input(shape = x_train.shape[1:])

    x = Conv2D(filters = 32 , kernel_size = 3 , strides = 1 , padding = "same")(input_img)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=(2,2), padding="valid")(x)

    x = Conv2D(filters = 48 , kernel_size = 4 , strides = 1 , padding = "same")(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=(2,2), padding="valid")(x)

    x = Conv2D(filters = 64 , kernel_size = 5 , strides = 1 , padding = "same")(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=(2,2), padding="valid")(x)

    x = Flatten()(x)

    # x = Dense(units = 100 , kernel_initializer =init_w, bias_initializer=tf.initializers.Constant(0))(x)
    # x = Activation("relu")(x)

    x = Dense(units = num_classes, kernel_initializer =init_w, bias_initializer=tf.initializers.Constant(0))(x)
    output = Activation("softmax")(x)

    model = Model(inputs = input_img , outputs = output)

########################################################################################### 
########################## NETZWERK #######################################################

    opti  = optimizer(learning_rate = learning_rate)
    
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

    model = model_NN(Adam,0.0005)   #Using Adam with a learing rate of 0.0005

    model.fit(x= x_train,
            y = y_train,
            epochs = epochs ,
            batch_size = batch_size,
            validation_split = 0.1,
            callbacks = Create_callbacks(),
    )
    model.load_weights(mnist_model_path)
    score1 = model.evaluate(x = x_test, y = y_test)
    print(score1)