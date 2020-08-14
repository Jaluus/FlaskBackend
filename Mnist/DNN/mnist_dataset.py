import os
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def call_data_flat():

    #Daten Einlesen
    (x_train,y_train),(x_test,y_test) = mnist.load_data()

    train_size, width, height = x_train.shape
    num_features = width*height
    num_classes = 10 # Null bis Neun

    #Onehot vektor erstellen
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    x_train = x_train.reshape(-1, num_features)
    x_test = x_test.reshape(-1, num_features)

    return (x_train,y_train),(x_test,y_test), num_features, num_classes

class MNIST:
    def __init__(self):
        # Daten Einlesen
        (self.x_train,self.y_train),(self.x_test,self.y_test) = mnist.load_data()
        self.x_val = None
        self.x_train_splitted = None
        self.y_val = None
        self.y_train_splitted = None
        self.val_size = None
        self.train_splitted_size = None
        # Dim erweitern
        self.x_train = np.expand_dims(self.x_train,axis=-1)
        self.x_test = np.expand_dims(self.x_test,axis=-1)

        self.train_size, self.width, self.height, self.depth = self.x_train.shape
        self.num_features = self.width*self.height*self.depth
        self.num_classes = 10 # Null bis Neun
        # OneHot
        self.y_train = to_categorical(self.y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(self.y_test, num_classes=self.num_classes)

    def get_test_set(self):
        return (self.x_test,self.y_test)

    def get_train_set(self):
        return (self.x_train,self.y_train)

    def get_splitted_train_and_validation_set(self,val_size = 0.10):
        self.x_train_splitted , self.x_val , self.y_train_splitted , self.y_val = train_test_split(self.x_train,self.y_train ,test_size = val_size, shuffle = True)
        self.val_size = self.x_val[0]
        self.train_splitted_size = self.x_train_splitted[0]
        return self.x_train_splitted , self.x_val , self.y_train_splitted , self.y_val

    def scale_data(self, scaling_cls):
        self.scaler = scaling_cls()

        self.x_train = self.x_train.reshape((-1,self.num_features))
        self.x_test = self.x_test.reshape((-1,self.num_features))

        self.scaler.fit(self.x_train) # Nur die Train daten, da wir anhand dieser trainierung und die Test daten daran anpassen
        self.x_train = self.scaler.transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

        self.x_train = self.x_train.reshape((-1, self.width, self.height, self.depth))
        self.x_test = self.x_test.reshape((-1, self.width, self.height, self.depth))

    def to_binary(self):

        self.x_train = self.x_train.reshape((-1,self.num_features))
        self.x_test = self.x_test.reshape((-1,self.num_features))
        
        for pic in self.x_train:
            pic = np.array(pic//120, dtype=bool)
            pic = np.array(pic, dtype=int)

        for pic in self.x_test:
            pic = np.array(pic//120, dtype=bool)
            pic = np.array(pic, dtype=int)


        self.x_train = self.x_train.reshape((-1, self.width, self.height, self.depth))
        self.x_test = self.x_test.reshape((-1, self.width, self.height, self.depth))


    def data_augmentation(self, augment_size = 10000):
        # Create instance of generator Class
        image_generator = ImageDataGenerator(
            rotation_range = 10, #15 grad drehung random
            zoom_range = 0.1,
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            fill_mode = "constant",
            cval = 0.0
        )
        # augment data
        image_generator.fit(self.x_train,augment=True)
        # get train data
        rand_idxs = np.random.randint(self.train_size , size = augment_size)
        x_augmented = self.x_train[rand_idxs].copy()    #Damit man nicht nur einen daten verweiß hat
        y_augmented = self.y_train[rand_idxs].copy()
        x_augmented = image_generator.flow(x_augmented, batch_size = augment_size, shuffle=False).next()
        #Append new images
        self.x_train = np.concatenate((self.x_train,x_augmented))
        self.y_train = np.concatenate((self.y_train,y_augmented))
        self.train_size = self.x_train.shape[0]

if __name__ == "__main__":
    mnist = MNIST()
    print("start")
    mnist.to_binary()
    print("finish")