from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
IMAGE_DIR = os.path.join(FILE_DIR,"images")
MODEL_DIR = os.path.join(FILE_DIR,"models")

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 10    #Change it later to 100
        self.sample_noise = np.random.normal(0, 1, (25,self.latent_dim))
        optimizer = Adam(0.0002, 0.5)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        self.generator = self.build_generator()
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        validity = self.discriminator(img)
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        noise = Input(shape = (self.latent_dim,))

        x = Dense(256)(noise)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha = 0.2)(x)

        x = Dense(512)(noise)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha = 0.2)(x)

        x = Dense(1024)(noise)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha = 0.2)(x)

        x = Dense(np.prod(self.img_shape), activation='tanh')(x)
        output = Reshape(self.img_shape)(x)

        model = Model(inputs = noise , outputs = output)
        return model

    def build_discriminator(self):

        img = Input(shape = self.img_shape)
        x = Flatten()(img)
        x = Dense(512)(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = Dense(256)(x)
        x = LeakyReLU(alpha = 0.2)(x)
        validity = Dense(1, activation='sigmoid')(x)

        model = Model(inputs = img,outputs = validity)

        return model

    def train(self, epochs, batch_size=128, sample_interval=50):
        print("Start Training")
        print("Start Loading Data ")
        (X_train, _), (_, _) = mnist.load_data()
        print("Loading Data Complete")
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)
            gen_imgs = self.generator.predict(noise)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        gen_imgs = self.generator.predict(self.sample_noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        self.generator.save(os.path.join(MODEL_DIR,"%d.h5" % epoch))
        fig.savefig(os.path.join(IMAGE_DIR,"%d.png" % epoch))
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=100000, batch_size=132, sample_interval=100)