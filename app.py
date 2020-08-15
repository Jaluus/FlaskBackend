import os
from flask import Flask, jsonify, request
from flask_restful import Resource ,Api, reqparse
from flask_cors import CORS

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)
api = Api(app)

app_dir = os.path.dirname(os.path.realpath(__file__))
CNN_model_path = os.path.join(app_dir, "Mnist", "CNN" , "mnist_CNN_model.h5")
DNN_model_path = os.path.join(app_dir, "Mnist", "DNN" , "mnist_DNN_model.h5")
GAN_model_path = os.path.join(app_dir, "Mnist", "GAN" , "mnist_GAN_model.h5")

print("Initialising Models")
CNN_Model = tf.keras.models.load_model(CNN_model_path)
DNN_Model = tf.keras.models.load_model(DNN_model_path)
GAN_Model = tf.keras.models.load_model(GAN_model_path, compile=False)
print("Models ready")

class MnistClassifier(Resource):
    def get(self):
        return {"Message" : "This is the MNIST Classifier Network"}, 200

    def post(self):   
        reqData = request.get_json()

        if "data" in reqData:
            Data = reqData["data"]
        else:
            Data = None
        
        if Data and isinstance(Data,list) and np.array(Data).shape == (28,28):
            X = np.array(Data)

            # Expanding dims from (rows,cols) to (samples, rows , cols , channels)
            X_CNN = np.expand_dims(np.expand_dims(X,axis=-1),axis=0)

            # Flatten the Array for the DNN
            X_DNN = np.expand_dims(X.flatten(),axis=0)
            
            #  Predicting
            pred_vector_CNN = (np.array(CNN_Model.predict(X_CNN)))
            pred_vector_DNN = np.array(DNN_Model.predict(X_DNN))

            resData = {
                "message" : "Prediction complete",
                "pred_vector_CNN" : pred_vector_CNN.tolist(),
                "pred_vector_DNN" : pred_vector_DNN.tolist()
            }
            return resData, 200
        return { "message" : "You need to pass 'data' as an array in the shape of (28,28) of with all entries need to be Ints or Floats between 0 and 255"}, 400

class MnistGenerator(Resource):
    def get(self):
        return {"Message" : "This is the MNIST Generator Network"}, 200

    def post(self):   
        reqData = request.get_json()
        minVal = 100
        maxVal = -100

        if "latentVector" in reqData:
            LV = reqData["latentVector"]
        else:
            LV = None
        
        if LV and isinstance(LV,list) and np.array(LV).shape == (10,):
            LV = np.array(LV)

            # Expanding dims from (rows,cols) to (samples, rows , cols , channels)
            LV = np.expand_dims(LV,axis=0)
            # Generating
            GenImg = np.array(GAN_Model.predict(LV))
            GenImg = np.reshape(GenImg,(28,28))

            # get the values between 0 and 1
            maxVal = np.amax(GenImg)
            minVal = np.amin(GenImg)
            deltaValue = maxVal - minVal
            GenImg = (GenImg - minVal) / deltaValue

            resData = {
                "message" : "Generation complete",
                "GANImage" : GenImg.tolist()
            }
            return resData, 200

        return { "message" : "You need to pass 'latentVector' as an array in the shape of (10) of with all entries need to be Floats between 0 and 1"}, 400

api.add_resource(MnistClassifier, "/api/ML/mnist/classifier")
api.add_resource(MnistGenerator, "/api/ML/mnist/generator")

if __name__ == "__main__":
    app.run()