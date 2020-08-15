import os
from flask import Flask, jsonify, request
from flask_restful import Resource ,Api, reqparse
from flask_cors import CORS

import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)
api = Api(app)

app_dir = os.path.dirname(os.path.realpath(__file__))
CNN_model_path = os.path.join(app_dir, "Mnist", "CNN" , "mnist_CNN_model.h5")
DNN_model_path = os.path.join(app_dir, "Mnist", "DNN" , "mnist_DNN_model.h5")

print("Initialising Models")
CNN_Model = tf.keras.models.load_model(CNN_model_path)
DNN_Model = tf.keras.models.load_model(DNN_model_path)
print("Models ready")

class Mnist(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument(
        "network",
        type=str,
        required= True,
        help="You need to define the network"
    )
    parser.add_argument(
        "data",
        action='append',
        required= True,
        help="You need to send data"
    )
    

    def get(self):
        return {"Message" : "This is the MNIST Network"}, 200

    def post(self):   
        reqData = request.get_json()

        if "data" in reqData:
            Data = reqData["data"]
        else:
            Data = None
        
        if Data and isinstance(Data,list) and np.array(Data).shape == (28,28):
            X = np.array(reqData["data"])

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


api.add_resource(Mnist, "/api/ML/mnist")

if __name__ == "__main__":
    app.run()