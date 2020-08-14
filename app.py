from flask import Flask, jsonify, request
from flask_restful import Resource ,Api, reqparse
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)


class Mnist(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument("data",
        type=float,
        required= True,
        help="You need to send Data"
    )

    def get(self):
        return {"Message" : "This is the MNIST Network"}, 200

    def post(self):   
        reqData = Mnist.parser.parse_args()   
        resData = {
            "Number" : "Not yet implemented"
        }
        return resData, 201



api.add_resource(Mnist, "/api/ML/mnist")

if __name__ == "__main__":
    app.run()