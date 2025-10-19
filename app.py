from flask import Flask, request,jsonify
import numpy as np
import tensorflow as tf
model=tf.keras.models.load_model("iris_ann_model.h5")
app=Flask("Iris API")
@app.route("/")
def home():
    return "API Iris est opérationnelle"

@app.route("/ping",methods=["get"])
def ping():
    return jsonify({"message":"ok"})

@app.route("/predict",methods=["post"])
def predict():
    data=request.get_json()
    features=data["features"]
    features=np.array(features).reshape(1,-1)
    prediction=model.predict(features)
    predicted_class=int(np.argmax(prediction,axis=1)[0])
    return jsonify({"predicted_class":predicted_class})
def print_hi(name):
# Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}') # Press Ctrl+F8 to toggle the breakpoint.
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)