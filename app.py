import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
# load the model
reg_model = pickle.load(open("regmodel.pkl", "rb"))
scalar = pickle.load(open("scaling.pkl", "rb"))
feature = pickle.load(open("feature_main.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    predict_data = feature.transform(np.array(list(new_data.values())).reshape(1, -1))
    print(predict_data)
    output = reg_model.predict(predict_data)
    print(output[0])
    return jsonify(output[0])


@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    final_input_1 = feature.transform(np.array(final_input).reshape(1, -1))
    print(final_input_1)
    output = reg_model.predict(final_input_1)[0]
    return render_template(
        "home.html", prediction_text="The House price prediction is {}".format(output)
    )


if __name__ == "__main__":
    app.run(debug=True)
