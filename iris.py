from flask import Flask, render_template, request
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# LOAD MODEL
model = joblib.load(os.path.join(BASE_DIR, "model", "iris_dt.pkl"))
model_accuracy = joblib.load(os.path.join(BASE_DIR, "model", "accuracy.pkl"))
evaluation = joblib.load(os.path.join(BASE_DIR, "model", "evaluation.pkl"))
metrics = joblib.load(os.path.join(BASE_DIR, "model", "metrics.pkl"))

accuracy_train = round(metrics["train"] * 100, 2)
accuracy_test = round(metrics["test"] * 100, 2)

iris_info = {
    "Iris-setosa": {
        "image": "setosa.jpeg",
        "description": "Iris Setosa merupakan spesies iris yang paling mudah dikenali."
    },
    "Iris-versicolor": {
        "image": "versicolor.jpeg",
        "description": "Iris Versicolor memiliki ukuran sedang."
    },
    "Iris-virginica": {
        "image": "virginica.jpeg",
        "description": "Iris Virginica adalah spesies terbesar."
    }
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = image = description = None

    if request.method == "POST":
        data = np.array([[ 
            float(request.form["sepal_length"]),
            float(request.form["sepal_width"]),
            float(request.form["petal_length"]),
            float(request.form["petal_width"])
        ]])

        prediction = model.predict(data)[0]
        image = iris_info[prediction]["image"]
        description = iris_info[prediction]["description"]

    return render_template(
        "index.html",
        prediction=prediction,
        image=image,
        description=description,
        accuracy_train=accuracy_train,
        accuracy_test=accuracy_test,
        model_accuracy=model_accuracy
    )

app = app
