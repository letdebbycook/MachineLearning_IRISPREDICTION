from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# =========================
# LOAD MODEL & EVALUATION
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model
model_path = os.path.join(BASE_DIR, "model", "iris_dt.pkl")
model = joblib.load(model_path)

# Global accuracy (optional)
accuracy_path = os.path.join(BASE_DIR, "model", "accuracy.pkl")
model_accuracy = joblib.load(accuracy_path)

# Evaluation result (training & testing)
evaluation_path = os.path.join(BASE_DIR, "model", "evaluation.pkl")
evaluation = joblib.load(evaluation_path)

class_report = evaluation["classification_report"]

# Alternatively, load metrics from a single file
metrics = joblib.load(os.path.join(BASE_DIR, "model", "metrics.pkl"))

accuracy_train = round(metrics["train"] * 100, 2)
accuracy_test = round(metrics["test"] * 100, 2)

# =========================
# IRIS INFO
# =========================
iris_info = {
    "Iris-setosa": {
        "image": "setosa.jpeg",
        "description": (
            "Iris Setosa merupakan spesies iris yang paling mudah dikenali. "
            "Memiliki kelopak kecil, bunga berwarna cerah, dan biasanya tumbuh "
            "di daerah beriklim dingin."
        )
    },
    "Iris-versicolor": {
        "image": "versicolor.jpeg",
        "description": (
            "Iris Versicolor memiliki ukuran sedang dengan warna bunga ungu kebiruan. "
            "Spesies ini sering ditemukan di daerah lembab dan rawa."
        )
    },
    "Iris-virginica": {
        "image": "virginica.jpeg",
        "description": (
            "Iris Virginica adalah spesies terbesar di antara iris. "
            "Memiliki kelopak panjang dan bunga berwarna ungu tua hingga biru."
        )
    }
}

# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image = None
    description = None

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

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
