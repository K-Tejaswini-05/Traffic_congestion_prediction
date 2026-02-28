from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import tensorflow as tf

app = Flask(__name__)

# Load trained model (.keras format)
model = tf.keras.models.load_model("traffic_model.keras")

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

sequence_length = 10
labels = ["Low Traffic", "Medium Traffic", "High Traffic"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert inputs to float
        input_data = np.array([[
            float(data["temp"]),
            float(data["rain"]),
            float(data["snow"]),
            float(data["clouds"]),
            float(data["hour"]),
            float(data["day"]),
            float(data["month"])
        ]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Create LSTM sequence (repeat same row 10 times)
        input_seq = np.repeat(input_scaled, sequence_length, axis=0)
        input_seq = input_seq.reshape(1, sequence_length, input_scaled.shape[1])

        # Predict
        prediction = model.predict(input_seq, verbose=0)
        traffic_class = np.argmax(prediction)

        return jsonify({
            "prediction": labels[traffic_class]
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)