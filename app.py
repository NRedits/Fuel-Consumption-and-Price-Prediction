import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Load the trained model
with open("fuel.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        data = request.form
        input_features = [
            int(data["year"]),
            int(data["make"]),
            int(data["model"]),
            int(data["vehicle_class"]),
            float(data["engine_size"]),
            int(data["cylinders"]),
            int(data["transmission"]),
            int(data["fuel"]),
            float(data["hwy_l_100km"]),
            float(data["comb_l_100km"]),
            int(data["comb_mpg"]),
            float(data["emissions"])
        ]

        # Convert to DataFrame
        input_df = pd.DataFrame([input_features], columns=[
            "YEAR", "MAKE", "MODEL", "VEHICLE CLASS", "ENGINE SIZE", "CYLINDERS",
            "TRANSMISSION", "FUEL", "HWY (L/100 km)", "COMB (L/100 km)", "COMB (mpg)", "EMISSIONS"
        ])

        # Predict fuel consumption
        predicted_consumption = model.predict(input_df)[0]
        predicted_price = predicted_consumption * 100  # Assuming price is 100 times consumption

        return jsonify({
            "predicted_fuel_consumption": f"{predicted_consumption:.2f} liters",
            "predicted_price": f"Rs. {predicted_price:.2f}"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
