from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

#Load pkl model
with open("penguin_model.pkl", "rb") as f:
    model = pickle.load(f)

#Labeling
species_map = {
    0: "Adelie",
    1: "Chinstrap",
    2: "Gentoo"
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        #Form inputs
        bill_depth = float(request.form["bill_depth"])
        island = request.form["island"]

        #One-hot encode island (default)
        island_Dream = 1 if island == "Dream" else 0
        island_Torgersen = 1 if island == "Torgersen" else 0

        #Feature vector in correct order
        features = np.array([[bill_depth, island_Dream, island_Torgersen]])

        #Prediction
        pred_label = model.predict(features)[0]
        prediction = species_map[pred_label]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
