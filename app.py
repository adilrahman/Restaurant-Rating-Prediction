from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("Model/model.pkl", "rb"))

#Label encoding 
cuisines_encoder = pickle.load(open("Label Encoders/cuisines_encoder.pkl", "rb"))
listed_in_type_encoder = pickle.load(open("Label Encoders/listed_in_type_encoder.pkl", "rb"))
location_encoder = pickle.load(open("Label Encoders/location_encoder.pkl", "rb"))
rest_type_encoder = pickle.load(open("Label Encoders/type_encoder.pkl", "rb"))


cuisines = {idx : cls for idx, cls in enumerate(cuisines_encoder.classes_)}
listed_in_type = {idx : cls for idx, cls in enumerate(listed_in_type_encoder.classes_)}
location = {idx : cls for idx, cls in enumerate(location_encoder.classes_)}
rest_type = {idx : cls for idx, cls in enumerate(rest_type_encoder.classes_)}

options = {
    "cuisines" : [len(cuisines), cuisines],
    "listed_in_type" : [len(listed_in_type), listed_in_type],
    "location" : [len(location), location],
    "rest_type" : [len(rest_type), rest_type]
}

## normalizer
normalizer = pickle.load(open("Normalizer/normalizer.pkl", "rb"))

@app.route("/", methods = ["GET"])
def home():
    return render_template("home.html", options = options)


@app.route("/predict", methods = ["GET", "POST"])
def predict():
    if request.method == "POST":
         cuisine = request.form["Cuisine"]
         online_order = request.form['online-order']
         votes = request.form["votes"]
         table_booking = request.form['table-booking']
        #  location = request.form["Location"]
         restaurant_type = request.form['Restaurant_Type']
         listed_in_restaurant_type = request.form["listed_in_restaurant_type"]
         approx_cost = request.form['approx-cost']

    
         prediction = dict()
         prediction["online_order"] = [float(online_order)]
         prediction["book_table"] = [float(table_booking)]
         prediction["votes"] = [float(votes)]
        #  prediction["location"] = [float(location)]
         prediction["cuisines"] = [float(cuisine)]
         prediction["approx_cost(for two people)"] = [float(approx_cost)]
         prediction["listed_in(type)"] = [float(listed_in_restaurant_type)]
         prediction["type"] = [float(restaurant_type)]
         
         pred = pd.DataFrame(prediction)
         
         pred = normalizer.transform(pred)
         pred = model.predict(pred)[0]

         return render_template('home.html',options = options, prediction_text=f"Rate prediction : {pred:.2f}  \u2B50")


    return render_template("home.html")