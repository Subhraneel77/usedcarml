from flask import Flask, request, render_template, jsonify
# Alternatively can use Django, FastAPI, or anything similar
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')
@app.route('/predict', methods = ['POST', "GET"])

def predict_datapoint(): 
    if request.method == "GET": 
        return render_template("form.html")
    else: 
        data = CustomData(
            manufacturer = float(request.form.get("manufacturer")),
            model = float(request.form.get("model")),
            category = float(request.form.get("category")), 
            leatherinterior= float(request.form.get("leatherinterior")), 
            fueltype = float(request.form.get("fueltype")),
            enginevolume = float(request.form.get("enginevolume")), 
            gearboxtype = float(request.form.get("gearboxtype")), 
            mileage = float(request.form.get("mileage")), 
            drivewheels = float(request.form.get("drivewheels")), 
            doors = float(request.form.get("doors")),
            wheel = float(request.form.get("wheel")),
            color = float(request.form.get("color")),
            productionyear = request.form.get('productionyear'), 
            cylinders = request.form.get('cylinders'), 
            airbags = request.form.get('airbags')
        )
    new_data = data.get_data_as_dataframe()
    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(new_data)

    results = round(pred[0],2)

    return render_template("results.html", final_result = results)

if __name__ == "__main__": 
    app.run(port=8000, debug= True)

#http://127.0.0.1:8000/ in browser