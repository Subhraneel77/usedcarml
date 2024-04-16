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
            manufacturer = request.form.get("manufacturer"),
            model = request.form.get("model"),
            category = request.form.get("category"), 
            leatherinterior= request.form.get("leatherinterior"), 
            fueltype = request.form.get("fueltype"),
            enginevolume = request.form.get("enginevolume"), 
            gearboxtype = request.form.get("gearboxtype"), 
            mileage = request.form.get("mileage"), 
            drivewheels = request.form.get("drivewheels"), 
            doors = request.form.get("doors"),
            wheel = request.form.get("wheel"),
            color = request.form.get("color"),
            productionyear = float(request.form.get('productionyear')), 
            cylinders = float(request.form.get('cylinders')), 
            airbags = float(request.form.get('airbags'))
        )
    new_data = data.get_data_as_dataframe()
    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(new_data)

    results = round(pred[0],2)

    return render_template("results.html", final_result = results)

if __name__ == "__main__": 
    app.run(port=8000, debug= True)

#http://127.0.0.1:8000/ in browser