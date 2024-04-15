import sys 
import os 
from src.exception import CustomException 
from src.logger import logging 
from src.utils import load_obj
import pandas as pd

class PredictPipeline: 
    def __init__(self) -> None:
        pass

    def predict(self, features): 
        try: 
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e: 
            logging.info("Error occured in predict function in prediction_pipeline location")
            raise CustomException(e,sys)
        
class CustomData: 
        def __init__(self, productionyear:float, 
                     cylinders:float, 
                     airbags:float, 
                     manufacturer:str, 
                     model:str, 
                     category:str, 
                     leatherinterior:str, 
                     fueltype:str, 
                     enginevolume:str,
                     mileage:str,
                     gearboxtype:str,
                     drivewheels:str,
                     doors:str,
                     wheel:str,
                     color:str): 
             self.productionyear = productionyear
             self.cylinders = cylinders
             self.airbags = airbags
             self.manufacturer = manufacturer
             self.model = model
             self.category = category
             self.leatherinterior = leatherinterior 
             self.fueltype = fueltype 
             self.enginevolume = enginevolume
             self.mileage = mileage
             self.gearboxtype = gearboxtype
             self.drivewheels = drivewheels
             self.doors = doors
             self.wheel = wheel
             self.color = color
        
        def get_data_as_dataframe(self): 
             try: 
                  custom_data_input_dict = {
                       'productionyear': [self.productionyear], 
                       'cylinders': [self.cylinders], 
                       'airbags': [self.airbags], 
                       'manufacturer': [self.manufacturer],
                       'model':[self.model],
                       'category':[self.category], 
                       'leatherinterior': [self.leatherinterior], 
                       'fueltype': [self.fueltype], 
                       'enginevolume': [self.enginevolume],
                       'mileage': [self.mileage],
                       'gearboxtype': [self.gearboxtype],
                       'drivewheels': [self.drivewheels],
                       'doors': [self. doors],
                       'wheel': [self.wheel],
                       'color': [self.color]

                  }
                  df = pd.DataFrame(custom_data_input_dict)
                  logging.info("Dataframe created")
                  return df
             except Exception as e:
                  logging.info("Error occured in get_data_as_dataframe function in prediction_pipeline")
                  raise CustomException(e,sys) 
             
             
        