from joblib import load
import os 
from django.conf import settings
import numpy as np
from .Polynomial_Model import Polynomial

class Model:

    def __init__(self, data, model_path):
        self.data = data
        self.model_path = model_path
        self.model = self.load_model()  
        



    def model_predict(self):
        mileage = self.z_score_scale(self.data['mileage'])

        maxPower = self.min_max_scale_max_power(self.data['max_power'])
        
        year = self.min_max_scale_year(self.data['year'])
        

        data_array = np.array([year,maxPower,mileage]).reshape(1, -1)
       
        return f"The predicted price :{int(np.exp(self.model.predict(data_array))[0])} $"




    def min_max_scale_max_power(self, input_data):
        minVal = 0
        maxVal = 282
        return (input_data - minVal) / (maxVal - minVal)
    



    def min_max_scale_year(self, input_data):
        minVal = 1983
        maxVal = 2020
        return (input_data - minVal) / (maxVal - minVal)



    def z_score_scale(self, input_data):
        mean = 19.405247616118118
        std = 3.9714218411022917
        return (input_data - mean) / std




    def load_model(self):
        file_path = os.path.join(settings.BASE_DIR, "app2/" + self.model_path)
        return load(file_path)
    



    def load_model_poly(self):
        file_path = os.path.join(settings.BASE_DIR, "app2/" + self.model_path)
        return load(file_path)
    
    
