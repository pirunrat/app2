from joblib import load
import os 
from django.conf import settings

class Model:

    def __init__(self, data, model_path):
        self.data = data
        self.model_path = model_path
        self.model = self.load_model()  # Load the model when initializing the object

    def model_predict(self):
        return self.model.predict(self.data)

    def min_max_scale(self, input_data):
        minVal = 0
        maxVal = 400
        return (input_data - minVal) / (maxVal - minVal)

    def z_score_scale(self, input_data):
        mean = 19.405247616118118
        std = 3.9714218411022917
        return (input_data - mean) / std

    def load_model(self):
        file_path = os.path.join(settings.BASE_DIR, "backend/app2/" + self.model_path)
        return load(file_path)
