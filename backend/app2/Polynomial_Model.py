from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from joblib import load, dump

class Polynomial:
    def __init__(self, degree=2,n_splits=3, init_method='xavier', method='batch', lr=0.001, num_epochs=500, batch_size=10,momentum=None):
        self.degree = degree
        self.init_method = init_method
        self.method = method
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.cv = KFold(n_splits=n_splits)
        self.momentum = momentum
        self.theta = None  # Initialize theta here
        self.theta_list = []
        

    def min_max_scale_max_power(self, input_data):
        minVal = 0
        maxVal = 282
        return (input_data - minVal) / (maxVal - minVal)
    


    def min_max_scale_year(self, input_data):
        minVal = 1983
        maxVal = 2020
        return (input_data - minVal) / (maxVal - minVal)




    def z_score_scale(self, input_data):
        mean = 19.414276674283705
        std = 3.965679176460303
        return (input_data - mean) / std
    



    def predict(self,weight,input):
        mileage = self.z_score_scale(input[2])
        max_power = self.min_max_scale_max_power(input[1])
        year = self.min_max_scale_year(input[0])
        prepared_input = np.array([year,max_power,mileage]).reshape(1,3)

        X_poly = self._polynomial_features(prepared_input)
        
        return f"The predicted price :{int(np.exp(X_poly @ weight.reshape(6,1)))} $"




    def _polynomial_features(self, X):
        n = X.shape[1]

        # List to store the polynomial features
        features = []

        for j in range(n):
            for d in range(1, self.degree + 1):
                features.append(X[:, j:j+1]**d)

        # Stack them horizontally to form the output matrix
        X_poly = np.hstack(features)

        return X_poly
    



    


