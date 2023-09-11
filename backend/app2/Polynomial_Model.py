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
        
    def r2(self, y_true, y_pred):
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        r2_score = 1 - (ss_res/ss_tot)
        return r2_score

    def mse(self, y_true, y_pred):
        if np.isscalar(y_true):
            return (y_pred - y_true) ** 2
        return ((y_pred - y_true) ** 2).sum() / y_true.shape[0]

    def _train(self, X, y):
        yhat = self.predict(X)
        X_poly = self._polynomial_features(X)
        m = X.shape[0]
        grad = (1/m) * X_poly.T @ (yhat - y)
        
        if self.momentum:
            if not hasattr(self, 'momentum_velocity'):
                self.momentum_velocity = np.zeros_like(self.theta)
            self.momentum_velocity = self.momentum * self.momentum_velocity - self.lr * grad
            self.theta += self.momentum_velocity
        else:
            self.theta -= self.lr * grad

    
        return self.mse(y, yhat)

    
    def fit(self, X_train, y_train):
        
        #create a list of kfold scores
        self.kfold_scores = list()
        
        #X_poly = self._polynomial_features(X_train)
        
        
        #print(X_poly)
        
       

        # Add bias term (intercept) to X_poly
        #reset val loss
        self.val_loss_old = np.infty

        
        
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            
            
           
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val = X_train[val_idx]
            y_cross_val = y_train[val_idx]    
           

            # Initialization of theta
            if self.init_method == 'zeros':
                
                self.theta = np.zeros(X_train.shape[1] * self.degree)
                
            elif self.init_method == 'xavier':
                
                m = X_train.shape[1] * self.degree

                 # calculate the range for the weights
                lower, upper = -(1.0 / np.sqrt(m)), (1.0 / np.sqrt(m))


                 # you need to basically randomly pick weights within this range
                 # generate random numbers
                numbers = np.random.rand(1000)
                scaled = lower + numbers * (upper - lower)

                # Randomly pick a number from scaled
                self.theta = np.random.choice(scaled,size=m)
                
                #variance = 2.0 / (X_poly.shape[1] + 1)  # +1 for bias term
                #self.theta = np.random.randn(X_poly.shape[1]) * np.sqrt(variance)
            else:
                raise ValueError("Unknown init_method: {}".format(self.init_method))
                
                
                #one epoch will exhaust the WHOLE training set
            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                
                params = {"method": self.method, "lr": self.lr, "reg": type(self).__name__}
                mlflow.log_params(params=params)
                
                for epoch in range(self.num_epochs):
                
                    #with replacement or no replacement
                    #with replacement means just randomize
                    #with no replacement means 0:50, 51:100, 101:150, ......300:323
                    #shuffle your index
                    perm = np.random.permutation(X_cross_train.shape[0])
                    X_cross_train = X_cross_train[perm]
                    y_cross_train = y_cross_train[perm]
                    
                    
                    
                    if self.method == 'sto':
                        for batch_idx in range(X_cross_train.shape[0]):
                            X_method_train = X_cross_train[batch_idx].reshape(1, -1) #(11,) ==> (1, 11) ==> (m, n)
                            y_method_train = y_cross_train[batch_idx].reshape(-1) 
                            train_loss = self._train(X_method_train, y_method_train)
                    elif self.method == 'mini':
                        for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                            #batch_idx = 0, 50, 100, 150
                            X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                            y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                            train_loss = self._train(X_method_train, y_method_train)
                    else:
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self._train(X_method_train, y_method_train)

                        
                        
                    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)
                    yhat_val = self.predict(X_cross_val)
                    val_loss_new = self.mse(y_cross_val, yhat_val)
                    mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch)
                    
                    
                    
                    #record dataset
                    mlflow_train_data = mlflow.data.from_numpy(features=X_method_train, targets=y_method_train)
                    mlflow.log_input(mlflow_train_data, context="training")
                    
                    mlflow_val_data = mlflow.data.from_numpy(features=X_cross_val, targets=y_cross_val)
                    mlflow.log_input(mlflow_val_data, context="validation")
                    
                    
                    
                    
                    #early stopping
                    if np.allclose(val_loss_new, self.val_loss_old):
                        break
                    self.val_loss_old = val_loss_new
                    
                self.theta_list.append(self.theta.copy())
                self.kfold_scores.append(val_loss_new)
                print(f"Fold {fold}: {val_loss_new}")

       
    def predict(self, X):
        X_poly = self._polynomial_features(X)
        return X_poly @ self.theta
    
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
    
    def predict_app(self,weight,input):
        mileage = self.z_score_scale(input[2])
        max_power = self.min_max_scale_max_power(input[1])
        year = self.min_max_scale_year(input[0])
        prepared_input = np.array([year,max_power,mileage]).reshape(1,3)

        X_poly = self._polynomial_features(prepared_input)
        
        return int(np.exp(X_poly @ weight.reshape(6,1)))

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
    
    def plot_feature_importance(self,weights, custom_labels, figsize=(12,5)):
    
        # Ensure weights and custom_labels are of the same length
        assert len(weights) == len(custom_labels), "Length of weights and custom_labels must be the same"

        data = np.abs(weights)
        x = np.arange(len(weights))

        fig, axes = plt.subplots(figsize=figsize)

        axes.bar(x=x, height=data)

        # Set custom x-tick labels for each x-value
        axes.set_xticks(x)
        axes.set_xticklabels(custom_labels)
        axes.set_xlabel("Features")
        axes.set_ylabel("Importance")
        plt.show()


