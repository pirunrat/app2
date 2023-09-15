from django.shortcuts import render, redirect
from .form import CarPredictionForm
from .Machine_Learning_Model import Model
from .Polynomial_Model import Polynomial
import numpy as np




def main_page(request):

    # Validate the form data
    if request.method == "POST":  # Check if the request method is POST
        choice = request.POST.get('choice')  # Getting the choice value
        
        if choice == 'old_page':
            return old_page(request)
        elif choice == 'new_page':
            return new_page(request)

    return render(request, 'main_page.html')

def old_page(request):
    prediction = ''

    if request.method == 'POST':

        form = CarPredictionForm(request.POST)

        if form.is_valid():
            mileage = form.cleaned_data['mileage']
            max_power = form.cleaned_data['max_power']
            year = form.cleaned_data['year']
            input_data = {'mileage': mileage, 'max_power': max_power,'year':year}
            model = Model(input_data,'car-pridiction.model')
            prediction = model.model_predict()
            
    else:
        form = CarPredictionForm()

    return render(request, 'old_page.html', {'form': form, 'prediction': prediction})

    

def new_page(request):
    prediction = ''

    if request.method == 'POST':

        form = CarPredictionForm(request.POST)

        if form.is_valid():

            # Get the data from user
            mileage = form.cleaned_data['mileage']
            max_power = form.cleaned_data['max_power']
            year = form.cleaned_data['year']

            # Rearrange the data as an array
            input_data = np.array([year,max_power,mileage])

            # Instantiate a polynomial class
            polynomial_model = Polynomial()

            # Instantiate a model class
            model = Model(input_data,'weight_1.3.model')

            # Load the trained model
            model_loaded = model.load_model_poly()

            # Make a prediction
            prediction = polynomial_model.predict(model_loaded,input_data)
    else:
        form = CarPredictionForm()

    return render(request, 'new_page.html', {'form': form, 'prediction': prediction})

