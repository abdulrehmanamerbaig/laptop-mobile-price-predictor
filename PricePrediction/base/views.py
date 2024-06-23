from django.shortcuts import render 
from django.http import HttpResponse
import pandas as pd
import numpy as np
import pickle
import numpy as np

# Load the saved model


# Create your views here.
def home(request):
    return render(request, 'home.html')

def load_model():
    import os
    import pickle

    # Get the directory of the current Python script (models.py)
    current_dir = os.path.dirname(__file__)

    # Construct the path to the .pkl file relative to the current directory
    pkl_file_path = os.path.join(current_dir, 'laptop_price_model.pkl')

    # Load the pickled model
    with open(pkl_file_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_mobile_model():
    import os
    from joblib import load

    # Get the directory of the current Python script (models.py)
    current_dir = os.path.dirname(__file__)

    # Construct the path to the .joblib file relative to the current directory
    joblib_file_path = os.path.join(current_dir, 'mobile_price_model.joblib')

    # Load the joblib model
    mobile_model = load(joblib_file_path)
    return mobile_model

def laptop(request):
    if request.method == 'POST':
        brand = request.POST['brand']
        operating_system = request.POST['operating_system']
        processor = request.POST['processor']
        # Extract numeric part of screen size and convert to float
        screen_size_str = request.POST['screen_size']
        
        screen_size_str = screen_size_str.split()[0]  # Get only the numeric part before the first space
        try:
            screen_size_str = float(screen_size_str)
        except ValueError:
            screen_size_str = None  # If conversion fails, set to None
        print ("HELLLLLO", operating_system, screen_size_str)
        # Extract numeric part of storage and RAM and convert to integers
        storage_str = request.POST['storage']
        storage_str = storage_str.split('GB')[0]  # Get only the numeric part before "GB"
        try:
            storage = int(storage_str)
        except ValueError:
            storage = None  # If conversion fails, set to None

        ram_str = request.POST['ram']
        ram_str = ram_str.split('GB')[0]  # Get only the numeric part before "GB"
        try:
            storage = int(storage_str)
        except ValueError:
            storage = None  # If conversion fails, set to None

        ram_str = request.POST['ram']
        ram_str = ram_str.split('GB')[0]  # Get only the numeric part before "GB"
        try:
            ram = int(ram_str)
        except ValueError:
            ram = None  # If conversion fails, set to None

        touch_screen = request.POST['touch_screen']

        # Load the model
        model = load_model()

        # Define the query data
        query_data = {
            'Brand': [brand],
            'Processor': [processor],
            'Operating System': [operating_system],
            'Storage': [storage],
            'RAM': [ram],
            'Screen Size': [screen_size_str],
            'Touch_Screen': [touch_screen]
        }

        values = np.array([operating_system, screen_size_str])
        # Convert the query data into a DataFrame
        query_df = pd.DataFrame(query_data)

        # Make predictions using the model
        predicted_log_price = model.predict(query_df)
        predicted_price = np.exp(predicted_log_price)
        values = values.tolist()
        return render(request, 'result.html', {'query_data': query_data, 'values': values, 'predicted_price': predicted_price*3.1})

    return render(request, 'laptop.html')


def result(request):
    return HttpResponse('Price is 1000')


def mobile(request):
    if request.method == 'POST':
        weight = request.POST['weight']
        try:
            weight = float(weight)
        except ValueError:
            weight = None

        resolution = request.POST['resolution']
        try:
            resolution = float(resolution)
        except ValueError:
            resolution = None

        ppi = request.POST['ppi']
        try:
            ppi = float(ppi)
        except ValueError:
            ppi = None

        cpu_core = request.POST['cpu_core']
        try:
            cpu_core = float(cpu_core)
        except ValueError:
            cpu_core = None
        
        cpu_freq = request.POST['cpu_freq']
        try:
            cpu_freq = float(cpu_freq)
        except ValueError:
            cpu_freq = None

        internal_mem = request.POST['internal_mem']
        try:
            internal_mem = float(internal_mem)
        except ValueError:
            internal_mem = None

        ram = request.POST['ram']
        try:
            ram = float(ram)
        except ValueError:
            ram = None
        
        front_camera = request.POST['front_camera'] # Get only the numeric part before "GB"
        try:
            front_camera = float(front_camera)
        except ValueError:
            front_camera = None

        battery = request.POST['battery']
        try:
            battery = int(battery)
        except ValueError:
            battery = None
        
        thickness = request.POST['thickness']
        try:
            thickness = float(thickness)
        except ValueError:
            thickness = None

        values=np.array([weight, resolution, ppi, cpu_core, cpu_freq,
                 internal_mem, ram, front_camera, battery, thickness])
        # Load the model
        model = load_mobile_model()
        print (values)
        column_names = ["weight", "resoloution", "ppi",
                "cpu core", "cpu freq", "internal mem", "ram",
                "Front_Cam", "battery", "thickness"]
        # Make predictions using the model
        query_df = pd.DataFrame(data=[values], columns=column_names)
        predicted_price = model.predict(query_df)
        print (predicted_price)
        values_list = values.tolist()
        return render(request, 'result_mobile.html', {'query_df': query_df, 'values':values_list, 'predicted_price': predicted_price*3.4})

    return render(request, 'mobile.html')
