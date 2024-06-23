Laptop and Mobile Price Prediction in PKR
This project is a web application that predicts the prices of laptops and mobile phones in Pakistani Rupees (PKR). The models were trained on data collected via web scraping from various e-commerce websites.

Features
Laptop Price Prediction: Predict laptop prices based on brand, operating system, processor, storage, RAM, screen size, and touch screen availability.
Mobile Price Prediction: Predict mobile phone prices based on brand, model, storage, RAM, camera specifications, battery capacity, and screen size.
The predicted prices are displayed in PKR.
User-friendly web interface to input device specifications and get predictions.
Project Structure
views.py: Contains the views for handling user requests and rendering HTML templates.
urls.py: URL configuration for routing requests to appropriate views.
templates: HTML templates for rendering the web pages.
laptop_price_model.pkl: Pickle file of the trained laptop price prediction model.
mobile_price_model.pkl: Pickle file of the trained mobile price prediction model.
Data Collection
The training data was collected through web scraping various e-commerce websites. The datasets include details such as brand, operating system, processor, storage, RAM, screen size, touch screen availability, camera specifications, battery capacity, and price.

Installation
Prerequisites
Python 3.x
Django
Pandas
NumPy
Scikit-learn
Pickle

Steps
Clone the repository:

git clone git@github.com:naumansameja/Laptop-Price-Prediction.git
cd device-price-prediction

Install the required packages:
pip install -r requirements.txt

Migrate the database:
python manage.py migrate

Run the development server:
python manage.py runserver

Open your browser and go to:
http://127.0.0.1:8000/
Usage
Home Page:

The home page welcomes the user to the device price prediction application.
Predict Laptop Price:

Navigate to the laptop price prediction page.
Enter the specifications of the laptop including brand, operating system, processor, storage, RAM, screen size, and touch screen availability.
Submit the form to get the predicted price in PKR.
Predict Mobile Price:

Navigate to the mobile price prediction page.
Enter the specifications of the mobile phone including brand, model, storage, RAM, camera specifications, battery capacity, and screen size.
Submit the form to get the predicted price in PKR.
Example
Laptop Price Prediction:
Input:
Brand: Apple
Operating System: Mac
Processor: Mac
Storage: 256 GB
RAM: 8 GB
Screen Size: 13.3 inches
Touch Screen: No
Output:
Predicted Price: PKR 180,000 (example value)
Mobile Price Prediction:
Input:
Brand: Samsung
Model: Galaxy S21
Storage: 128 GB
RAM: 8 GB
Camera: 108 MP
Battery: 4000 mAh
Screen Size: 6.2 inches
Output:
Predicted Price: PKR 150,000 (example value)
Note
The models were trained using the prices and specifications of devices available online. The predicted prices may vary depending on the current market trends.

Acknowledgments
The datasets used for training were collected via web scraping from various e-commerce websites.