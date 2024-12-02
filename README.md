# 📱 Mobile Price Prediction 🔮

Ever wondered how much a mobile phone should cost based on its specs? This project predicts mobile phone prices using Linear Regression. It includes everything from data preprocessing to model training, evaluation, and visualization. Plus, I've made it interactive with a Flask web app so you can try it out for yourself!

## 📝 Project Overview

In this project, we’re building a Linear Regression model to predict mobile phone prices. The model is trained on a dataset with mobile phone specifications, like camera quality, RAM, battery, and more. After training, it predicts the price based on those features. We also have a simple Flask web app that lets you input your own phone specs and get a price prediction.

## ✨ Features

- **🔧 Data Preprocessing: Cleaning and preparing data to make it model-ready.
- **📊 Model Training: Training the linear regression model using scikit-learn.
- **✅ Model Evaluation: Measuring model performance with various metrics.
- **📈 Visualization: Visualizing data and model performance.
- **🌐 Web Application: A simple Flask web app to get price predictions based on your inputs.

## 🛠️ Setup and Installation

1. **Clone the repository**:
    ```
    git clone https://github.com/fazaulfath/Mobile_Price_Prediction.git
    ```

2. **Create and activate a virtual environment**:
    ```
    python -m venv venv
    # On Windows use: venv\Scripts\activate
    source venv/bin/activate  # For macOS/Linux
    ```

3. **Install the required packages**:
    ```
    pip install -r requirements.txt
    ```

4. **Run the web application**:
    ```
    python app.py
    ```

## 🚀 Usage

1. **Data Preparation**:
    - Load the dataset and preprocess it.
    - Split the data into training and testing sets.

2. **Model Training and Evaluation**:
    - Train the linear regression model.
    - Evaluate the model's performance on the test set.

3. **Web Application**:
    - Access the web app at `http://localhost:5000`.
    - Enter the mobile phone specifications in the form.
    - Get the predicted price for the mobile phone.

## 📂 File Structure

- `app.py`: The main Flask web app script.
- `model.pkl`: The trained linear regression model.
- `scaler.pkl`: The scaler used for standardizing input features.
- `requirements.txt`: List of required Python packages.

## 📋 Requirements

- joblib
- numpy
- flask
- scikit-learn
- seaborn
- pandas