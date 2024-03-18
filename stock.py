import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf

# Function to fetch historical stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to preprocess the data
def preprocess_data(stock_data):
    # Extracting 'Close' prices as the target variable
    target = stock_data['Close'].values

    # Extracting 'Open' prices as the feature
    features = stock_data['Open'].values.reshape(-1, 1)

    return features, target

# Function to train the model
def train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R-squared Score:", r2)

# Function to make predictions
def make_predictions(model, features):
    predictions = model.predict(features)
    return predictions

# Main function
def main():
    # Input parameters
    ticker = 'AAPL'  # Example: Apple stock ticker
    start_date = '2020-01-01'
    end_date = '2022-01-01'

    # Get historical stock data
    stock_data = get_stock_data(ticker, start_date, end_date)

    # Preprocess the data
    features, target = preprocess_data(stock_data)

    # Train the model
    model, X_test, y_test = train_model(features, target)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Make predictions
    future_date = '2024-01-01'
    future_data = get_stock_data(ticker, end_date, future_date)
    future_features, _ = preprocess_data(future_data)
    predictions = make_predictions(model, future_features)

    print("Predictions for future dates:")
    print(predictions)

if __name__ == "__main__":
    main()
