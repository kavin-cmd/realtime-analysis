# train.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_collection import fetch_stock_data
from preprocessing import preprocess_data
from feature_engineering import create_features
from model import build_lstm_model, train_model

def prepare_data(df, target_column='close'):
    features = df.drop(columns=[target_column])
    target = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
    X_train = np.expand_dims(X_train.values, axis=1)
    X_test = np.expand_dims(X_test.values, axis=1)
    return X_train, X_test, y_train, y_test

# Fetch stock data
api_key = 'YOUR_API_KEY'
symbol = 'AAPL'
df = fetch_stock_data(api_key, symbol)

# Preprocess data
df_preprocessed = preprocess_data(df)

# Create features
df_with_features = create_features(df_preprocessed)

# Prepare data
X_train, X_test, y_train, y_test = prepare_data(df_with_features)

# Build and train model
model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
history = train_model(model, X_train, y_train)
