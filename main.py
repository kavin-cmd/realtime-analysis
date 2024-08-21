# main.py

from data_collection import fetch_stock_data
from preprocessing import preprocess_data
from feature_engineering import create_features
from train import prepare_data
from model import build_lstm_model, train_model

def main(api_key, symbol):
    # Step 1: Fetch stock data
    df = fetch_stock_data(api_key, symbol)
    
    # Step 2: Preprocess data
    df_preprocessed = preprocess_data(df)
    
    # Step 3: Create features
    df_with_features = create_features(df_preprocessed)
    
    # Step 4: Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df_with_features)
    
    # Step 5: Build and train model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    train_model(model, X_train, y_train)

if __name__ == "__main__":
    api_key = 'OHYFACJD1WUOBAX6'
    symbol = 'AAPL'
    main(api_key, symbol)
