import pandas as pd
import numpy as np
import os


def convert_csv_to_parquet(csv_path):
    """Convert CSV file to Parquet for better performance"""
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return False

    parquet_path = csv_path.replace('.csv', '.parquet')

    # Check if parquet file already exists and is newer than csv
    if os.path.exists(parquet_path) and os.path.getmtime(parquet_path) > os.path.getmtime(csv_path):
        print(f"Using existing parquet file: {parquet_path}")
        return parquet_path

    # Read CSV and convert
    df = pd.read_csv(csv_path)

    # Write to parquet
    df.to_parquet(parquet_path, index=False)
    print(f"Converted {csv_path} to {parquet_path}")
    return parquet_path


def load_price_data(symbol, timeframe, base_dir="data/raw"):
    """Load price data, preferring Parquet if available"""
    parquet_path = os.path.join(base_dir, symbol, f"{symbol}_{timeframe}.parquet")
    csv_path = os.path.join(base_dir, symbol, f"{symbol}_{timeframe}.csv")

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Convert time column to datetime if it exists
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
    else:
        raise FileNotFoundError(f"No data file found for {symbol} {timeframe}")

    return df


def save_to_csv(df, symbol, timeframe, base_dir="data/raw"):
    """Save DataFrame to CSV in organized folder structure"""
    # Create a copy of the DataFrame to avoid modifying the original
    df_to_save = df.copy()

    # Drop the real_volume column if it exists (MT5 specific)
    if 'real_volume' in df_to_save.columns:
        df_to_save = df_to_save.drop(columns=['real_volume'])

    # Create directory structure if it doesn't exist
    symbol_dir = os.path.join(base_dir, symbol)
    os.makedirs(symbol_dir, exist_ok=True)

    # Define output file path
    file_path = os.path.join(symbol_dir, f"{symbol}_{timeframe}.csv")

    print(len(df_to_save))

    # Save to CSV
    df_to_save.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")
    return file_path


def save_features(df, symbol, timeframe, base_dir="data/processed"):
    """Save DataFrame with features to Parquet format"""
    # Create directory structure if it doesn't exist
    symbol_dir = os.path.join(base_dir, symbol)
    os.makedirs(symbol_dir, exist_ok=True)

    # Define output file path
    file_path = os.path.join(symbol_dir, f"{symbol}_{timeframe}_features.parquet")

    print(f"feature length = {len(df)}")

    # Save to Parquet
    df.to_parquet(file_path, index=False)
    print(f"Features saved to {file_path}")
    return file_path


def load_features(symbol, timeframe, base_dir="data/processed"):
    """Load features from Parquet file"""
    file_path = os.path.join(base_dir, symbol, f"{symbol}_{timeframe}_features.parquet")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Features file not found: {file_path}")

    return pd.read_parquet(file_path)