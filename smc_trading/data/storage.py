import pandas as pd
import os


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

    # Save to CSV
    df_to_save.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")
    return file_path


def load_price_data(symbol, timeframe, base_dir="data/raw"):
    """Load price data from CSV"""
    csv_path = os.path.join(base_dir, symbol, f"{symbol}_{timeframe}.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No data file found for {symbol} {timeframe}")

    df = pd.read_csv(csv_path)

    # Convert time column to datetime if it exists
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])

    return df


def save_features(df, symbol, timeframe, base_dir="data/processed"):
    """Save DataFrame with features to CSV format"""
    # Create directory structure if it doesn't exist
    symbol_dir = os.path.join(base_dir, symbol)
    os.makedirs(symbol_dir, exist_ok=True)

    # Define output file path
    file_path = os.path.join(symbol_dir, f"{symbol}_{timeframe}_features.csv")

    # Save to CSV
    df.to_csv(file_path, index=False)
    print(f"Features saved to {file_path}")
    print(f"Feature length = {len(df)}")

    return file_path


def load_features(symbol, timeframe, base_dir="data/processed"):
    """Load features from CSV file"""
    file_path = os.path.join(base_dir, symbol, f"{symbol}_{timeframe}_features.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Features file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Convert any list columns from string representation back to actual lists
    list_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in list_columns:
        try:
            # Handle empty strings (which would be empty lists)
            df[col] = df[col].apply(lambda x: eval(x) if pd.notna(x) and x != '' else [])
        except:
            # If eval fails, leave as is
            pass

    return df