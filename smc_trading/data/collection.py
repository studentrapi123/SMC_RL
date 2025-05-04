import MetaTrader5 as mt5
import pandas as pd
import os
import pytz
from datetime import datetime

# In smc_trading/data/storage.py

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

def get_historical_data(symbol, start_date, end_date, timeframe, mt5_instance):
    """
    Get historical data for a specific symbol and timeframe between start_date and end_date
    """
    # Set timezone to UTC
    timezone = pytz.timezone("Etc/UTC")

    # Ensure dates are in UTC timezone
    if start_date.tzinfo is None:
        start_date = timezone.localize(start_date)
    if end_date.tzinfo is None:
        end_date = timezone.localize(end_date)

    # Get rates from MT5
    rates = mt5_instance.copy_rates_range(symbol, timeframe, start_date, end_date)

    if rates is None or len(rates) == 0:
        print(f"No data received for {symbol} between {start_date} and {end_date}")
        print(f"Error: {mt5.last_error()}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)

    # Convert time column from unix timestamp to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')

    return df

def get_timeframe_name(timeframe):
    """Convert MT5 timeframe constant to string representation"""
    timeframe_dict = {
        mt5.TIMEFRAME_M1: "M1",
        mt5.TIMEFRAME_M5: "M5",
        mt5.TIMEFRAME_M15: "M15",
        mt5.TIMEFRAME_M30: "M30",
        mt5.TIMEFRAME_H1: "H1",
        mt5.TIMEFRAME_H4: "H4",
        mt5.TIMEFRAME_D1: "D1",
        mt5.TIMEFRAME_W1: "W1",
        mt5.TIMEFRAME_MN1: "MN1"
    }
    return timeframe_dict.get(timeframe, "Unknown")


def download_all_timeframes(symbol, start_date, end_date, mt5_instance):
    """Download data for multiple timeframes for a specific symbol"""
    try:
        # List of timeframes to fetch
        timeframes = [
            mt5.TIMEFRAME_M1,  # 1-minute
            mt5.TIMEFRAME_M5,  # 5-minute
            mt5.TIMEFRAME_M15,  # 15-minute
            mt5.TIMEFRAME_M30,  # 30-minute
            mt5.TIMEFRAME_H1,  # 1-hour
            mt5.TIMEFRAME_H4,  # 4-hour
            mt5.TIMEFRAME_D1  # 1-day
        ]

        # Default dates if not provided
        if start_date is None:
            start_date = datetime(2025, 1, 1)
        if end_date is None:
            end_date = datetime(2025, 5, 3)

        print(f"Retrieving {symbol} data from {start_date} to {end_date} for multiple timeframes")

        # Process each timeframe
        results = {}
        for timeframe in timeframes:
            timeframe_name = get_timeframe_name(timeframe)
            print(f"\nFetching {timeframe_name} data for {symbol}...")

            # Get historical data
            df = get_historical_data(symbol, start_date, end_date, timeframe, mt5_instance)

            if df is not None and not df.empty:
                results[timeframe_name] = df

                # CHANGE THIS LINE: use base_dir instead of output_dir
                save_to_csv(df, symbol, timeframe_name, base_dir="data/raw")
            else:
                print(f"Failed to retrieve {timeframe_name} data for {symbol}")

        return results

    except Exception as e:
        print(f"Error in download_all_timeframes: {e}")
        return {}