"""Data preprocessing utilities"""
import pandas as pd
import numpy as np


def clean_price_data(df):
    """Clean and prepare price data for feature extraction"""
    # Make a copy to avoid modifying the original
    df = df.copy()

    # Remove any duplicate timestamps
    df = df.drop_duplicates(subset=['time'])

    # Sort by time
    df = df.sort_values('time')

    # Reset index
    df = df.reset_index(drop=True)

    # Ensure required columns exist
    required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")

    return df


def add_basic_indicators(df):
    """Add basic technical indicators to price data"""
    df = df.copy()

    # Add some simple moving averages
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['ma200'] = df['close'].rolling(200).mean()

    # Basic trend indicator
    df['trend'] = np.where(df['close'] > df['ma50'], 1, -1)

    return df