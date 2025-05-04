"""Complete SMC feature extraction pipeline"""
import pandas as pd
import numpy as np
import time
import os

from ..data.storage import save_features
from .swing_points import calculate_atr_vectorized, directional_change_adaptive_optimized, mark_swing_points
from .order_blocks import detect_bos_events, add_order_block_features
from .fair_value_gaps import find_fair_value_gaps_optimized, add_fair_value_gap_features
from .structure import add_advanced_smc_features_optimized


def extract_all_smc_features(df, atr_period=14, atr_multiplier=1.5, min_bars_between=1, confirmation_bars=1):
    """
    Extract all SMC features from price data with the improved order block tracking

    Args:
        df: DataFrame with OHLC price data
        atr_period: Period for ATR calculation
        atr_multiplier: Multiplier for ATR threshold
        min_bars_between: Minimum bars between swing points
        confirmation_bars: Bars for confirmation

    Returns:
        DataFrame with all SMC features added
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Extract price arrays
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    # Step 1: Calculate ATR and detect swing points
    from .swing_points import directional_change_adaptive_optimized, mark_swing_points
    tops, bottoms, atr = directional_change_adaptive_optimized(
        close, high, low,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        min_bars_between=min_bars_between,
        confirmation_bars=confirmation_bars
    )
    df['atr'] = atr
    print(f"Swing point detection completed")

    # Step 2: Mark swing points in dataframe
    df = mark_swing_points(df, tops, bottoms)

    # Step 3: Detect Break of Structure events
    bos_events, _ = detect_bos_events(df)
    print(
        f"BOS detection completed - Found {len(bos_events['bullish'])} bullish and {len(bos_events['bearish'])} bearish BOS events")

    # Step 4: Add enhanced order block features
    df = add_order_block_features(df, bos_events, atr)

    # Now we can safely print information about order blocks
    print(f"Active bullish OBs: {df['bull_ob_count'].iloc[-1]}, Active bearish OBs: {df['bear_ob_count'].iloc[-1]}")

    # Step 5: Add other SMC features (fair value gaps, etc.)
    # ... (your existing code)
    '''df = add_fair_value_gap_features(df, fvg_events, atr)
        df = add_advanced_smc_features_optimized(df, tops, bottoms, bos_events, order_blocks, fvg_events, atr)
        print(f"Feature calculation completed in {time.time() - start_time:.2f} seconds")'''

    # Return processed dataframe
    return df


def process_symbol_timeframe(symbol, timeframe, base_dir="data"):
    """Process a single symbol/timeframe combination"""
    from ..data.storage import load_price_data, save_features

    raw_data_dir = os.path.join(base_dir, "raw")
    processed_dir = os.path.join(base_dir, "processed")

    # Define paths
    features_path = os.path.join(processed_dir, symbol, f"{symbol}_{timeframe}_features.parquet")

    # Check if features already exist
    if os.path.exists(features_path):
        # Check if raw data is newer than features
        raw_path = os.path.join(raw_data_dir, symbol, f"{symbol}_{timeframe}.parquet")
        if os.path.exists(raw_path) and os.path.getmtime(raw_path) <= os.path.getmtime(features_path):
            print(f"Using existing features for {symbol} {timeframe}")
            return True

    try:
        # Load price data
        df = load_price_data(symbol, timeframe, base_dir=raw_data_dir)

        # Extract features
        print(f"Processing {symbol} {timeframe}...")
        df_features = extract_all_smc_features(df)

        # Save features
        save_features(df_features, symbol, timeframe, base_dir=processed_dir)

        print(f"Completed processing {symbol} {timeframe}")
        return True
    except Exception as e:
        print(f"Error processing {symbol} {timeframe}: {e}")
        import traceback
        traceback.print_exc()
        return False