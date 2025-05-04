# main.py
import os
from datetime import datetime
import MetaTrader5 as mt5
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

# Import your modules
from smc_trading.data.collection import download_all_timeframes, get_historical_data, get_timeframe_name
from smc_trading.data.storage import save_to_csv, load_price_data
from smc_trading.features.feature_set import process_symbol_timeframe
from smc_trading.features.swing_points import directional_change_adaptive_optimized, mark_swing_points
from smc_trading.features.order_blocks import detect_bos_events, plot_swing_bos_orderblocks


def main():
    # Define symbols and timeframes to process
    symbols = ["EURUSD"]
    timeframes = ["M5", "M15", "M30", "H1", "H4", "D1"]

    # Create data directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)

    # Initialize MT5 connection
    if not mt5.initialize():
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        return

    print("MT5 initialized successfully")

    # Login to MT5 account
    account = 1510637775  # Your account number
    password = "4x7@?wQI59?R3"  # Your password
    server = "FTMO-Demo"  # Your broker's server

    if not mt5.login(account, password, server):
        print(f"Failed to login: {mt5.last_error()}")
        mt5.shutdown()
        return

    print(f"Logged in to account {account}")

    # Download data from MT5
    print("Downloading price data...")
    start_date = datetime(2025, 4, 1)
    end_date = datetime(2025, 5, 10)

    for symbol in symbols:
        download_all_timeframes(symbol, start_date, end_date, mt5)

    # Shutdown MT5 when done downloading
    mt5.shutdown()

    # Process features in parallel
    print("Processing SMC features...")
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for symbol in symbols:
            for timeframe in timeframes:
                futures.append(executor.submit(process_symbol_timeframe, symbol, timeframe))

        # Wait for all tasks to complete
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error processing task: {e}")

    print("All processing completed!")

    # Create visualizations for each timeframe (H4 and H1 for brevity)
    print("Generating order block visualizations...")
    visualization_timeframes = ["M5", "M15"]

    for symbol in symbols:
        for timeframe in visualization_timeframes:
            print(f"Creating visualization for {symbol} {timeframe}...")

            # Load the raw price data
            df = load_price_data(symbol, timeframe, base_dir="data/raw")

            # Calculate swing points
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            tops, bottoms, atr = directional_change_adaptive_optimized(
                close, high, low, atr_period=14, atr_multiplier=1.5
            )

            # Mark swing points
            df = mark_swing_points(df, tops, bottoms)

            # Detect BOS events and order blocks
            bos_events, order_blocks = detect_bos_events(df)

            # Print some stats
            print(f"Found {len(bos_events['bullish'])} bullish BOS events for {symbol} {timeframe}")
            print(f"Found {len(bos_events['bearish'])} bearish BOS events for {symbol} {timeframe}")
            print(f"Found {len(order_blocks['bullish'])} bullish order blocks for {symbol} {timeframe}")
            print(f"Found {len(order_blocks['bearish'])} bearish order blocks for {symbol} {timeframe}")

            # Create multiple visualizations for different sections of the data
            sections = [
                (0, 100),  # Beginning
                (len(df) // 2 - 50, len(df) // 2 + 50),  # Middle
                (max(0, len(df) - 100), len(df))  # End
            ]

            for start_idx, end_idx in sections:
                section_name = "beginning" if start_idx == 0 else ("middle" if start_idx > 100 else "end")
                save_path = f"visualizations/{symbol}_{timeframe}_{section_name}_orderblocks.png"

                # Create the visualization
                plot_swing_bos_orderblocks(
                    df, tops, bottoms, atr, bos_events, order_blocks,
                    start_idx=start_idx, end_idx=end_idx, save_path=save_path
                )

                print(f"Visualization saved to {save_path}")


if __name__ == "__main__":
    main()