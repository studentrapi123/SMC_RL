# main.py
import os
from datetime import datetime
import MetaTrader5 as mt5
from concurrent.futures import ProcessPoolExecutor

# Import your modules
from smc_trading.data.collection import download_all_timeframes, get_historical_data, get_timeframe_name
from smc_trading.data.storage import save_to_csv
from smc_trading.features.feature_set import process_symbol_timeframe


def main():
    # Define symbols and timeframes to process
    symbols = ["EURUSD"]
    timeframes = ["M5", "M15", "M30", "H1", "H4", "D1"]

    # Create data directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

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
    start_date = datetime(2025, 1, 1)
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


if __name__ == "__main__":
    main()