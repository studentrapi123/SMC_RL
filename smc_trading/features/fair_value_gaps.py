"""Fair Value Gap detection and related features"""
import numpy as np
import pandas as pd

def find_fair_value_gaps_optimized(df):
    """Optimized detection of Fair Value Gaps"""
    # Pre-extract arrays for faster access
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_prices = df['open'].values

    # Dictionary to store events
    fvg_events = {'bullish': [], 'bearish': []}

    # Find Bullish Fair Value Gaps
    # (when the low of a candle is higher than the high of the candle before the previous one)
    for i in range(2, len(df)):
        if low[i] > high[i - 2]:
            # Gap detected
            start_idx = i - 2
            end_idx = i
            top_price = low[i]
            bottom_price = high[i - 2]

            # Check for mitigation
            mitigated = False
            mitigation_idx = None

            # Vectorized mitigation check
            min_open_close = np.minimum(open_prices[end_idx + 1:], close[end_idx + 1:])
            max_open_close = np.maximum(open_prices[end_idx + 1:], close[end_idx + 1:])

            # Find where price body enters the gap
            mitigated_indices = np.where(
                (min_open_close <= top_price) &
                (max_open_close >= bottom_price)
            )[0]

            if len(mitigated_indices) > 0:
                mitigated = True
                mitigation_idx = end_idx + 1 + mitigated_indices[0]

            fvg_events['bullish'].append((start_idx, end_idx, top_price, bottom_price, mitigated, mitigation_idx))

    # Find Bearish Fair Value Gaps
    # (when the high of a candle is lower than the low of the candle before the previous one)
    for i in range(2, len(df)):
        if high[i] < low[i - 2]:
            # Gap detected
            start_idx = i - 2
            end_idx = i
            top_price = low[i - 2]
            bottom_price = high[i]

            # Check for mitigation
            mitigated = False
            mitigation_idx = None

            # Vectorized mitigation check
            min_open_close = np.minimum(open_prices[end_idx + 1:], close[end_idx + 1:])
            max_open_close = np.maximum(open_prices[end_idx + 1:], close[end_idx + 1:])

            # Find where price body enters the gap
            mitigated_indices = np.where(
                (min_open_close <= top_price) &
                (max_open_close >= bottom_price)
            )[0]

            if len(mitigated_indices) > 0:
                mitigated = True
                mitigation_idx = end_idx + 1 + mitigated_indices[0]

            fvg_events['bearish'].append((start_idx, end_idx, top_price, bottom_price, mitigated, mitigation_idx))

    return fvg_events  # This line should be de-indented to align with the function definition

def add_fair_value_gap_features(df, fvg_events, atr=None):
    """Add fair value gap features to the dataframe without lookahead bias"""
    # Initialize fair value gap features
    df = df.copy()
    df['fvg_bull_present'] = 0
    df['fvg_bear_present'] = 0
    df['fvg_bull_distance_pct'] = np.nan
    df['fvg_bear_distance_pct'] = np.nan
    df['fvg_size_atr_ratio'] = 0.0

    # Create arrays for tracking all active FVGs at each bar
    bull_fvgs_active = []  # List of active bullish FVGs
    bear_fvgs_active = []  # List of active bearish FVGs

    # Process each row chronologically to avoid lookahead bias
    for i in range(len(df)):
        # Check if any new FVGs form at this bar
        for start_idx, end_idx, top_price, bottom_price, _, _ in fvg_events['bullish']:
            mid_idx = start_idx + 1
            if mid_idx == i:
                new_bull_fvg = True
                fvg_size = top_price - bottom_price

                # Volatility-adjusted size
                size_atr_ratio = 0.0
                if atr is not None and i < len(atr):
                    size_atr_ratio = fvg_size / atr[i]

                # Add to active FVGs list with essential info only
                bull_fvgs_active.append({
                    'idx': i,
                    'top_price': top_price,
                    'bottom_price': bottom_price,
                    'size_atr_ratio': size_atr_ratio,
                    'mitigated': False,
                    'formation_time': i
                })

                # Mark this candle as having a bullish FVG
                df.loc[i, 'fvg_bull_present'] = 1
                df.loc[i, 'fvg_size_atr_ratio'] = size_atr_ratio

        for start_idx, end_idx, top_price, bottom_price, _, _ in fvg_events['bearish']:
            mid_idx = start_idx + 1
            if mid_idx == i:
                new_bear_fvg = True
                fvg_size = top_price - bottom_price

                # Volatility-adjusted size
                size_atr_ratio = 0.0
                if atr is not None and i < len(atr):
                    size_atr_ratio = fvg_size / atr[i]

                # Add to active FVGs list with essential info only
                bear_fvgs_active.append({
                    'idx': i,
                    'top_price': top_price,
                    'bottom_price': bottom_price,
                    'size_atr_ratio': size_atr_ratio,
                    'mitigated': False,
                    'formation_time': i
                })

                # Mark this candle as having a bearish FVG
                df.loc[i, 'fvg_bear_present'] = 1
                df.loc[i, 'fvg_size_atr_ratio'] = size_atr_ratio

        # Update features and check for mitigation
        if i > 0:  # Skip first bar
            # Process bullish FVGs
            mitigated_bull_fvgs = []
            for j, fvg in enumerate(bull_fvgs_active):
                # Check for mitigation (candle body enters or crosses FVG)
                if min(df['open'].iloc[i], df['close'].iloc[i]) <= fvg['top_price'] and \
                        max(df['open'].iloc[i], df['close'].iloc[i]) >= fvg['bottom_price'] and \
                        not fvg['mitigated']:
                    bull_fvgs_active[j]['mitigated'] = True
                    mitigated_bull_fvgs.append(j)

            # Process bearish FVGs
            mitigated_bear_fvgs = []
            for j, fvg in enumerate(bear_fvgs_active):
                # Check for mitigation (candle body enters or crosses FVG)
                if min(df['open'].iloc[i], df['close'].iloc[i]) <= fvg['top_price'] and \
                        max(df['open'].iloc[i], df['close'].iloc[i]) >= fvg['bottom_price'] and \
                        not fvg['mitigated']:
                    bear_fvgs_active[j]['mitigated'] = True
                    mitigated_bear_fvgs.append(j)

            # Remove mitigated FVGs
            for j in sorted(mitigated_bull_fvgs, reverse=True):
                bull_fvgs_active.pop(j)

            for j in sorted(mitigated_bear_fvgs, reverse=True):
                bear_fvgs_active.pop(j)

        # Set features based on most relevant FVG
        if bull_fvgs_active:
            # Calculate distances and importances
            distances = []
            importances = []
            for fvg in bull_fvgs_active:
                # Distance calculation
                if df['close'].iloc[i] < fvg['bottom_price']:
                    distance = (fvg['bottom_price'] - df['close'].iloc[i]) / df['close'].iloc[i] * 100
                elif df['close'].iloc[i] > fvg['top_price']:
                    distance = (df['close'].iloc[i] - fvg['top_price']) / df['close'].iloc[i] * 100
                else:
                    distance = 0.0  # Inside the FVG
                distances.append((distance, fvg))

                # Calculate importance with exponential decay
                age = i - fvg['formation_time']
                importance = fvg['size_atr_ratio'] * np.exp(-0.1 * age)
                importances.append((importance, fvg))

            # Find closest FVG
            closest_fvg = min(distances, key=lambda x: abs(x[0]) if x[0] is not None else float('inf'))
            df.loc[i, 'fvg_bull_distance_pct'] = closest_fvg[0]

            # Find most important FVG
            most_important_fvg = max(importances, key=lambda x: x[0])
            df.loc[i, 'fvg_size_atr_ratio'] = most_important_fvg[1]['size_atr_ratio'] * np.exp(
                -0.1 * (i - most_important_fvg[1]['formation_time']))

        # Similar logic for bearish FVGs
        if bear_fvgs_active:
            distances = []
            importances = []
            for fvg in bear_fvgs_active:
                if df['close'].iloc[i] < fvg['bottom_price']:
                    distance = (fvg['bottom_price'] - df['close'].iloc[i]) / df['close'].iloc[i] * 100
                elif df['close'].iloc[i] > fvg['top_price']:
                    distance = (df['close'].iloc[i] - fvg['top_price']) / df['close'].iloc[i] * 100
                else:
                    distance = 0.0
                distances.append((distance, fvg))

                age = i - fvg['formation_time']
                importance = fvg['size_atr_ratio'] * np.exp(-0.1 * age)
                importances.append((importance, fvg))

            closest_fvg = min(distances, key=lambda x: abs(x[0]) if x[0] is not None else float('inf'))
            df.loc[i, 'fvg_bear_distance_pct'] = closest_fvg[0]

            most_important_fvg = max(importances, key=lambda x: x[0])
            df.loc[i, 'fvg_size_atr_ratio'] = most_important_fvg[1]['size_atr_ratio'] * np.exp(
                -0.1 * (i - most_important_fvg[1]['formation_time']))

    # Add trend context features
    if 'trend' not in df.columns and len(df) >= 50:
        df['ma50'] = df['close'].rolling(50).mean()
        df['trend'] = np.where(df['close'] > df['ma50'], 1, -1)

        # FVGs in context of trend
        df['fvg_bull_with_trend'] = df['fvg_bull_present'] * (df['trend'] == 1)
        df['fvg_bear_with_trend'] = df['fvg_bear_present'] * (df['trend'] == -1)

    # Calculate active FVG counts
    df['active_fvg_count'] = (df['fvg_bull_present'].rolling(10).sum() +
                              df['fvg_bear_present'].rolling(10).sum())

    return df