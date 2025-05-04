"""Swing point detection and related utilities"""
import numpy as np
import pandas as pd


def calculate_atr_vectorized(high, low, close, period=14):
    """Vectorized ATR calculation for better performance"""
    # Create array of previous close values (shifted by 1)
    prev_close = np.zeros_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    # Calculate True Range components
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)

    # True Range is the maximum of the three components
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    # Initialize ATR array
    atr = np.zeros_like(tr)
    atr[0] = tr[0]

    # Calculate smoothed ATR
    for i in range(1, len(tr)):
        atr[i] = ((period - 1) * atr[i - 1] + tr[i]) / period

    return atr


def directional_change_adaptive_optimized(close, high, low, atr_period=14, atr_multiplier=1.5,
                                          min_bars_between=1, confirmation_bars=1):
    """Optimized directional change algorithm with adaptive threshold based on ATR"""
    # Calculate ATR using vectorized function
    atr = calculate_atr_vectorized(high, low, close, atr_period)

    # Pre-allocate arrays for results
    tops = []
    bottoms = []

    # Initialize tracking variables
    up_zig = True
    tmp_max = high[0]
    tmp_min = low[0]
    tmp_max_i = 0
    tmp_min_i = 0
    last_extreme_i = 0

    # Calculate threshold array once (instead of inside the loop)
    threshold_pct = atr * atr_multiplier / close

    # Main loop
    for i in range(len(close) - confirmation_bars):
        # Ensure minimum distance from last extreme point
        if i - last_extreme_i < min_bars_between:
            continue

        # Get current threshold
        current_threshold = threshold_pct[i]

        if up_zig:
            if high[i] > tmp_max:
                tmp_max = high[i]
                tmp_max_i = i
            # Check if we have a potential top
            elif close[i] < tmp_max - tmp_max * current_threshold:
                # Faster confirmation check
                is_confirmed = True
                for j in range(1, confirmation_bars + 1):
                    if i + j < len(high) and high[i + j] >= tmp_max:
                        is_confirmed = False
                        break

                if is_confirmed:
                    tops.append([i, tmp_max_i, tmp_max])
                    last_extreme_i = tmp_max_i

                    up_zig = False
                    tmp_min = low[i]
                    tmp_min_i = i
        else:
            if low[i] < tmp_min:
                tmp_min = low[i]
                tmp_min_i = i
            # Check if we have a potential bottom
            elif close[i] > tmp_min + tmp_min * current_threshold:
                # Confirmation check
                is_confirmed = True
                for j in range(1, confirmation_bars + 1):
                    if i + j < len(low) and low[i + j] <= tmp_min:
                        is_confirmed = False
                        break

                if is_confirmed:
                    bottoms.append([i, tmp_min_i, tmp_min])
                    last_extreme_i = tmp_min_i

                    up_zig = True
                    tmp_max = high[i]
                    tmp_max_i = i

    return tops, bottoms, atr


def mark_swing_points(df, tops, bottoms):
    """Mark swing points in the dataframe"""
    df = df.copy()

    # Initialize columns
    df['is_swing_high'] = 0
    df['is_swing_low'] = 0

    # Mark swing highs
    for _, max_idx, _ in tops:
        if max_idx < len(df):
            df.loc[max_idx, 'is_swing_high'] = 1

    # Mark swing lows
    for _, min_idx, _ in bottoms:
        if min_idx < len(df):
            df.loc[min_idx, 'is_swing_low'] = 1

    return df