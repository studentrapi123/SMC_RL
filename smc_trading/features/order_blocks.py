"""Order block detection and related features"""
import numpy as np
import pandas as pd

def find_order_block_for_bullish_bos(df, swing_high_index, break_index):
    """
    Find the order block for a bullish BOS (the last bearish candle in the range)
    """
    # Loop backward from the break index to the swing high index
    for i in range(break_index - 1, swing_high_index, -1):
        # Check if it's a bearish candle (close < open)
        if df['close'].iloc[i] < df['open'].iloc[i]:
            return i

    return -1  # No suitable candle found

def find_order_block_for_bearish_bos(df, swing_low_index, break_index):
    """
    Find the order block for a bearish BOS (the last bullish candle in the range)
    """
    # Loop backward from the break index to the swing low index
    for i in range(break_index - 1, swing_low_index, -1):
        # Check if it's a bullish candle (close > open)
        if df['close'].iloc[i] > df['open'].iloc[i]:
            return i

    return -1  # No suitable candle found

def find_mitigation_point(df, ob_index, start_idx, end_idx, is_bullish=True):
    """
    Find the point where an order block gets mitigated (price CLOSES inside or beyond the zone)
    """
    ob_high = df['high'].iloc[ob_index]
    ob_low = df['low'].iloc[ob_index]

    for i in range(start_idx, end_idx):
        if is_bullish:
            # For bullish order blocks, mitigation happens when a candle closes below/inside the zone
            if df['close'].iloc[i] <= ob_high:
                return i
        else:
            # For bearish order blocks, mitigation happens when a candle closes above/inside the zone
            if df['close'].iloc[i] >= ob_low:
                return i

    return end_idx  # No mitigation found

def detect_bos_events_optimized(df):
    """Optimized detection of Break of Structure (BOS) events and order blocks"""
    # Pre-extract numpy arrays for faster access
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_prices = df['open'].values
    is_swing_high = df['is_swing_high'].values
    is_swing_low = df['is_swing_low'].values

    # Pre-compute swing point indices for faster lookup
    swing_high_indices = np.where(is_swing_high == 1)[0]
    swing_low_indices = np.where(is_swing_low == 1)[0]

    # Dictionaries to store results
    bos_events = {'bullish': [], 'bearish': []}
    order_blocks = {'bullish': [], 'bearish': []}

    # Process bullish BOS events (more efficiently)
    for i in swing_high_indices:
        swing_high_val = high[i]

        # Find next swing high (if any)
        next_swing_idx = swing_high_indices[swing_high_indices > i]
        next_swing_idx = next_swing_idx[0] if len(next_swing_idx) > 0 else len(close)

        # Look for a close above this swing high
        for j in range(i + 1, next_swing_idx):
            if close[j] > swing_high_val:
                # Bullish BOS detected
                bos_events['bullish'].append((i, j))

                # Find the order block
                ob_index = -1
                for k in range(j - 1, i, -1):
                    if close[k] < open_prices[k]:  # Bearish candle
                        ob_index = k
                        break

                if ob_index != -1:
                    # Find mitigation
                    mitigation_index = len(close)
                    for k in range(j + 1, len(close)):
                        if close[k] <= high[ob_index]:
                            mitigation_index = k
                            break

                    order_blocks['bullish'].append((ob_index, i, j, mitigation_index))

                break

    # Process bearish BOS events (more efficiently)
    for i in swing_low_indices:
        swing_low_val = low[i]

        # Find next swing low (if any)
        next_swing_idx = swing_low_indices[swing_low_indices > i]
        next_swing_idx = next_swing_idx[0] if len(next_swing_idx) > 0 else len(close)

        # Look for a close below this swing low
        for j in range(i + 1, next_swing_idx):
            if close[j] < swing_low_val:
                # Bearish BOS detected
                bos_events['bearish'].append((i, j))

                # Find the order block
                ob_index = -1
                for k in range(j - 1, i, -1):
                    if close[k] > open_prices[k]:  # Bullish candle
                        ob_index = k
                        break

                if ob_index != -1:
                    # Find mitigation
                    mitigation_index = len(close)
                    for k in range(j + 1, len(close)):
                        if close[k] >= low[ob_index]:
                            mitigation_index = k
                            break

                    order_blocks['bearish'].append((ob_index, i, j, mitigation_index))

                break

    return bos_events, order_blocks

def add_order_block_features(df, order_blocks, atr=None):
    """Add order block features to the dataframe without lookahead bias"""
    # Initialize order block features (essential features only)
    df = df.copy()
    df['bullish_ob_present'] = 0
    df['bearish_ob_present'] = 0
    df['ob_bull_retests'] = 0
    df['ob_bear_retests'] = 0
    df['ob_bull_distance_pct'] = np.nan
    df['ob_bear_distance_pct'] = np.nan
    df['ob_bull_volatility_ratio'] = 0.0
    df['ob_bear_volatility_ratio'] = 0.0

    # Create arrays for tracking all active order blocks at each bar
    bull_obs_active = []  # List of active bullish OBs
    bear_obs_active = []  # List of active bearish OBs

    # Process each row chronologically to avoid lookahead bias
    for i in range(len(df)):
        # Check if any new order blocks form at this bar
        for ob_idx, _, _, _ in order_blocks['bullish']:
            if ob_idx == i:
                ob_high = df['high'].iloc[i]
                ob_low = df['low'].iloc[i]
                ob_size = ob_high - ob_low

                # Volatility ratio (normalize by ATR)
                vol_ratio = 0.0
                if atr is not None and i < len(atr):
                    vol_ratio = ob_size / atr[i]

                # Add to active OBs list with essential info only
                bull_obs_active.append({
                    'idx': i,
                    'high': ob_high,
                    'low': ob_low,
                    'vol_ratio': vol_ratio,
                    'retests': 0,
                    'mitigated': False
                })

                # Mark this candle as having a bullish OB
                df.loc[i, 'bullish_ob_present'] = 1
                df.loc[i, 'ob_bull_volatility_ratio'] = vol_ratio

        for ob_idx, _, _, _ in order_blocks['bearish']:
            if ob_idx == i:
                ob_high = df['high'].iloc[i]
                ob_low = df['low'].iloc[i]
                ob_size = ob_high - ob_low

                # Volatility ratio (normalize by ATR)
                vol_ratio = 0.0
                if atr is not None and i < len(atr):
                    vol_ratio = ob_size / atr[i]

                # Add to active OBs list with essential info only
                bear_obs_active.append({
                    'idx': i,
                    'high': ob_high,
                    'low': ob_low,
                    'vol_ratio': vol_ratio,
                    'retests': 0,
                    'mitigated': False
                })

                # Mark this candle as having a bearish OB
                df.loc[i, 'bearish_ob_present'] = 1
                df.loc[i, 'ob_bear_volatility_ratio'] = vol_ratio

        # Update active order blocks (retest counts, mitigation, etc.)
        if i > 0:  # Skip first bar
            # Process bullish OBs
            mitigated_bull_obs = []
            for j, ob in enumerate(bull_obs_active):
                # Calculate distance to OB (negative if inside)
                if df['close'].iloc[i] > ob['high']:
                    distance = (df['close'].iloc[i] - ob['high']) / df['close'].iloc[i] * 100
                elif df['close'].iloc[i] < ob['low']:
                    distance = (df['close'].iloc[i] - ob['low']) / df['close'].iloc[i] * 100
                else:
                    distance = 0.0  # Inside the OB

                # Check for new retest
                if df['low'].iloc[i] <= ob['high'] and df['high'].iloc[i] >= ob['low']:
                    bull_obs_active[j]['retests'] += 1

                # Check for mitigation (CLOSE inside or beyond the zone)
                if df['close'].iloc[i] <= ob['high'] and not ob['mitigated']:
                    bull_obs_active[j]['mitigated'] = True
                    mitigated_bull_obs.append(j)

            # Process bearish OBs
            mitigated_bear_obs = []
            for j, ob in enumerate(bear_obs_active):
                # Calculate distance to OB (negative if inside)
                if df['close'].iloc[i] > ob['high']:
                    distance = (df['close'].iloc[i] - ob['high']) / df['close'].iloc[i] * 100
                elif df['close'].iloc[i] < ob['low']:
                    distance = (df['close'].iloc[i] - ob['low']) / df['close'].iloc[i] * 100
                else:
                    distance = 0.0  # Inside the OB

                # Check for new retest
                if df['high'].iloc[i] >= ob['low'] and df['low'].iloc[i] <= ob['high']:
                    bear_obs_active[j]['retests'] += 1

                # Check for mitigation (CLOSE inside or beyond the zone)
                if df['close'].iloc[i] >= ob['low'] and not ob['mitigated']:
                    bear_obs_active[j]['mitigated'] = True
                    mitigated_bear_obs.append(j)

            # Remove mitigated OBs (in reverse order to avoid index issues)
            for j in sorted(mitigated_bull_obs, reverse=True):
                bull_obs_active.pop(j)

            for j in sorted(mitigated_bear_obs, reverse=True):
                bear_obs_active.pop(j)

        # Set current features based on closest OB
        if bull_obs_active:
            # Calculate distances for all active OBs
            distances = []
            for ob in bull_obs_active:
                if df['close'].iloc[i] > ob['high']:
                    distance = (df['close'].iloc[i] - ob['high']) / df['close'].iloc[i] * 100
                elif df['close'].iloc[i] < ob['low']:
                    distance = (df['close'].iloc[i] - ob['low']) / df['close'].iloc[i] * 100
                else:
                    distance = 0.0  # Inside the OB
                distances.append((distance, ob))

            # Find closest OB by absolute distance
            closest_ob = min(distances, key=lambda x: abs(x[0]) if x[0] is not None else float('inf'))
            df.loc[i, 'ob_bull_distance_pct'] = closest_ob[0]
            df.loc[i, 'ob_bull_retests'] = closest_ob[1]['retests']
            df.loc[i, 'ob_bull_volatility_ratio'] = closest_ob[1]['vol_ratio']

        if bear_obs_active:
            # Calculate distances for all active OBs
            distances = []
            for ob in bear_obs_active:
                if df['close'].iloc[i] > ob['high']:
                    distance = (df['close'].iloc[i] - ob['high']) / df['close'].iloc[i] * 100
                elif df['close'].iloc[i] < ob['low']:
                    distance = (df['close'].iloc[i] - ob['low']) / df['close'].iloc[i] * 100
                else:
                    distance = 0.0  # Inside the OB
                distances.append((distance, ob))

            # Find closest OB by absolute distance
            closest_ob = min(distances, key=lambda x: abs(x[0]) if x[0] is not None else float('inf'))
            df.loc[i, 'ob_bear_distance_pct'] = closest_ob[0]
            df.loc[i, 'ob_bear_retests'] = closest_ob[1]['retests']
            df.loc[i, 'ob_bear_volatility_ratio'] = closest_ob[1]['vol_ratio']

    # Add trend context
    if 'trend' not in df.columns and len(df) >= 50:
        df['ma50'] = df['close'].rolling(50).mean()
        df['trend'] = np.where(df['close'] > df['ma50'], 1, -1)

        # Order blocks in context of trend (without lookahead)
        df['ob_bull_with_trend'] = df['bullish_ob_present'] * (df['trend'] == 1)
        df['ob_bear_with_trend'] = df['bearish_ob_present'] * (df['trend'] == -1)

    # Add volatility regime awareness
    if atr is not None:
        df['atr_z_score'] = (df['atr'] - df['atr'].rolling(100).mean()) / df['atr'].rolling(100).std()
        df['high_volatility'] = (df['atr'] > df['atr'].rolling(100).mean()).astype(int)

    return df