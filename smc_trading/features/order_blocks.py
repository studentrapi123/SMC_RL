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

def calculate_ob_quality(df, i, ob_size, ob_body_size, atr_value=None):
    """
    Calculate a quality score (0-1) for an order block based on multiple metrics.

    Args:
        df: DataFrame with price data
        i: Current bar index
        ob_size: Size of the order block (high - low)
        ob_body_size: Size of the candle body
        atr_value: ATR value at this bar (if available)

    Returns:
        float: Quality score between 0 and 1
    """
    # 1. Candle size relative to recent candles (10-bar lookback)
    recent_range = df['high'].iloc[max(0, i - 10):i + 1].max() - df['low'].iloc[max(0, i - 10):i + 1].min()
    relative_size = ob_size / recent_range if recent_range > 0 else 0

    # 2. Body to wick ratio (larger body is better for OB)
    body_ratio = ob_body_size / ob_size if ob_size > 0 else 0

    # 3. Volume relative to recent volume (if volume data available)
    vol_factor = 1.0
    if 'volume' in df.columns or 'tick_volume' in df.columns:
        vol_col = 'volume' if 'volume' in df.columns else 'tick_volume'
        recent_avg_vol = df[vol_col].iloc[max(0, i - 10):i].mean()
        if recent_avg_vol > 0:
            vol_factor = min(2.0, df[vol_col].iloc[i] / recent_avg_vol) / 2.0

    # 4. Volatility ratio (normalize by ATR)
    vol_ratio = 0.0
    if atr_value is not None and atr_value > 0:
        vol_ratio = min(2.0, ob_size / atr_value) / 2.0

    # Combine metrics for overall quality score (0-1)
    # Weight the components based on importance
    quality = min(1.0, (0.3 * relative_size + 0.3 * body_ratio + 0.2 * vol_factor + 0.2 * vol_ratio))

    return quality

def is_clean_ob(df, ob_idx, break_idx, is_bullish=True):
    """
    Check if an order block is "clean" - had minimal price interaction before breakout.

    Args:
        df: DataFrame with price data
        ob_idx: Index of the order block
        break_idx: Index of the breakout bar
        is_bullish: True for bullish OB, False for bearish OB

    Returns:
        int: 1 if clean, 0 if not
    """
    if ob_idx >= break_idx or break_idx >= len(df):
        return 0

    ob_high = df['high'].iloc[ob_idx]
    ob_low = df['low'].iloc[ob_idx]

    # Get prices between OB and breakout
    prices_between = df.iloc[ob_idx + 1:break_idx]

    if is_bullish:
        # For bullish OB, price should stay below the OB zone before breakout
        if len(prices_between) > 0 and (prices_between['close'] < ob_low).all():
            return 1
    else:
        # For bearish OB, price should stay above the OB zone before breakout
        if len(prices_between) > 0 and (prices_between['close'] > ob_high).all():
            return 1

    return 0

def process_new_order_blocks(df, i, order_blocks, bull_obs_active, bear_obs_active, atr=None):
    """
    Process new order blocks forming at the current bar.

    Args:
        df: DataFrame with price data
        i: Current bar index
        order_blocks: Dictionary with bullish and bearish order blocks
        bull_obs_active: List of active bullish OBs
        bear_obs_active: List of active bearish OBs
        atr: ATR array or single value (if available)

    Returns:
        tuple: Updated bull_obs_active and bear_obs_active lists
    """
    # Get ATR value for this bar (handling both array and scalar cases)
    atr_value = None
    if atr is not None:
        if isinstance(atr, np.ndarray) and i < len(atr):
            atr_value = atr[i]
        else:
            # Handle case where atr is a scalar value
            atr_value = atr

    # Check for new bullish order blocks
    for ob_idx, swing_idx, break_idx, _ in order_blocks['bullish']:
        if ob_idx == i:
            ob_high = df['high'].iloc[i]
            ob_low = df['low'].iloc[i]
            ob_size = ob_high - ob_low
            ob_body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])

            # Calculate volatility ratio
            vol_ratio = 0.0
            if atr_value is not None and atr_value > 0:
                vol_ratio = ob_size / atr_value

            # Calculate quality score
            quality = calculate_ob_quality(df, i, ob_size, ob_body_size, atr_value)

            # Check if it's a clean OB
            clean = is_clean_ob(df, ob_idx, break_idx, is_bullish=True)

            # Add to active OBs list
            bull_obs_active.append({
                'idx': i,
                'high': ob_high,
                'low': ob_low,
                'vol_ratio': vol_ratio,
                'retests': 0,
                'mitigated': False,
                'quality': quality,
                'clean': clean,
                'creation_time': i,
                'age': 0
            })

            # Mark this candle with OB properties
            df.loc[i, 'bullish_ob_present'] = 1
            df.loc[i, 'ob_bull_volatility_ratio'] = vol_ratio
            df.loc[i, 'ob_bull_quality'] = quality
            df.loc[i, 'ob_bull_clean'] = clean
            df.loc[i, 'ob_bull_age'] = 0

    # Check for new bearish order blocks
    for ob_idx, swing_idx, break_idx, _ in order_blocks['bearish']:
        if ob_idx == i:
            ob_high = df['high'].iloc[i]
            ob_low = df['low'].iloc[i]
            ob_size = ob_high - ob_low
            ob_body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])

            # Calculate volatility ratio
            vol_ratio = 0.0
            if atr_value is not None and atr_value > 0:
                vol_ratio = ob_size / atr_value

            # Calculate quality score
            quality = calculate_ob_quality(df, i, ob_size, ob_body_size, atr_value)

            # Check if it's a clean OB
            clean = is_clean_ob(df, ob_idx, break_idx, is_bullish=False)

            # Add to active OBs list
            bear_obs_active.append({
                'idx': i,
                'high': ob_high,
                'low': ob_low,
                'vol_ratio': vol_ratio,
                'retests': 0,
                'mitigated': False,
                'quality': quality,
                'clean': clean,
                'creation_time': i,
                'age': 0
            })

            # Mark this candle with OB properties
            df.loc[i, 'bearish_ob_present'] = 1
            df.loc[i, 'ob_bear_volatility_ratio'] = vol_ratio
            df.loc[i, 'ob_bear_quality'] = quality
            df.loc[i, 'ob_bear_clean'] = clean
            df.loc[i, 'ob_bear_age'] = 0

    return bull_obs_active, bear_obs_active

def update_active_order_blocks(df, i, bull_obs_active, bear_obs_active):
    """
    Update existing active order blocks (retests, mitigation, age).

    Args:
        df: DataFrame with price data
        i: Current bar index
        bull_obs_active: List of active bullish OBs
        bear_obs_active: List of active bearish OBs

    Returns:
        tuple: Updated lists and indexes of mitigated OBs
    """
    mitigated_bull_obs = []
    mitigated_bear_obs = []

    # Update bullish OBs
    for j, ob in enumerate(bull_obs_active):
        # Check for new retest
        if df['low'].iloc[i] <= ob['high'] and df['high'].iloc[i] >= ob['low']:
            bull_obs_active[j]['retests'] += 1

        # Check for mitigation (CLOSE inside or beyond the zone)
        if df['close'].iloc[i] <= ob['high'] and not ob['mitigated']:
            bull_obs_active[j]['mitigated'] = True
            mitigated_bull_obs.append(j)

        # Update age
        bull_obs_active[j]['age'] = i - ob['creation_time']

    # Update bearish OBs
    for j, ob in enumerate(bear_obs_active):
        # Check for new retest
        if df['high'].iloc[i] >= ob['low'] and df['low'].iloc[i] <= ob['high']:
            bear_obs_active[j]['retests'] += 1

        # Check for mitigation (CLOSE inside or beyond the zone)
        if df['close'].iloc[i] >= ob['low'] and not ob['mitigated']:
            bear_obs_active[j]['mitigated'] = True
            mitigated_bear_obs.append(j)

        # Update age
        bear_obs_active[j]['age'] = i - ob['creation_time']

    return bull_obs_active, bear_obs_active, mitigated_bull_obs, mitigated_bear_obs

def calculate_ob_distance(df, i, ob):
    """
    Calculate distance from current price to order block (in percent).
    """
    current_price = df['close'].iloc[i]

    if current_price > ob['high']:
        return (current_price - ob['high']) / current_price * 100
    elif current_price < ob['low']:
        return (current_price - ob['low']) / current_price * 100
    else:
        return 0.0  # Inside the OB

def update_ob_features(df, i, bull_obs_active, bear_obs_active):
    """
    Update order block feature columns in the dataframe.
    """
    # Update bullish OB features
    if bull_obs_active:
        # Calculate distances for all active OBs
        distances = []
        for ob in bull_obs_active:
            distance = calculate_ob_distance(df, i, ob)
            # Explicitly make sure the distance is a float
            distances.append((float(distance), ob))

        if distances:
            # Find closest OB by absolute distance
            closest_ob = min(distances, key=lambda x: abs(x[0]))
            # Directly assign the distance value
            df.at[i, 'ob_bull_distance_pct'] = closest_ob[0]
            df.at[i, 'ob_bull_retests'] = closest_ob[1]['retests']
            df.at[i, 'ob_bull_volatility_ratio'] = closest_ob[1]['vol_ratio']
            if 'quality' in closest_ob[1]:
                df.at[i, 'ob_bull_quality'] = closest_ob[1]['quality']
            if 'clean' in closest_ob[1]:
                df.at[i, 'ob_bull_clean'] = closest_ob[1]['clean']
            if 'age' in closest_ob[1]:
                df.at[i, 'ob_bull_age'] = closest_ob[1]['age']

    # Update bearish OB features
    if bear_obs_active:
        # Calculate distances for all active OBs
        distances = []
        for ob in bear_obs_active:
            distance = calculate_ob_distance(df, i, ob)
            # Explicitly make sure the distance is a float
            distances.append((float(distance), ob))

        if distances:
            # Find closest OB by absolute distance
            closest_ob = min(distances, key=lambda x: abs(x[0]))
            # Directly assign the distance value
            df.at[i, 'ob_bear_distance_pct'] = closest_ob[0]
            df.at[i, 'ob_bear_retests'] = closest_ob[1]['retests']
            df.at[i, 'ob_bear_volatility_ratio'] = closest_ob[1]['vol_ratio']
            if 'quality' in closest_ob[1]:
                df.at[i, 'ob_bear_quality'] = closest_ob[1]['quality']
            if 'clean' in closest_ob[1]:
                df.at[i, 'ob_bear_clean'] = closest_ob[1]['clean']
            if 'age' in closest_ob[1]:
                df.at[i, 'ob_bear_age'] = closest_ob[1]['age']

def add_trend_context(df, atr=None):
    """
    Add trend context and volatility awareness to the dataframe.

    Args:
        df: DataFrame with price data
        atr: ATR array or single value (if available)

    Returns:
        DataFrame: Updated dataframe with trend features
    """
    # Add trend context
    if 'trend' not in df.columns and len(df) >= 50:
        df['ma50'] = df['close'].rolling(50).mean()
        df['trend'] = np.where(df['close'] > df['ma50'], 1, -1)

        # Order blocks in context of trend
        df['ob_bull_with_trend'] = df['bullish_ob_present'] * (df['trend'] == 1)
        df['ob_bear_with_trend'] = df['bearish_ob_present'] * (df['trend'] == -1)

    # Add volatility regime awareness
    if atr is not None and 'atr' in df.columns:
        df['atr_z_score'] = (df['atr'] - df['atr'].rolling(100).mean()) / df['atr'].rolling(100).std()
        df['high_volatility'] = (df['atr'] > df['atr'].rolling(100).mean()).astype(int)

    return df

def add_order_block_features(df, order_blocks, atr=None):
    """
    Main function to add order block features to the dataframe without lookahead bias.

    Args:
        df: DataFrame with price data
        order_blocks: Dictionary with bullish and bearish order blocks
        atr: ATR array (if available)

    Returns:
        DataFrame: Updated dataframe with order block features
    """
    # Initialize order block features
    df = df.copy()

    # Basic OB features
    df['bullish_ob_present'] = 0
    df['bearish_ob_present'] = 0
    df['ob_bull_retests'] = 0
    df['ob_bear_retests'] = 0
    df['ob_bull_distance_pct'] = np.nan
    df['ob_bear_distance_pct'] = np.nan
    df['ob_bull_volatility_ratio'] = 0.0
    df['ob_bear_volatility_ratio'] = 0.0

    # New quality features
    df['ob_bull_quality'] = 0.0
    df['ob_bear_quality'] = 0.0
    df['ob_bull_clean'] = 0
    df['ob_bear_clean'] = 0
    df['ob_bull_age'] = np.nan
    df['ob_bear_age'] = np.nan

    # Create arrays for tracking all active order blocks at each bar
    bull_obs_active = []  # List of active bullish OBs
    bear_obs_active = []  # List of active bearish OBs

    # Process each row chronologically to avoid lookahead bias
    for i in range(len(df)):
        # Check for new order blocks at this bar
        bull_obs_active, bear_obs_active = process_new_order_blocks(
            df, i, order_blocks, bull_obs_active, bear_obs_active, atr)

        # Update existing active order blocks
        if i > 0:  # Skip first bar
            bull_obs_active, bear_obs_active, mitigated_bull_obs, mitigated_bear_obs = update_active_order_blocks(
                df, i, bull_obs_active, bear_obs_active)

            # Remove mitigated OBs (in reverse order to avoid index issues)
            for j in sorted(mitigated_bull_obs, reverse=True):
                bull_obs_active.pop(j)

            for j in sorted(mitigated_bear_obs, reverse=True):
                bear_obs_active.pop(j)

        # Update feature columns based on active OBs
        update_ob_features(df, i, bull_obs_active, bear_obs_active)

    # Add trend context and volatility awareness
    df = add_trend_context(df, atr)

    return df