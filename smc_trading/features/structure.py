"""Market structure analysis and advanced SMC features"""
import numpy as np
import pandas as pd

def add_advanced_smc_features_optimized(df, tops, bottoms, bos_events, order_blocks, fvg_events, atr=None):
    """
    Optimized version of add_advanced_smc_features using vectorized operations
    """
    # Initialize new feature columns with numpy arrays
    df = df.copy()
    df_len = len(df)
    structure_trend = np.zeros(df_len)
    swing_high_distance = np.full(df_len, np.nan)
    swing_low_distance = np.full(df_len, np.nan)
    bos_confirmation = np.zeros(df_len)
    ob_fvg_confluence = np.zeros(df_len)
    liquidity_zone = np.full(df_len, np.nan)
    pattern_freshness = np.zeros(df_len)
    recent_volatility = np.zeros(df_len)

    # Pre-extract arrays for faster operations
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    # Pre-process swing points for faster lookup
    top_indices = np.array([max_idx for _, max_idx, _ in tops])
    top_prices = np.array([price for _, _, price in tops])
    bottom_indices = np.array([min_idx for _, min_idx, _ in bottoms])
    bottom_prices = np.array([price for _, _, price in bottoms])

    # Set pip value based on instrument
    instrument_digits = len(str(close[0]).split('.')[-1]) if '.' in str(close[0]) else 0
    pip_value = 10 ** (-instrument_digits)

    # Calculate ATR statistics for normalization (if provided)
    if atr is not None:
        atr_std = np.std(atr) if np.std(atr) > 0 else 1.0
        atr_mean = np.mean(atr)
    else:
        atr_std = 1.0
        atr_mean = 0.0

    # Temporary arrays to track market structure
    recent_tops = []  # [(index, price)]
    recent_bottoms = []  # [(index, price)]
    bos_confirmations = []  # [(index, direction)]

    # Pre-calculate active patterns for each bar
    active_bull_obs = np.zeros(df_len, dtype=int)
    active_bear_obs = np.zeros(df_len, dtype=int)
    active_bull_fvgs = np.zeros(df_len, dtype=int)
    active_bear_fvgs = np.zeros(df_len, dtype=int)

    # Pre-process BOS events for faster lookup
    bullish_bos_breaks = [break_idx for _, break_idx in bos_events['bullish']]
    bearish_bos_breaks = [break_idx for _, break_idx in bos_events['bearish']]

    # Mark active order blocks
    for ob_idx, _, _, mitigation_idx in order_blocks['bullish']:
        end_idx = min(mitigation_idx, df_len)
        active_bull_obs[ob_idx:end_idx] += 1

    for ob_idx, _, _, mitigation_idx in order_blocks['bearish']:
        end_idx = min(mitigation_idx, df_len)
        active_bear_obs[ob_idx:end_idx] += 1

    # Mark active FVGs
    for start_idx, _, _, _, mitigated, mitigation_idx in fvg_events['bullish']:
        mid_idx = start_idx + 1
        if mid_idx < df_len:
            end_idx = df_len if not mitigated else (mitigation_idx if mitigation_idx is not None else df_len)
            end_idx = min(end_idx, df_len)
            active_bull_fvgs[mid_idx:end_idx] += 1

    for start_idx, _, _, _, mitigated, mitigation_idx in fvg_events['bearish']:
        mid_idx = start_idx + 1
        if mid_idx < df_len:
            end_idx = df_len if not mitigated else (mitigation_idx if mitigation_idx is not None else df_len)
            end_idx = min(end_idx, df_len)
            active_bear_fvgs[mid_idx:end_idx] += 1

    # Calculate confluence (total active patterns at each bar)
    ob_fvg_confluence = active_bull_obs + active_bear_obs + active_bull_fvgs + active_bear_fvgs

    # Process each bar for features that can't be fully vectorized
    for i in range(df_len):
        # Update swing points tracking
        for _, max_idx, price in tops:
            if max_idx == i:
                recent_tops.append((i, price))

        for _, min_idx, price in bottoms:
            if min_idx == i:
                recent_bottoms.append((i, price))

        # Structure trend calculation with Fibonacci retracement
        if len(recent_tops) >= 3 and len(recent_bottoms) >= 3:
            last_high = recent_tops[-1][1]
            prev_high = recent_tops[-2][1]
            last_low = recent_bottoms[-1][1]
            prev_low = recent_bottoms[-2][1]

            bull_retrace = (last_high - last_low) * 0.618
            bear_retrace = (last_high - last_low) * 0.618

            current_close = close[i]

            if last_high > prev_high and last_low > prev_low and current_close > (last_low + bull_retrace):
                structure_trend[i] = 1
            elif last_high < prev_high and last_low < prev_low and current_close < (last_high - bear_retrace):
                structure_trend[i] = -1
            else:
                structure_trend[i] = 0

        # Calculate distance to nearest swing points
        if recent_tops:
            closest_high_dist = float('inf')
            for idx, price in recent_tops:
                if idx < i:  # Only consider past swing highs
                    dist = abs(price - close[i]) / pip_value
                    closest_high_dist = min(closest_high_dist, dist)
            if closest_high_dist != float('inf'):
                swing_high_distance[i] = closest_high_dist

        if recent_bottoms:
            closest_low_dist = float('inf')
            for idx, price in recent_bottoms:
                if idx < i:  # Only consider past swing lows
                    dist = abs(price - close[i]) / pip_value
                    closest_low_dist = min(closest_low_dist, dist)
            if closest_low_dist != float('inf'):
                swing_low_distance[i] = closest_low_dist

        # BOS Confirmation
        if i in bullish_bos_breaks:
            bos_confirmations.append((i, 1))  # Bullish BOS
        if i in bearish_bos_breaks:
            bos_confirmations.append((i, -1))  # Bearish BOS

        if bos_confirmations:
            most_recent_bos = bos_confirmations[-1]
            if i - most_recent_bos[0] < 10:  # Recent BOS (within 10 bars)
                bos_confirmation[i] = most_recent_bos[1]

        # Set confluence from pre-calculated arrays
        ob_fvg_confluence[i] = active_bull_obs[i] + active_bear_obs[i] + active_bull_fvgs[i] + active_bear_fvgs[i]

        # Liquidity Zone Detection
        liquidity_points = []

        # Add recent swing highs as liquidity zones
        for idx, price in recent_tops:
            if i - idx < 50 and idx < i:  # Recent swing points only
                # Count touches near this level
                touches = np.sum(np.abs(high[:i] - price) < 5 * pip_value)
                if touches >= 3:
                    liquidity_points.append((price, touches))

        # Add recent swing lows as liquidity zones
        for idx, price in recent_bottoms:
            if i - idx < 50 and idx < i:  # Recent swing points only
                # Count touches near this level
                touches = np.sum(np.abs(low[:i] - price) < 5 * pip_value)
                if touches >= 3:
                    liquidity_points.append((price, touches))

        # Find distance to significant liquidity zones
        if liquidity_points:
            # Sort by importance (number of touches)
            liquidity_points.sort(key=lambda x: x[1], reverse=True)
            # Take top 3 most significant points
            top_liquidity = liquidity_points[:min(3, len(liquidity_points))]
            # Find distance to nearest
            if top_liquidity:
                nearest_dist = min(abs(price - close[i]) for price, _ in top_liquidity)
                liquidity_zone[i] = nearest_dist / pip_value

        # Pattern Freshness
        pattern_ages = []
        pattern_weights = []

        # Calculate weighted avg age of active patterns at this bar
        if active_bull_obs[i] > 0 or active_bear_obs[i] > 0 or active_bull_fvgs[i] > 0 or active_bear_fvgs[i] > 0:
            # Bullish OBs
            for ob_idx, _, _, mitigation_idx in order_blocks['bullish']:
                if ob_idx < i and (mitigation_idx > i or mitigation_idx == df_len):
                    age = i - ob_idx
                    weight = np.exp(-0.1 * age)  # Exponential decay
                    pattern_ages.append(age)
                    pattern_weights.append(weight)

            # Bearish OBs
            for ob_idx, _, _, mitigation_idx in order_blocks['bearish']:
                if ob_idx < i and (mitigation_idx > i or mitigation_idx == df_len):
                    age = i - ob_idx
                    weight = np.exp(-0.1 * age)
                    pattern_ages.append(age)
                    pattern_weights.append(weight)

            # Bullish FVGs
            for start_idx, _, _, _, mitigated, mitigation_idx in fvg_events['bullish']:
                mid_idx = start_idx + 1
                if mid_idx < i and (not mitigated or (mitigation_idx is not None and mitigation_idx > i)):
                    age = i - mid_idx
                    weight = np.exp(-0.1 * age)
                    pattern_ages.append(age)
                    pattern_weights.append(weight)

            # Bearish FVGs
            for start_idx, _, _, _, mitigated, mitigation_idx in fvg_events['bearish']:
                mid_idx = start_idx + 1
                if mid_idx < i and (not mitigated or (mitigation_idx is not None and mitigation_idx > i)):
                    age = i - mid_idx
                    weight = np.exp(-0.1 * age)
                    pattern_ages.append(age)
                    pattern_weights.append(weight)

            # Calculate weighted average age and apply sigmoid transformation
            if pattern_weights:
                avg_age = np.average(pattern_ages, weights=pattern_weights) if sum(pattern_weights) > 0 else 0
                # Sigmoid function for smoother decay between 0 and 1
                pattern_freshness[i] = 1.0 / (1.0 + np.exp(0.2 * (avg_age - 10)))

        # Recent Volatility Context
        if atr is not None and i >= 10:
            # Track volatility differences between pattern formation and current
            volatility_zscores = []

            # Active OBs and FVGs
            for ob_idx, _, _, mitigation_idx in order_blocks['bullish'] + order_blocks['bearish']:
                if ob_idx < i and ob_idx >= 10 and (mitigation_idx > i or mitigation_idx == df_len):
                    formation_atr = atr[ob_idx]
                    current_atr = atr[i]
                    if atr_std > 0:
                        vol_zscore = (current_atr - formation_atr) / atr_std
                        volatility_zscores.append(vol_zscore)

            for start_idx, _, _, _, mitigated, mitigation_idx in fvg_events['bullish'] + fvg_events['bearish']:
                mid_idx = start_idx + 1
                if mid_idx < i and mid_idx >= 10 and (
                        not mitigated or (mitigation_idx is not None and mitigation_idx > i)):
                    formation_atr = atr[mid_idx]
                    current_atr = atr[i]
                    if atr_std > 0:
                        vol_zscore = (current_atr - formation_atr) / atr_std
                        volatility_zscores.append(vol_zscore)

            # Calculate average volatility z-score
            if volatility_zscores:
                recent_volatility[i] = np.mean(volatility_zscores)

    # Assign calculated values to dataframe
    df['structure_trend'] = structure_trend
    df['swing_high_distance'] = swing_high_distance
    df['swing_low_distance'] = swing_low_distance
    df['bos_confirmation'] = bos_confirmation
    df['ob_fvg_confluence'] = ob_fvg_confluence
    df['liquidity_zone'] = liquidity_zone
    df['pattern_freshness'] = pattern_freshness
    df['recent_volatility'] = recent_volatility

    # Add multi-timeframe confirmation - optimized
    if len(df) >= 200:
        # Calculate EMAs using vectorized operations
        ema20 = df['close'].ewm(span=20, adjust=False).mean().values
        ema100 = df['close'].ewm(span=100, adjust=False).mean().values

        # Calculate slopes
        ema20_slope = np.zeros_like(ema20)
        ema20_slope[5:] = (ema20[5:] - ema20[:-5]) / 5

        ema100_slope = np.zeros_like(ema100)
        ema100_slope[20:] = (ema100[20:] - ema100[:-20]) / 20

        # Calculate multi-timeframe confirmation
        multi_timeframe_confirm = np.zeros(df_len)

        # Bullish confirmation mask
        bull_mask = (ema20 > ema100) & (ema20_slope > 0) & (structure_trend == 1)
        multi_timeframe_confirm[bull_mask] = 1

        # Bearish confirmation mask
        bear_mask = (ema20 < ema100) & (ema20_slope < 0) & (structure_trend == -1)
        multi_timeframe_confirm[bear_mask] = -1

        # Store in dataframe
        df['ema20'] = ema20
        df['ema100'] = ema100
        df['ema20_slope'] = ema20_slope
        df['ema100_slope'] = ema100_slope
        df['multi_timeframe_confirm'] = multi_timeframe_confirm
    else:
        df['multi_timeframe_confirm'] = 0

    # Shift features to prevent lookahead bias
    shift_features = [
        'structure_trend', 'bos_confirmation', 'ob_fvg_confluence',
        'pattern_freshness', 'recent_volatility', 'multi_timeframe_confirm'
    ]

    for feature in shift_features:
        if feature in df.columns:
            df[feature] = df[feature].shift(1)

    return df