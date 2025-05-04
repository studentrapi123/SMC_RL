"""Enhanced Order Block detection and feature extraction"""
import numpy as np
import pandas as pd


class OrderBlock:
    """Class representing a single order block with all its properties and state"""
    def __init__(self, idx, high, low, ob_type, swing_idx, break_idx, quality=0.0, clean=0, vol_ratio=0.0):
        # Basic properties
        self.idx = idx                  # Index where this OB formed
        self.high = high                # High price of the OB
        self.low = low                  # Low price of the OB
        self.ob_type = ob_type          # "bullish" or "bearish"
        self.swing_idx = swing_idx      # Related swing point
        self.break_idx = break_idx      # BOS break index

        # Derived properties
        self.creation_time = idx
        self.quality = quality
        self.clean = clean
        self.vol_ratio = vol_ratio

        # State properties (updated over time)
        self.mitigated = False
        self.mitigation_idx = None
        self.retests = 0
        self.age = 0

    def calculate_distance(self, current_price):
        """Calculate distance from current price to order block (in percent)"""
        if current_price > self.high:
            return (current_price - self.high) / current_price * 100
        elif current_price < self.low:
            return (current_price - self.low) / current_price * 100
        else:
            return 0.0  # Inside the OB

    def is_active(self, current_idx):
        """Check if this order block is still active"""
        if self.mitigated:
            return False
        return True

    def update(self, current_idx, current_price, current_high, current_low):
        """Update the order block state based on new price data"""
        # Update age
        self.age = current_idx - self.creation_time

        # Check for retest (price touching the zone)
        if (self.ob_type == "bullish" and
            current_low <= self.high and current_high >= self.low):
            self.retests += 1

        if (self.ob_type == "bearish" and
            current_high >= self.low and current_low <= self.high):
            self.retests += 1

        # Check for mitigation
        if not self.mitigated:
            if (self.ob_type == "bullish" and current_price <= self.high):
                self.mitigated = True
                self.mitigation_idx = current_idx

            if (self.ob_type == "bearish" and current_price >= self.low):
                self.mitigated = True
                self.mitigation_idx = current_idx


class OrderBlockManager:
    """Class for managing and tracking all active order blocks"""
    def __init__(self):
        self.all_blocks = []
        self.active_bull_blocks = []
        self.active_bear_blocks = []

    def add_order_block(self, order_block):
        """Add a new order block to tracking"""
        self.all_blocks.append(order_block)
        if order_block.ob_type == "bullish":
            self.active_bull_blocks.append(order_block)
        else:
            self.active_bear_blocks.append(order_block)

    def update_all_blocks(self, idx, price, high, low):
        """Update all active order blocks with new data"""
        # Update all bullish OBs
        for ob in self.active_bull_blocks[:]:  # Copy to avoid modification during iteration
            ob.update(idx, price, high, low)
            if not ob.is_active(idx):
                self.active_bull_blocks.remove(ob)

        # Update all bearish OBs
        for ob in self.active_bear_blocks[:]:  # Copy to avoid modification during iteration
            ob.update(idx, price, high, low)
            if not ob.is_active(idx):
                self.active_bear_blocks.remove(ob)

    def get_closest_bull_block(self, price):
        """Get the bullish order block closest to current price"""
        if not self.active_bull_blocks:
            return None

        closest_block = min(self.active_bull_blocks,
                           key=lambda ob: abs(ob.calculate_distance(price)))
        return closest_block

    def get_closest_bear_block(self, price):
        """Get the bearish order block closest to current price"""
        if not self.active_bear_blocks:
            return None

        closest_block = min(self.active_bear_blocks,
                           key=lambda ob: abs(ob.calculate_distance(price)))
        return closest_block

    def get_feature_summary(self, price):
        """Create a feature summary from all active blocks"""
        features = {
            'bull_ob_count': len(self.active_bull_blocks),
            'bear_ob_count': len(self.active_bear_blocks),
            'bull_ob_present': 1 if self.active_bull_blocks else 0,
            'bear_ob_present': 1 if self.active_bear_blocks else 0,
            'bull_distance_pct': None,
            'bear_distance_pct': None,
            'bull_retests_avg': None,
            'bear_retests_avg': None,
            'bull_quality_avg': None,
            'bear_quality_avg': None,
            'bull_age_avg': None,
            'bear_age_avg': None,
            'bull_closest_quality': None,
            'bear_closest_quality': None,
            'bull_closest_retests': None,
            'bear_closest_retests': None,
            'bull_closest_clean': None,
            'bear_closest_clean': None,
            'bull_closest_vol_ratio': None,
            'bear_closest_vol_ratio': None,
        }

        # Get closest blocks
        closest_bull = self.get_closest_bull_block(price)
        closest_bear = self.get_closest_bear_block(price)

        # Add closest block features
        if closest_bull:
            features['bull_distance_pct'] = closest_bull.calculate_distance(price)
            features['bull_closest_quality'] = closest_bull.quality
            features['bull_closest_retests'] = closest_bull.retests
            features['bull_closest_clean'] = closest_bull.clean
            features['bull_closest_vol_ratio'] = closest_bull.vol_ratio

        if closest_bear:
            features['bear_distance_pct'] = closest_bear.calculate_distance(price)
            features['bear_closest_quality'] = closest_bear.quality
            features['bear_closest_retests'] = closest_bear.retests
            features['bear_closest_clean'] = closest_bear.clean
            features['bear_closest_vol_ratio'] = closest_bear.vol_ratio

        # Calculate average metrics
        if self.active_bull_blocks:
            features['bull_retests_avg'] = sum(ob.retests for ob in self.active_bull_blocks) / len(self.active_bull_blocks)
            features['bull_quality_avg'] = sum(ob.quality for ob in self.active_bull_blocks) / len(self.active_bull_blocks)
            features['bull_age_avg'] = sum(ob.age for ob in self.active_bull_blocks) / len(self.active_bull_blocks)

        if self.active_bear_blocks:
            features['bear_retests_avg'] = sum(ob.retests for ob in self.active_bear_blocks) / len(self.active_bear_blocks)
            features['bear_quality_avg'] = sum(ob.quality for ob in self.active_bear_blocks) / len(self.active_bear_blocks)
            features['bear_age_avg'] = sum(ob.age for ob in self.active_bear_blocks) / len(self.active_bear_blocks)

        return features


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
    vol_ratio = 0.5  # Default mid-value
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


def add_order_block_features(df, bos_events, atr=None):
    """
    Enhanced function to add order block features to the dataframe.

    This version properly tracks all active order blocks at each time.

    Args:
        df: DataFrame with price data
        bos_events: Dictionary with bullish and bearish BOS events
        atr: ATR array (if available)

    Returns:
        DataFrame: Updated dataframe with order block features
    """
    df = df.copy()

    # Create an OrderBlockManager to track all blocks
    ob_manager = OrderBlockManager()

    # Pre-process all order blocks from BOS events
    bull_order_blocks = []
    bear_order_blocks = []

    # Find order blocks from BOS events
    for swing_idx, break_idx in bos_events['bullish']:
        ob_idx = find_order_block_for_bullish_bos(df, swing_idx, break_idx)
        if ob_idx != -1:
            bull_order_blocks.append((ob_idx, swing_idx, break_idx))

    for swing_idx, break_idx in bos_events['bearish']:
        ob_idx = find_order_block_for_bearish_bos(df, swing_idx, break_idx)
        if ob_idx != -1:
            bear_order_blocks.append((ob_idx, swing_idx, break_idx))

    # Initialize feature columns
    feature_columns = [
        'bullish_ob_present', 'bearish_ob_present',
        'bull_ob_count', 'bear_ob_count',
        'ob_bull_distance_pct', 'ob_bear_distance_pct',
        'ob_bull_retests', 'ob_bear_retests',
        'ob_bull_volatility_ratio', 'ob_bear_volatility_ratio',
        'ob_bull_quality', 'ob_bear_quality',
        'ob_bull_clean', 'ob_bear_clean',
        'ob_bull_age', 'ob_bear_age',
        'ob_bull_avg_quality', 'ob_bear_avg_quality'
    ]

    # Initialize columns with appropriate NaN or 0 values
    for col in feature_columns:
        if col in ['ob_bull_distance_pct', 'ob_bear_distance_pct', 'ob_bull_age', 'ob_bear_age',
                  'ob_bull_avg_quality', 'ob_bear_avg_quality', 'ob_bull_volatility_ratio', 'ob_bear_volatility_ratio']:
            df[col] = float('nan')
        else:
            df[col] = 0

    # Process each bar and update features
    for i in range(len(df)):
        # Check for new order blocks forming at this bar
        for ob_idx, swing_idx, break_idx in bull_order_blocks:
            if ob_idx == i:
                # Calculate OB properties
                ob_high = df['high'].iloc[i]
                ob_low = df['low'].iloc[i]
                ob_size = ob_high - ob_low
                ob_body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])

                # Get ATR value for this bar
                atr_value = None
                if atr is not None:
                    if isinstance(atr, np.ndarray) and i < len(atr):
                        atr_value = atr[i]
                    else:
                        atr_value = atr

                # Calculate quality score
                quality = calculate_ob_quality(df, i, ob_size, ob_body_size, atr_value)

                # Calculate volatility ratio
                vol_ratio = 0.0
                if atr_value is not None and atr_value > 0:
                    vol_ratio = ob_size / atr_value

                # Check if it's a clean OB
                clean = is_clean_ob(df, ob_idx, break_idx, is_bullish=True)

                # Create the order block object
                new_ob = OrderBlock(
                    idx=i,
                    high=ob_high,
                    low=ob_low,
                    ob_type="bullish",
                    swing_idx=swing_idx,
                    break_idx=break_idx,
                    quality=quality,
                    clean=clean,
                    vol_ratio=vol_ratio
                )

                # Add to manager
                ob_manager.add_order_block(new_ob)

        # Same for bearish order blocks
        for ob_idx, swing_idx, break_idx in bear_order_blocks:
            if ob_idx == i:
                # Calculate OB properties
                ob_high = df['high'].iloc[i]
                ob_low = df['low'].iloc[i]
                ob_size = ob_high - ob_low
                ob_body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])

                # Get ATR value for this bar
                atr_value = None
                if atr is not None:
                    if isinstance(atr, np.ndarray) and i < len(atr):
                        atr_value = atr[i]
                    else:
                        atr_value = atr

                # Calculate quality score
                quality = calculate_ob_quality(df, i, ob_size, ob_body_size, atr_value)

                # Calculate volatility ratio
                vol_ratio = 0.0
                if atr_value is not None and atr_value > 0:
                    vol_ratio = ob_size / atr_value

                # Check if it's a clean OB
                clean = is_clean_ob(df, ob_idx, break_idx, is_bullish=False)

                # Create the order block object
                new_ob = OrderBlock(
                    idx=i,
                    high=ob_high,
                    low=ob_low,
                    ob_type="bearish",
                    swing_idx=swing_idx,
                    break_idx=break_idx,
                    quality=quality,
                    clean=clean,
                    vol_ratio=vol_ratio
                )

                # Add to manager
                ob_manager.add_order_block(new_ob)

        # Update all active order blocks with new price data
        ob_manager.update_all_blocks(
            idx=i,
            price=df['close'].iloc[i],
            high=df['high'].iloc[i],
            low=df['low'].iloc[i]
        )

        # Get feature summary and update dataframe
        features = ob_manager.get_feature_summary(df['close'].iloc[i])

        # Update dataframe columns
        df.loc[i, 'bullish_ob_present'] = features['bull_ob_present']
        df.loc[i, 'bearish_ob_present'] = features['bear_ob_present']
        df.loc[i, 'bull_ob_count'] = features['bull_ob_count']
        df.loc[i, 'bear_ob_count'] = features['bear_ob_count']

        # Update distance features (closest block)
        if features['bull_distance_pct'] is not None:
            df.loc[i, 'ob_bull_distance_pct'] = features['bull_distance_pct']
            df.loc[i, 'ob_bull_retests'] = features['bull_closest_retests']
            df.loc[i, 'ob_bull_quality'] = features['bull_closest_quality']
            df.loc[i, 'ob_bull_clean'] = features['bull_closest_clean']
            df.loc[i, 'ob_bull_volatility_ratio'] = features['bull_closest_vol_ratio']

        if features['bear_distance_pct'] is not None:
            df.loc[i, 'ob_bear_distance_pct'] = features['bear_distance_pct']
            df.loc[i, 'ob_bear_retests'] = features['bear_closest_retests']
            df.loc[i, 'ob_bear_quality'] = features['bear_closest_quality']
            df.loc[i, 'ob_bear_clean'] = features['bear_closest_clean']
            df.loc[i, 'ob_bear_volatility_ratio'] = features['bear_closest_vol_ratio']

        # Update average metrics
        if features['bull_quality_avg'] is not None:
            df.loc[i, 'ob_bull_avg_quality'] = features['bull_quality_avg']
            df.loc[i, 'ob_bull_age'] = features['bull_age_avg']

        if features['bear_quality_avg'] is not None:
            df.loc[i, 'ob_bear_avg_quality'] = features['bear_quality_avg']
            df.loc[i, 'ob_bear_age'] = features['bear_age_avg']

    # Add trend context (similar to original code)
    if 'trend' not in df.columns and len(df) >= 50:
        df['ma50'] = df['close'].rolling(50).mean()
        df['trend'] = np.where(df['close'] > df['ma50'], 1, -1)
        df['ob_bull_with_trend'] = df['bullish_ob_present'] * (df['trend'] == 1)
        df['ob_bear_with_trend'] = df['bearish_ob_present'] * (df['trend'] == -1)

    return df