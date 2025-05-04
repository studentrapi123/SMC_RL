"""
Enhanced Order Block detection and feature engineering for RL-based SMC trading.
This module provides classes and functions to detect, track, and analyze order blocks
for Smart Money Concepts (SMC) based trading strategies.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union


def prepare_features_for_storage(df):
    """
    Prepare array features for CSV storage

    Args:
        df: DataFrame with order block array features

    Returns:
        DataFrame with list columns converted to strings for storage
    """
    df_out = df.copy()

    # Find all list columns
    list_columns = [col for col in df.columns if isinstance(df.iloc[0].get(col, None), list)]

    # Convert lists to their string representation
    for col in list_columns:
        df_out[col] = df_out[col].apply(str)

    return df_out

class OrderBlock:
    """
    Class representing a single order block with all its properties and state.
    Tracks formation, evolution, and interaction with price.
    """

    def __init__(self,
                 idx: int,
                 high: float,
                 low: float,
                 ob_type: str,
                 swing_idx: int,
                 break_idx: int,
                 quality: float = 0.0,
                 clean: int = 0,
                 vol_ratio: float = 0.0):
        """
        Initialize a new order block.

        Args:
            idx: Index where this OB formed
            high: High price of the OB
            low: Low price of the OB
            ob_type: "bullish" or "bearish"
            swing_idx: Related swing point
            break_idx: BOS break index
            quality: Quality score [0-1]
            clean: Whether the OB formation was clean (minimal price interaction before breakout)
            vol_ratio: Volume relative to recent average
        """
        # Basic properties
        self.idx = idx  # Index where this OB formed
        self.high = high  # High price of the OB
        self.low = low  # Low price of the OB
        self.ob_type = ob_type  # "bullish" or "bearish"
        self.swing_idx = swing_idx  # Related swing point
        self.break_idx = break_idx  # BOS break index
        self.direction = 1 if ob_type == "bullish" else -1  # Numerical direction

        # Derived properties
        self.creation_time = idx
        self.quality = quality
        self.clean = clean
        self.vol_ratio = vol_ratio
        self.size = high - low

        # State properties (updated over time)
        self.mitigated = False
        self.mitigation_idx = None
        self.retests = 0
        self.age = 0
        self.last_retest_idx = None

    def calculate_distance(self, current_price: float) -> float:
        """
        Calculate direction-aware distance from current price to order block.
        Positive values indicate price is in the expected reaction zone.

        Args:
            current_price: Current price to measure distance from

        Returns:
            Normalized directional distance
        """
        if self.ob_type == "bullish":
            # For bullish OB: positive when price below OB (expected move up to OB)
            if current_price < self.low:
                return (self.low - current_price) / (self.high - self.low)
            elif current_price > self.high:
                return -(current_price - self.high) / (self.high - self.low)
            else:
                return 0.0  # Inside the OB
        else:  # bearish
            # For bearish OB: positive when price above OB (expected move down to OB)
            if current_price > self.high:
                return (current_price - self.high) / (self.high - self.low)
            elif current_price < self.low:
                return -(self.low - current_price) / (self.high - self.low)
            else:
                return 0.0  # Inside the OB

    def is_active(self) -> bool:
        """Check if this order block is still active (not mitigated)"""
        return not self.mitigated

    def update(self, current_idx: int, current_price: float, current_high: float, current_low: float) -> None:
        """
        Update the order block state based on new price data.

        Args:
            current_idx: Current bar index
            current_price: Current close price
            current_high: Current high price
            current_low: Current low price
        """
        # Update age
        self.age = current_idx - self.creation_time

        # Check for retest (price touching or entering the zone)
        if self.ob_type == "bullish":
            if current_low <= self.high and current_high >= self.low:
                self.retests += 1
                self.last_retest_idx = current_idx
        else:  # bearish
            if current_high >= self.low and current_low <= self.high:
                self.retests += 1
                self.last_retest_idx = current_idx

        # Make mitigation require a full close inside the zone:
        if not self.mitigated:
            if self.ob_type == "bullish" and current_price <= self.low:  # stricter - needs price below the bottom
                self.mitigated = True
                self.mitigation_idx = current_idx
            elif self.ob_type == "bearish" and current_price >= self.high:  # stricter - needs price above the top
                self.mitigated = True
                self.mitigation_idx = current_idx

    def to_dict(self) -> Dict:
        """Convert order block to dictionary for easy serialization"""
        return {
            "direction": self.direction,
            "distance": None,  # Will be calculated dynamically later
            "high": self.high,
            "low": self.low,
            "quality": self.quality,
            "age": self.age,
            "retests": self.retests,
            "clean": self.clean,
            "vol_ratio": self.vol_ratio,
            "size": self.size,
            "creation_time": self.creation_time
        }


class OrderBlockManager:
    """
    Class for managing and tracking multiple order blocks.
    Handles detection, state updates, and feature calculation.
    """

    def __init__(self):
        """Initialize a new order block manager"""
        self.all_blocks = []  # Historical record of all blocks
        self.active_blocks = []  # Currently active blocks

    def add_order_block(self, order_block: OrderBlock) -> None:
        """
        Add a new order block to tracking.

        Args:
            order_block: OrderBlock object to add
        """
        self.all_blocks.append(order_block)
        self.active_blocks.append(order_block)

    def update_all_blocks(self, idx: int, price: float, high: float, low: float) -> None:
        """
        Update all active order blocks with new data.

        Args:
            idx: Current bar index
            price: Current close price
            high: Current high price
            low: Current low price
        """
        # Update each block and filter out mitigated ones
        still_active = []
        for ob in self.active_blocks:
            ob.update(idx, price, high, low)
            if ob.is_active():
                still_active.append(ob)

        self.active_blocks = still_active

    def get_closest_block(self, price: float, direction: Optional[int] = None) -> Optional[OrderBlock]:
        """
        Get the order block closest to current price, optionally filtered by direction.

        Args:
            price: Current price
            direction: Filter by direction (1=bullish, -1=bearish, None=any)

        Returns:
            Closest OrderBlock or None if no blocks match criteria
        """
        if not self.active_blocks:
            return None

        filtered_blocks = self.active_blocks
        if direction is not None:
            filtered_blocks = [ob for ob in self.active_blocks if ob.direction == direction]

        if not filtered_blocks:
            return None

        return min(filtered_blocks, key=lambda ob: abs(ob.calculate_distance(price)))

    def get_blocks_by_relevance(self, price: float, atr: float = None) -> List[OrderBlock]:
        """
        Sort active blocks by relevance to current price.

        Args:
            price: Current price
            atr: Current ATR value (optional)

        Returns:
            List of OrderBlocks sorted by relevance
        """
        if not self.active_blocks:
            return []

        # Calculate relevance scores
        scored_blocks = []
        for ob in self.active_blocks:
            distance = abs(ob.calculate_distance(price))
            # Exponential decay for distance (closer is better)
            distance_score = np.exp(-2 * distance) if distance is not None else 0

            # Age decay (newer is better)
            age_score = np.exp(-0.01 * ob.age)

            # Retest factor (more retests is better, up to a point)
            retest_score = min(1.0, ob.retests / 3)

            # Quality factor (directly use quality score)
            quality_score = ob.quality

            # Combine scores with weights
            relevance = (0.4 * distance_score +
                         0.3 * quality_score +
                         0.2 * age_score +
                         0.1 * retest_score)

            scored_blocks.append((ob, relevance))

        # Sort by descending relevance
        scored_blocks.sort(key=lambda x: x[1], reverse=True)

        # Return just the blocks
        return [block for block, _ in scored_blocks]

    def get_feature_arrays(self, price: float) -> Dict[str, List]:
        """
        Create arrays of order block features for storage in DataFrame.
        Each array contains values for all active order blocks.

        Args:
            price: Current price for distance calculations

        Returns:
            Dictionary of feature arrays
        """
        if not self.active_blocks:
            return {
                'ob_directions': [],
                'ob_distances': [],
                'ob_qualities': [],
                'ob_ages': [],
                'ob_high_prices': [],
                'ob_low_prices': [],
                'ob_retests': [],
                'ob_cleanness': [],
                'ob_sizes': [],
                'ob_volume_ratios': []
            }

        # Sort by relevance to ensure consistent ordering
        sorted_blocks = self.get_blocks_by_relevance(price)

        # Create feature arrays
        features = {
            'ob_directions': [ob.direction for ob in sorted_blocks],
            'ob_distances': [ob.calculate_distance(price) for ob in sorted_blocks],
            'ob_qualities': [ob.quality for ob in sorted_blocks],
            'ob_ages': [ob.age for ob in sorted_blocks],
            'ob_high_prices': [ob.high for ob in sorted_blocks],
            'ob_low_prices': [ob.low for ob in sorted_blocks],
            'ob_retests': [ob.retests for ob in sorted_blocks],
            'ob_cleanness': [ob.clean for ob in sorted_blocks],
            'ob_sizes': [ob.size for ob in sorted_blocks],
            'ob_volume_ratios': [ob.vol_ratio for ob in sorted_blocks]
        }

        return features

    def get_summary_stats(self, price: float) -> Dict[str, float]:
        """
        Calculate summary statistics about all active order blocks.

        Args:
            price: Current price

        Returns:
            Dictionary of summary statistics
        """
        stats = {
            'bull_ob_count': 0,
            'bear_ob_count': 0,
            'nearest_bull_distance': None,
            'nearest_bear_distance': None,
            'mean_quality': 0.0,
            'max_quality': 0.0
        }

        if not self.active_blocks:
            return stats

        # Count by direction
        bull_blocks = [ob for ob in self.active_blocks if ob.direction == 1]
        bear_blocks = [ob for ob in self.active_blocks if ob.direction == -1]

        stats['bull_ob_count'] = len(bull_blocks)
        stats['bear_ob_count'] = len(bear_blocks)

        # Nearest distances
        if bull_blocks:
            closest_bull = min(bull_blocks, key=lambda ob: abs(ob.calculate_distance(price)))
            stats['nearest_bull_distance'] = closest_bull.calculate_distance(price)

        if bear_blocks:
            closest_bear = min(bear_blocks, key=lambda ob: abs(ob.calculate_distance(price)))
            stats['nearest_bear_distance'] = closest_bear.calculate_distance(price)

        # Quality metrics
        if self.active_blocks:
            qualities = [ob.quality for ob in self.active_blocks]
            stats['mean_quality'] = sum(qualities) / len(qualities)
            stats['max_quality'] = max(qualities)

        return stats


def find_order_block_for_bullish_bos(df: pd.DataFrame, swing_high_index: int, break_index: int) -> int:
    """
    Find the order block for a bullish BOS (the last bearish candle in the range).

    Args:
        df: DataFrame with price data
        swing_high_index: Index of the swing high
        break_index: Index of the breakout bar

    Returns:
        Index of the order block or -1 if none found
    """
    # Loop backward from the break index to the swing high index
    for i in range(break_index - 1, swing_high_index, -1):
        # Check if it's a bearish candle (close < open)
        if df['close'].iloc[i] < df['open'].iloc[i]:
            return i

    return -1  # No suitable candle found


def find_order_block_for_bearish_bos(df: pd.DataFrame, swing_low_index: int, break_index: int) -> int:
    """
    Find the order block for a bearish BOS (the last bullish candle in the range).

    Args:
        df: DataFrame with price data
        swing_low_index: Index of the swing low
        break_index: Index of the breakout bar

    Returns:
        Index of the order block or -1 if none found
    """
    # Loop backward from the break index to the swing low index
    for i in range(break_index - 1, swing_low_index, -1):
        # Check if it's a bullish candle (close > open)
        if df['close'].iloc[i] > df['open'].iloc[i]:
            return i

    return -1  # No suitable candle found


def is_clean_ob(df: pd.DataFrame, ob_idx: int, break_idx: int, is_bullish: bool = True) -> int:
    """
    Check if an order block is "clean" - had minimal price interaction before breakout.

    Args:
        df: DataFrame with price data
        ob_idx: Index of the order block
        break_idx: Index of the breakout bar
        is_bullish: True for bullish OB, False for bearish OB

    Returns:
        1 if clean, 0 if not
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


def calculate_ob_quality(df: pd.DataFrame, i: int, ob_size: float, ob_body_size: float,
                         atr_value: Optional[float] = None) -> float:
    """
    Calculate a quality score (0-1) for an order block based on multiple metrics.

    Args:
        df: DataFrame with price data
        i: Current bar index
        ob_size: Size of the order block (high - low)
        ob_body_size: Size of the candle body
        atr_value: ATR value at this bar (if available)

    Returns:
        Quality score between 0 and 1
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


def detect_bos_events(df: pd.DataFrame) -> Tuple[Dict[str, List], Dict[str, List]]:
    """
    Detect Break of Structure (BOS) events and order blocks.

    Args:
        df: DataFrame with price data containing 'is_swing_high' and 'is_swing_low' columns

    Returns:
        Tuple of (bos_events, order_blocks) dictionaries
    """
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

    # Process bullish BOS events
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

    # Process bearish BOS events
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


def add_order_block_features(df: pd.DataFrame, bos_events: Dict[str, List],
                             atr: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Add order block features to the dataframe using array-based approach.

    Args:
        df: DataFrame with price data
        bos_events: Dictionary with bullish and bearish BOS events
        atr: ATR array (if available)

    Returns:
        DataFrame with added order block features
    """
    df = df.copy()

    # Create an OrderBlockManager to track all blocks
    ob_manager = OrderBlockManager()

    # Initialize basic indicator columns
    df['bullish_ob_present'] = 0
    df['bearish_ob_present'] = 0

    # Pre-process all order blocks from BOS events
    bull_order_blocks = []
    bear_order_blocks = []

    # Find order blocks from BOS events
    for swing_idx, break_idx in bos_events['bullish']:
        ob_idx = find_order_block_for_bullish_bos(df, swing_idx, break_idx)
        if ob_idx != -1:
            bull_order_blocks.append((ob_idx, swing_idx, break_idx))
            # Mark this candle as a bullish order block
            df.loc[ob_idx, 'bullish_ob_present'] = 1

    for swing_idx, break_idx in bos_events['bearish']:
        ob_idx = find_order_block_for_bearish_bos(df, swing_idx, break_idx)
        if ob_idx != -1:
            bear_order_blocks.append((ob_idx, swing_idx, break_idx))
            # Mark this candle as a bearish order block
            df.loc[ob_idx, 'bearish_ob_present'] = 1

    # Initialize array columns as Python lists, not empty strings
    array_columns = [
        'ob_directions', 'ob_distances', 'ob_qualities',
        'ob_ages', 'ob_high_prices', 'ob_low_prices', 'ob_retests',
        'ob_cleanness', 'ob_sizes', 'ob_volume_ratios'
    ]

    for col in array_columns:
        df[col] = [[] for _ in range(len(df))]

    # Initialize summary columns
    summary_columns = [
        'bull_ob_count', 'bear_ob_count',
        'nearest_bull_distance', 'nearest_bear_distance',
        'mean_quality', 'max_quality'
    ]

    for col in summary_columns:
        if col.endswith('_count'):
            df[col] = 0
        else:
            df[col] = float('nan')

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

        # Get feature arrays and update dataframe
        feature_arrays = ob_manager.get_feature_arrays(df['close'].iloc[i])
        for col, values in feature_arrays.items():
            df.at[i, col] = values

        # Get summary stats and update dataframe
        summary_stats = ob_manager.get_summary_stats(df['close'].iloc[i])
        for col, value in summary_stats.items():
            df.at[i, col] = value

    # Add trend context
    if 'trend' not in df.columns and len(df) >= 50:
        df['ma50'] = df['close'].rolling(50).mean()
        df['trend'] = np.where(df['close'] > df['ma50'], 1, -1)
        df['ob_bull_with_trend'] = (df['bull_ob_count'] > 0) & (df['trend'] == 1)
        df['ob_bear_with_trend'] = (df['bear_ob_count'] > 0) & (df['trend'] == -1)

    # Make sure all feature arrays are actually lists (not strings)
    for col in array_columns:
        for i in range(len(df)):
            if isinstance(df.at[i, col], str):
                df.at[i, col] = eval(df.at[i, col])

    return df


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def plot_swing_bos_orderblocks(df, tops, bottoms, atr, bos_events, order_blocks,
                               start_idx=None, end_idx=None, save_path=None):
    """
    Create a visualization of swing points, BOS events, and order blocks

    Args:
        df: DataFrame with price data
        tops: List of top swing points [detection_index, max_index, price]
        bottoms: List of bottom swing points [detection_index, min_index, price]
        atr: Array of ATR values
        bos_events: Dictionary with bullish and bearish BOS events
        order_blocks: Dictionary with bullish and bearish order blocks
        start_idx: Start index for the plot (default: last 100 candles)
        end_idx: End index for the plot (default: end of data)
        save_path: Path to save the visualization (default: 'swing_bos_orderblocks.png')
    """
    # Set default window if not provided
    if start_idx is None:
        start_idx = max(0, len(df) - 100)  # Last 100 candles by default
    if end_idx is None:
        end_idx = len(df)
    if save_path is None:
        save_path = 'swing_bos_orderblocks.png'

    # Take subset of data to display
    df_subset = df.iloc[start_idx:end_idx].reset_index(drop=True)
    atr_subset = atr[start_idx:end_idx]

    # Create mapping from original indices to plot indices
    idx_map = {orig_idx: plot_idx for plot_idx, orig_idx in enumerate(range(start_idx, end_idx))}

    # Set figure properties
    plt.rcParams['figure.dpi'] = 300
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   facecolor='white')

    # Set colors
    bullish_color = '#2E7D32'  # Dark green
    bearish_color = '#C62828'  # Dark red
    bos_bullish_color = 'green'
    bos_bearish_color = 'red'
    bullish_ob_color = (144 / 255, 238 / 255, 144 / 255, 0.5)  # Light green with transparency
    bearish_ob_color = (255 / 255, 182 / 255, 193 / 255, 0.5)  # Light pink with transparency

    # First, draw order blocks (behind everything else)
    for ob_idx, swing_idx, break_idx, mitigation_idx in order_blocks['bullish']:
        if start_idx <= ob_idx < end_idx:
            # Only show OBs that are visible in current view
            plot_ob_idx = idx_map.get(ob_idx)

            # Calculate mitigation point (or end of view if not mitigated)
            mitigation_plot_idx = None
            if mitigation_idx < end_idx:
                mitigation_plot_idx = idx_map.get(mitigation_idx)
            else:
                mitigation_plot_idx = len(df_subset) - 1

            if plot_ob_idx is not None and mitigation_plot_idx is not None:
                # Draw rectangle for order block
                rect = patches.Rectangle(
                    (plot_ob_idx, df.iloc[ob_idx]['low']),
                    mitigation_plot_idx - plot_ob_idx,
                    df.iloc[ob_idx]['high'] - df.iloc[ob_idx]['low'],
                    linewidth=0, edgecolor='none', facecolor=bullish_ob_color, zorder=1
                )
                ax1.add_patch(rect)

                # Add OB label
                mid_x = plot_ob_idx + 2
                mid_y = (df.iloc[ob_idx]['high'] + df.iloc[ob_idx]['low']) / 2
                ax1.annotate('OB', (mid_x, mid_y), ha='center', va='center',
                             color=bos_bullish_color, fontsize=8, fontweight='bold')

                # Print debug info
                print(f"Bullish OB at idx={ob_idx}, real_idx={ob_idx + start_idx}, "
                      f"price range: {df.iloc[ob_idx]['low']}-{df.iloc[ob_idx]['high']}")

    # Same for bearish order blocks
    for ob_idx, swing_idx, break_idx, mitigation_idx in order_blocks['bearish']:
        if start_idx <= ob_idx < end_idx:
            # Only show OBs that are visible in current view
            plot_ob_idx = idx_map.get(ob_idx)

            # Calculate mitigation point (or end of view if not mitigated)
            mitigation_plot_idx = None
            if mitigation_idx < end_idx:
                mitigation_plot_idx = idx_map.get(mitigation_idx)
            else:
                mitigation_plot_idx = len(df_subset) - 1

            if plot_ob_idx is not None and mitigation_plot_idx is not None:
                # Draw rectangle for order block
                rect = patches.Rectangle(
                    (plot_ob_idx, df.iloc[ob_idx]['low']),
                    mitigation_plot_idx - plot_ob_idx,
                    df.iloc[ob_idx]['high'] - df.iloc[ob_idx]['low'],
                    linewidth=0, edgecolor='none', facecolor=bearish_ob_color, zorder=1
                )
                ax1.add_patch(rect)

                # Add OB label
                mid_x = plot_ob_idx + 2
                mid_y = (df.iloc[ob_idx]['high'] + df.iloc[ob_idx]['low']) / 2
                ax1.annotate('OB', (mid_x, mid_y), ha='center', va='center',
                             color=bos_bearish_color, fontsize=8, fontweight='bold')

                # Print debug info
                print(f"Bearish OB at idx={ob_idx}, real_idx={ob_idx + start_idx}, "
                      f"price range: {df.iloc[ob_idx]['low']}-{df.iloc[ob_idx]['high']}")

    # Plot candlesticks
    for i in range(len(df_subset)):
        x = i
        open_price = df_subset['open'].iloc[i]
        close_price = df_subset['close'].iloc[i]
        high_price = df_subset['high'].iloc[i]
        low_price = df_subset['low'].iloc[i]

        # Determine special highlighting for OB candles
        color = bullish_color if close_price >= open_price else bearish_color

        # Check if this is an order block (adjust index for subset)
        real_idx = i + start_idx
        if df.iloc[real_idx].get('bullish_ob_present', 0) == 1:
            color = 'blue'  # Special color for bullish OB candles
        elif df.iloc[real_idx].get('bearish_ob_present', 0) == 1:
            color = 'purple'  # Special color for bearish OB candles

        # Draw candle body
        ax1.bar(x, abs(close_price - open_price), 0.6,
                bottom=min(open_price, close_price), color=color, zorder=3)

        # Draw candle wick
        ax1.plot([x, x], [low_price, high_price], color='black', linewidth=0.7, zorder=2)

    # Extract all swing points for zigzag lines
    all_swing_points = []

    # Add tops to swing points and draw swing high points
    for _, max_idx, price in tops:
        if start_idx <= max_idx < end_idx:
            plot_idx = idx_map[max_idx]
            all_swing_points.append((plot_idx, price, 'top'))

            # Draw swing high annotation
            ax1.scatter(plot_idx, price, color='blue', s=30, zorder=5, marker="^")
            ax1.annotate('SH', (plot_idx, price), xytext=(0, 7),
                         textcoords='offset points', ha='center', color='blue', fontsize=8)

    # Add bottoms to swing points and draw swing low points
    for _, min_idx, price in bottoms:
        if start_idx <= min_idx < end_idx:
            plot_idx = idx_map[min_idx]
            all_swing_points.append((plot_idx, price, 'bottom'))

            # Draw swing low annotation
            ax1.scatter(plot_idx, price, color='blue', s=30, zorder=5, marker="v")
            ax1.annotate('SL', (plot_idx, price), xytext=(0, -10),
                         textcoords='offset points', ha='center', color='blue', fontsize=8)

    # Sort all points by index to ensure correct line drawing sequence
    all_swing_points.sort(key=lambda x: x[0])

    # Draw zigzag lines connecting tops and bottoms
    if len(all_swing_points) >= 2:
        x_coords = [point[0] for point in all_swing_points]
        y_coords = [point[1] for point in all_swing_points]
        ax1.plot(x_coords, y_coords, color='blue', linewidth=1.0, zorder=4)  # Blue line

    # Plot bullish BOS events
    for swing_idx, break_idx in bos_events['bullish']:
        if start_idx <= swing_idx < end_idx and start_idx <= break_idx < end_idx:
            # Convert to plot indices
            plot_swing_idx = idx_map[swing_idx]
            plot_break_idx = idx_map[break_idx]

            # Draw horizontal line
            swing_high_val = df.iloc[swing_idx]['high']
            ax1.plot([plot_swing_idx, plot_break_idx], [swing_high_val, swing_high_val],
                     color=bos_bullish_color, linewidth=1.5, zorder=6)

            # Add BOS label
            mid_x = plot_swing_idx + (plot_break_idx - plot_swing_idx) / 2
            ax1.annotate('BOS', (mid_x, swing_high_val), xytext=(0, 7),
                         textcoords='offset points', ha='center', color=bos_bullish_color, fontsize=8)

    # Plot bearish BOS events
    for swing_idx, break_idx in bos_events['bearish']:
        if start_idx <= swing_idx < end_idx and start_idx <= break_idx < end_idx:
            # Convert to plot indices
            plot_swing_idx = idx_map[swing_idx]
            plot_break_idx = idx_map[break_idx]

            # Draw horizontal line
            swing_low_val = df.iloc[swing_idx]['low']
            ax1.plot([plot_swing_idx, plot_break_idx], [swing_low_val, swing_low_val],
                     color=bos_bearish_color, linewidth=1.5, zorder=6)

            # Add BOS label
            mid_x = plot_swing_idx + (plot_break_idx - plot_swing_idx) / 2
            ax1.annotate('BOS', (mid_x, swing_low_val), xytext=(0, -10),
                         textcoords='offset points', ha='center', color=bos_bearish_color, fontsize=8)

    # Plot ATR on bottom subplot
    ax2.plot(range(len(atr_subset)), atr_subset, color='#1565C0', linewidth=1.5)
    ax2.fill_between(range(len(atr_subset)), 0, atr_subset, color='#1565C0', alpha=0.2)

    # Add average ATR line
    avg_atr = np.mean(atr_subset)
    ax2.axhline(y=avg_atr, color='#795548', linestyle='--', linewidth=1)
    ax2.text(len(atr_subset) * 0.8, avg_atr * 1.1, f'Avg ATR: {avg_atr:.6f}',
             color='black', fontsize=10)

    # Add titles and styling
    ax1.set_title('Swing Points, Break of Structure (BOS), and Order Blocks',
                  color='black', fontsize=20, fontweight='bold')
    ax2.set_title('Average True Range (ATR) Volatility Metric', color='black', fontsize=16)

    # Add grid and improve styling
    ax1.grid(True, alpha=0.15, linestyle='-')
    ax2.grid(True, alpha=0.15, linestyle='-')

    # Add summary text with counts
    bull_ob_count = sum(1 for _, _, _, _ in order_blocks['bullish']
                        if start_idx <= _ < end_idx)
    bear_ob_count = sum(1 for _, _, _, _ in order_blocks['bearish']
                        if start_idx <= _ < end_idx)

    summary_text = (
        f"Total Bullish BOS: {len([b for b in bos_events['bullish'] if start_idx <= b[0] < end_idx])}\n"
        f"Total Bearish BOS: {len([b for b in bos_events['bearish'] if start_idx <= b[0] < end_idx])}\n"
        f"Bullish Order Blocks: {bull_ob_count}\n"
        f"Bearish Order Blocks: {bear_ob_count}\n"
        f"Active Bullish OBs: {df_subset['bull_ob_count'].iloc[-1] if 'bull_ob_count' in df_subset.columns else 'N/A'}\n"
        f"Active Bearish OBs: {df_subset['bear_ob_count'].iloc[-1] if 'bear_ob_count' in df_subset.columns else 'N/A'}"
    )
    plt.figtext(0.02, 0.02, summary_text, fontsize=10)

    # Style axes
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', colors='black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#555555')
        ax.spines['left'].set_color('#555555')

    # Remove x-axis labels from top subplot
    ax1.set_xticklabels([])
    ax1.set_xticks([])

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Display the figure
    plt.show()

    return fig