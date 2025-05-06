"""Fair Value Gap detection and related features with advanced array-based tracking"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union


class FairValueGap:
    """
    Class representing a single fair value gap with all its properties and state.
    Tracks formation, evolution, and interaction with price.
    """

    def __init__(self,
                 idx: int,
                 top_price: float,
                 bottom_price: float,
                 fvg_type: str,
                 size: float = 0.0,
                 quality: float = 0.0,
                 vol_ratio: float = 0.0):
        """
        Initialize a new fair value gap.

        Args:
            idx: Index where this FVG formed
            top_price: Top price of the FVG
            bottom_price: Bottom price of the FVG
            fvg_type: "bullish" or "bearish"
            size: Size of the gap (top_price - bottom_price)
            quality: Quality score [0-1]
            vol_ratio: Volume relative to recent average
        """
        # Basic properties
        self.idx = idx  # Index where this FVG formed
        self.top_price = top_price  # Top price of the FVG
        self.bottom_price = bottom_price  # Bottom price of the FVG
        self.fvg_type = fvg_type  # "bullish" or "bearish"
        self.direction = 1 if fvg_type == "bullish" else -1  # Numerical direction
        self.size = size if size > 0 else top_price - bottom_price

        # Derived properties
        self.creation_time = idx
        self.quality = quality
        self.vol_ratio = vol_ratio

        # State properties (updated over time)
        self.mitigated = False
        self.mitigation_idx = None
        self.retests = 0
        self.age = 0
        self.last_retest_idx = None

    def calculate_distance(self, current_price: float) -> float:
        """
        Calculate direction-aware distance from current price to fair value gap.
        Positive values indicate price is in the expected reaction zone.

        Args:
            current_price: Current price to measure distance from

        Returns:
            Normalized directional distance
        """
        if self.fvg_type == "bullish":
            # For bullish FVG: positive when price below FVG (expected move up into FVG)
            if current_price < self.bottom_price:
                return (self.bottom_price - current_price) / self.size
            elif current_price > self.top_price:
                return -(current_price - self.top_price) / self.size
            else:
                return 0.0  # Inside the FVG
        else:  # bearish
            # For bearish FVG: positive when price above FVG (expected move down into FVG)
            if current_price > self.top_price:
                return (current_price - self.top_price) / self.size
            elif current_price < self.bottom_price:
                return -(self.bottom_price - current_price) / self.size
            else:
                return 0.0  # Inside the FVG

    def is_active(self) -> bool:
        """Check if this fair value gap is still active (not mitigated)"""
        return not self.mitigated

    def update(self, current_idx: int, current_price: float, current_high: float, current_low: float, current_open: float) -> None:
        """
        Update the fair value gap state based on new price data.

        Args:
            current_idx: Current bar index
            current_price: Current close price
            current_high: Current high price
            current_low: Current low price
            current_open: Current open price
        """
        # Update age
        self.age = current_idx - self.creation_time

        # Check for retest (price touching or entering the gap)
        if current_low <= self.top_price and current_high >= self.bottom_price:
            self.retests += 1
            self.last_retest_idx = current_idx

        # Check for mitigation - requires price body (open/close) to be inside the FVG
        if not self.mitigated:
            min_body = min(current_price, current_open)
            max_body = max(current_price, current_open)

            # If the price body enters or crosses the FVG
            if min_body <= self.top_price and max_body >= self.bottom_price:
                self.mitigated = True
                self.mitigation_idx = current_idx

    def to_dict(self) -> Dict:
        """Convert fair value gap to dictionary for easy serialization"""
        return {
            "direction": self.direction,
            "distance": None,  # Will be calculated dynamically later
            "top_price": self.top_price,
            "bottom_price": self.bottom_price,
            "quality": self.quality,
            "age": self.age,
            "retests": self.retests,
            "size": self.size,
            "vol_ratio": self.vol_ratio,
            "creation_time": self.creation_time
        }


class FairValueGapManager:
    """
    Class for managing and tracking multiple fair value gaps.
    Handles detection, state updates, and feature calculation.
    """

    def __init__(self):
        """Initialize a new fair value gap manager"""
        self.all_gaps = []  # Historical record of all gaps
        self.active_gaps = []  # Currently active gaps

    def add_fair_value_gap(self, fvg: FairValueGap) -> None:
        """
        Add a new fair value gap to tracking.

        Args:
            fvg: FairValueGap object to add
        """
        self.all_gaps.append(fvg)
        self.active_gaps.append(fvg)

    def update_all_gaps(self, idx: int, price: float, high: float, low: float, open_price: float) -> None:
        """
        Update all active fair value gaps with new data.

        Args:
            idx: Current bar index
            price: Current close price
            high: Current high price
            low: Current low price
            open_price: Current open price
        """
        # Update each gap and filter out mitigated ones
        still_active = []
        for fvg in self.active_gaps:
            fvg.update(idx, price, high, low, open_price)
            if fvg.is_active():
                still_active.append(fvg)

        self.active_gaps = still_active

    def get_closest_gap(self, price: float, direction: Optional[int] = None) -> Optional[FairValueGap]:
        """
        Get the fair value gap closest to current price, optionally filtered by direction.

        Args:
            price: Current price
            direction: Filter by direction (1=bullish, -1=bearish, None=any)

        Returns:
            Closest FairValueGap or None if no gaps match criteria
        """
        if not self.active_gaps:
            return None

        filtered_gaps = self.active_gaps
        if direction is not None:
            filtered_gaps = [fvg for fvg in self.active_gaps if fvg.direction == direction]

        if not filtered_gaps:
            return None

        return min(filtered_gaps, key=lambda fvg: abs(fvg.calculate_distance(price)))

    def get_gaps_by_relevance(self, price: float, atr: float = None) -> List[FairValueGap]:
        """
        Sort active gaps by relevance to current price.

        Args:
            price: Current price
            atr: Current ATR value (optional)

        Returns:
            List of FairValueGaps sorted by relevance
        """
        if not self.active_gaps:
            return []

        # Calculate relevance scores
        scored_gaps = []
        for fvg in self.active_gaps:
            distance = abs(fvg.calculate_distance(price))
            # Exponential decay for distance (closer is better)
            distance_score = np.exp(-2 * distance) if distance is not None else 0

            # Age decay (newer is better)
            age_score = np.exp(-0.01 * fvg.age)

            # Retest factor (more retests is better, up to a point)
            retest_score = min(1.0, fvg.retests / 3)

            # Quality factor (directly use quality score)
            quality_score = fvg.quality

            # Combine scores with weights
            relevance = (0.4 * distance_score +
                         0.3 * quality_score +
                         0.2 * age_score +
                         0.1 * retest_score)

            scored_gaps.append((fvg, relevance))

        # Sort by descending relevance
        scored_gaps.sort(key=lambda x: x[1], reverse=True)

        # Return just the gaps
        return [gap for gap, _ in scored_gaps]

    def get_feature_arrays(self, price: float) -> Dict[str, List]:
        """
        Create arrays of fair value gap features for storage in DataFrame.
        Each array contains values for all active fair value gaps.

        Args:
            price: Current price for distance calculations

        Returns:
            Dictionary of feature arrays
        """
        if not self.active_gaps:
            return {
                'fvg_directions': [],
                'fvg_distances': [],
                'fvg_qualities': [],
                'fvg_ages': [],
                'fvg_top_prices': [],
                'fvg_bottom_prices': [],
                'fvg_retests': [],
                'fvg_sizes': [],
                'fvg_volume_ratios': []
            }

        # Sort by relevance to ensure consistent ordering
        sorted_gaps = self.get_gaps_by_relevance(price)

        # Create feature arrays
        features = {
            'fvg_directions': [fvg.direction for fvg in sorted_gaps],
            'fvg_distances': [fvg.calculate_distance(price) for fvg in sorted_gaps],
            'fvg_qualities': [fvg.quality for fvg in sorted_gaps],
            'fvg_ages': [fvg.age for fvg in sorted_gaps],
            'fvg_top_prices': [fvg.top_price for fvg in sorted_gaps],
            'fvg_bottom_prices': [fvg.bottom_price for fvg in sorted_gaps],
            'fvg_retests': [fvg.retests for fvg in sorted_gaps],
            'fvg_sizes': [fvg.size for fvg in sorted_gaps],
            'fvg_volume_ratios': [fvg.vol_ratio for fvg in sorted_gaps]
        }

        return features

    def get_summary_stats(self, price: float) -> Dict[str, float]:
        """
        Calculate summary statistics about all active fair value gaps.

        Args:
            price: Current price

        Returns:
            Dictionary of summary statistics
        """
        stats = {
            'bull_fvg_count': 0,
            'bear_fvg_count': 0,
            'nearest_bull_fvg_distance': None,
            'nearest_bear_fvg_distance': None,
            'mean_fvg_quality': 0.0,
            'max_fvg_quality': 0.0,
            'mean_fvg_size': 0.0
        }

        if not self.active_gaps:
            return stats

        # Count by direction
        bull_gaps = [fvg for fvg in self.active_gaps if fvg.direction == 1]
        bear_gaps = [fvg for fvg in self.active_gaps if fvg.direction == -1]

        stats['bull_fvg_count'] = len(bull_gaps)
        stats['bear_fvg_count'] = len(bear_gaps)

        # Nearest distances
        if bull_gaps:
            closest_bull = min(bull_gaps, key=lambda fvg: abs(fvg.calculate_distance(price)))
            stats['nearest_bull_fvg_distance'] = closest_bull.calculate_distance(price)

        if bear_gaps:
            closest_bear = min(bear_gaps, key=lambda fvg: abs(fvg.calculate_distance(price)))
            stats['nearest_bear_fvg_distance'] = closest_bear.calculate_distance(price)

        # Quality metrics
        if self.active_gaps:
            qualities = [fvg.quality for fvg in self.active_gaps]
            stats['mean_fvg_quality'] = sum(qualities) / len(qualities)
            stats['max_fvg_quality'] = max(qualities)

            # Size metrics
            sizes = [fvg.size for fvg in self.active_gaps]
            stats['mean_fvg_size'] = sum(sizes) / len(sizes)

        return stats


def calculate_fvg_quality(fvg_size: float, atr_value: Optional[float] = None, prev_candle_size: float = 0) -> float:
    """
    Calculate a quality score (0-1) for a fair value gap based on multiple metrics.

    Args:
        fvg_size: Size of the fair value gap
        atr_value: ATR value at this bar (if available)
        prev_candle_size: Size of the preceding candle (for context)

    Returns:
        Quality score between 0 and 1
    """
    # Normalize gap size by ATR if available
    atr_ratio = 0.5  # Default mid-value
    if atr_value is not None and atr_value > 0:
        atr_ratio = min(1.0, fvg_size / (2 * atr_value))  # Normalize by 2x ATR

    # Consider gap size relative to preceding candle
    relative_size = 0.5  # Default mid-value
    if prev_candle_size > 0:
        relative_size = min(1.0, fvg_size / prev_candle_size)

    # Combined quality score
    quality = 0.7 * atr_ratio + 0.3 * relative_size

    return quality


def find_fair_value_gaps_optimized(df):
    """
    Optimized detection of Fair Value Gaps

    Args:
        df: DataFrame with price data

    Returns:
        Dictionary with bullish and bearish FVG events
    """
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

    return fvg_events


def add_fair_value_gap_features(df: pd.DataFrame, fvg_events: Dict[str, List],
                               atr: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Add fair value gap features to the dataframe using array-based approach.

    Args:
        df: DataFrame with price data
        fvg_events: Dictionary with bullish and bearish fair value gaps
        atr: ATR array (if available)

    Returns:
        DataFrame with added fair value gap features
    """
    df = df.copy()

    # Create a FairValueGapManager to track all gaps
    fvg_manager = FairValueGapManager()

    # Initialize basic indicator columns
    df['fvg_bull_present'] = 0
    df['fvg_bear_present'] = 0

    # Initialize array columns as Python lists, not empty strings
    array_columns = [
        'fvg_directions', 'fvg_distances', 'fvg_qualities',
        'fvg_ages', 'fvg_top_prices', 'fvg_bottom_prices', 'fvg_retests',
        'fvg_sizes', 'fvg_volume_ratios'
    ]

    for col in array_columns:
        df[col] = [[] for _ in range(len(df))]

    # Initialize summary columns
    summary_columns = [
        'bull_fvg_count', 'bear_fvg_count',
        'nearest_bull_fvg_distance', 'nearest_bear_fvg_distance',
        'mean_fvg_quality', 'max_fvg_quality', 'mean_fvg_size'
    ]

    for col in summary_columns:
        if col.endswith('_count'):
            df[col] = 0
        else:
            df[col] = float('nan')

    # Process each bar and update features
    for i in range(len(df)):
        # Check for new fair value gaps forming at this bar
        for start_idx, end_idx, top_price, bottom_price, _, _ in fvg_events['bullish']:
            if end_idx == i:  # FVG forms at the current bar
                # Calculate the size of the FVG
                fvg_size = top_price - bottom_price

                # Get ATR value for this bar
                atr_value = None
                if atr is not None:
                    if isinstance(atr, np.ndarray) and i < len(atr):
                        atr_value = atr[i]
                    else:
                        atr_value = atr

                # Get previous candle size
                prev_candle_size = 0
                if i > 0:
                    prev_candle_size = df['high'].iloc[i-1] - df['low'].iloc[i-1]

                # Calculate quality score
                quality = calculate_fvg_quality(fvg_size, atr_value, prev_candle_size)

                # Calculate volatility ratio
                vol_ratio = 0.0
                if atr_value is not None and atr_value > 0:
                    vol_ratio = fvg_size / atr_value

                # Create the fair value gap object
                new_fvg = FairValueGap(
                    idx=i,
                    top_price=top_price,
                    bottom_price=bottom_price,
                    fvg_type="bullish",
                    size=fvg_size,
                    quality=quality,
                    vol_ratio=vol_ratio
                )

                # Add to manager
                fvg_manager.add_fair_value_gap(new_fvg)

                # Mark this candle as having a bullish FVG
                df.loc[i, 'fvg_bull_present'] = 1

        # Same for bearish FVGs
        for start_idx, end_idx, top_price, bottom_price, _, _ in fvg_events['bearish']:
            if end_idx == i:  # FVG forms at the current bar
                # Calculate the size of the FVG
                fvg_size = top_price - bottom_price

                # Get ATR value for this bar
                atr_value = None
                if atr is not None:
                    if isinstance(atr, np.ndarray) and i < len(atr):
                        atr_value = atr[i]
                    else:
                        atr_value = atr

                # Get previous candle size
                prev_candle_size = 0
                if i > 0:
                    prev_candle_size = df['high'].iloc[i-1] - df['low'].iloc[i-1]

                # Calculate quality score
                quality = calculate_fvg_quality(fvg_size, atr_value, prev_candle_size)

                # Calculate volatility ratio
                vol_ratio = 0.0
                if atr_value is not None and atr_value > 0:
                    vol_ratio = fvg_size / atr_value

                # Create the fair value gap object
                new_fvg = FairValueGap(
                    idx=i,
                    top_price=top_price,
                    bottom_price=bottom_price,
                    fvg_type="bearish",
                    size=fvg_size,
                    quality=quality,
                    vol_ratio=vol_ratio
                )

                # Add to manager
                fvg_manager.add_fair_value_gap(new_fvg)

                # Mark this candle as having a bearish FVG
                df.loc[i, 'fvg_bear_present'] = 1

        # Update all active fair value gaps with new price data
        fvg_manager.update_all_gaps(
            idx=i,
            price=df['close'].iloc[i],
            high=df['high'].iloc[i],
            low=df['low'].iloc[i],
            open_price=df['open'].iloc[i]
        )

        # Get feature arrays and update dataframe
        feature_arrays = fvg_manager.get_feature_arrays(df['close'].iloc[i])
        for col, values in feature_arrays.items():
            df.at[i, col] = values

        # Get summary stats and update dataframe
        summary_stats = fvg_manager.get_summary_stats(df['close'].iloc[i])
        for col, value in summary_stats.items():
            df.at[i, col] = value

    # Add trend context features if not already present
    if 'trend' not in df.columns and len(df) >= 50:
        df['ma50'] = df['close'].rolling(50).mean()
        df['trend'] = np.where(df['close'] > df['ma50'], 1, -1)
        df['fvg_bull_with_trend'] = (df['bull_fvg_count'] > 0) & (df['trend'] == 1)
        df['fvg_bear_with_trend'] = (df['bear_fvg_count'] > 0) & (df['trend'] == -1)

    # Make sure all feature arrays are actually lists (not strings)
    for col in array_columns:
        for i in range(len(df)):
            if isinstance(df.at[i, col], str):
                df.at[i, col] = eval(df.at[i, col])

    # Calculate active FVG counts
    df['active_fvg_count'] = df['bull_fvg_count'] + df['bear_fvg_count']

    # Add distance percentages for compatibility with existing code
    df['fvg_bull_distance_pct'] = df['nearest_bull_fvg_distance']
    df['fvg_bear_distance_pct'] = df['nearest_bear_fvg_distance']

    return df


def find_mitigation_for_fvg(df: pd.DataFrame, start_idx: int, top_price: float,
                           bottom_price: float) -> Tuple[bool, Optional[int]]:
    """
    Find if and when a fair value gap gets mitigated

    Args:
        df: DataFrame with price data
        start_idx: Starting index to check from (usually end_idx + 1 from FVG detection)
        top_price: Top price of the FVG
        bottom_price: Bottom price of the FVG

    Returns:
        Tuple of (mitigated, mitigation_idx)
    """
    mitigated = False
    mitigation_idx = None

    # Look for mitigation - defined as price body (open/close) entering the FVG
    for i in range(start_idx, len(df)):
        min_body = min(df['open'].iloc[i], df['close'].iloc[i])
        max_body = max(df['open'].iloc[i], df['close'].iloc[i])

        # Check if candle body overlaps with the FVG
        if min_body <= top_price and max_body >= bottom_price:
            mitigated = True
            mitigation_idx = i
            break

    return mitigated, mitigation_idx


def predict_fvg_reaction(df: pd.DataFrame, fvg_events: Dict[str, List]) -> pd.DataFrame:
    """
    Analyze past FVGs to predict potential reactions to current ones

    Args:
        df: DataFrame with price data
        fvg_events: Dictionary with bullish and bearish FVGs

    Returns:
        DataFrame with reaction statistics
    """
    # Track reaction stats
    reactions = []

    # Process bullish FVGs
    for start_idx, end_idx, top_price, bottom_price, mitigated, mitigation_idx in fvg_events['bullish']:
        if mitigated and mitigation_idx is not None:
            # Calculate stats only for mitigated FVGs to ensure we have complete data
            fvg_size = top_price - bottom_price
            time_to_reaction = mitigation_idx - end_idx

            # Calculate if price bounced from the FVG
            bounced = False
            bounce_size = 0.0

            if mitigation_idx + 10 < len(df):  # Ensure enough bars after mitigation
                # Check 10 bars after the mitigation for a bounce
                max_price_after = df['high'].iloc[mitigation_idx:mitigation_idx+10].max()
                bounce_size = max_price_after - top_price
                bounced = bounce_size > 0.25 * fvg_size  # Bounce of at least 25% of FVG size

            reactions.append({
                'type': 'bullish',
                'start_idx': start_idx,
                'end_idx': end_idx,
                'mitigation_idx': mitigation_idx,
                'fvg_size': fvg_size,
                'time_to_reaction': time_to_reaction,
                'bounced': bounced,
                'bounce_size': bounce_size,
                'bounce_ratio': bounce_size / fvg_size if fvg_size > 0 else 0
            })

    # Process bearish FVGs
    for start_idx, end_idx, top_price, bottom_price, mitigated, mitigation_idx in fvg_events['bearish']:
        if mitigated and mitigation_idx is not None:
            # Calculate stats only for mitigated FVGs to ensure we have complete data
            fvg_size = top_price - bottom_price
            time_to_reaction = mitigation_idx - end_idx

            # Calculate if price bounced from the FVG
            bounced = False
            bounce_size = 0.0

            if mitigation_idx + 10 < len(df):  # Ensure enough bars after mitigation
                # Check 10 bars after the mitigation for a bounce
                min_price_after = df['low'].iloc[mitigation_idx:mitigation_idx+10].min()
                bounce_size = bottom_price - min_price_after
                bounced = bounce_size > 0.25 * fvg_size  # Bounce of at least 25% of FVG size

            reactions.append({
                'type': 'bearish',
                'start_idx': start_idx,
                'end_idx': end_idx,
                'mitigation_idx': mitigation_idx,
                'fvg_size': fvg_size,
                'time_to_reaction': time_to_reaction,
                'bounced': bounced,
                'bounce_size': bounce_size,
                'bounce_ratio': bounce_size / fvg_size if fvg_size > 0 else 0
            })

    # Convert to DataFrame if we have reactions
    if reactions:
        reaction_df = pd.DataFrame(reactions)
        return reaction_df
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'type', 'start_idx', 'end_idx', 'mitigation_idx',
            'fvg_size', 'time_to_reaction', 'bounced',
            'bounce_size', 'bounce_ratio'
        ])


def integrate_with_feature_set(df, extracted_fvgs, atr):
    """
    Integrate Fair Value Gap features with the complete feature set

    Args:
        df: DataFrame with price data
        extracted_fvgs: Dictionary of FVG events from find_fair_value_gaps_optimized
        atr: ATR array

    Returns:
        DataFrame with FVG features added
    """
    # First add the array-based FVG features
    df = add_fair_value_gap_features(df, extracted_fvgs, atr)

    # Analyze historical FVG reactions
    reaction_stats = predict_fvg_reaction(df, extracted_fvgs)

    # Add FVG success metrics if we have data
    if not reaction_stats.empty:
        # Calculate average bounce success rate for bullish FVGs
        bull_bounce_rate = reaction_stats[reaction_stats['type'] == 'bullish']['bounced'].mean()
        df['bull_fvg_bounce_prob'] = bull_bounce_rate if not pd.isna(bull_bounce_rate) else 0.5

        # Calculate average bounce success rate for bearish FVGs
        bear_bounce_rate = reaction_stats[reaction_stats['type'] == 'bearish']['bounced'].mean()
        df['bear_fvg_bounce_prob'] = bear_bounce_rate if not pd.isna(bear_bounce_rate) else 0.5

        # Calculate average bounce magnitude relative to FVG size
        bull_bounce_mag = reaction_stats[reaction_stats['type'] == 'bullish']['bounce_ratio'].mean()
        df['bull_fvg_bounce_magnitude'] = bull_bounce_mag if not pd.isna(bull_bounce_mag) else 0.0

        bear_bounce_mag = reaction_stats[reaction_stats['type'] == 'bearish']['bounce_ratio'].mean()
        df['bear_fvg_bounce_magnitude'] = bear_bounce_mag if not pd.isna(bear_bounce_mag) else 0.0

    # Return the enhanced dataframe
    return df


def identify_fvg_confluence_zones(df, lookback=20):
    """
    Identify areas where FVGs from different timeframes overlap, creating strong zones

    Args:
        df: DataFrame with FVG features
        lookback: Number of bars to look back for active FVGs

    Returns:
        DataFrame with confluence zones added
    """
    df_out = df.copy()

    # Pre-allocate arrays
    confluence_zones = []
    confluence_strength = np.zeros(len(df))

    # Find areas where multiple FVGs overlap
    for i in range(lookback, len(df)):
        # Look at all active FVGs in the lookback window
        active_fvgs = []

        for j in range(i-lookback, i+1):
            # Get bullish FVGs
            if df.at[j, 'fvg_bull_present'] == 1:
                for idx, (top, bottom) in enumerate(zip(df.at[j, 'fvg_top_prices'], df.at[j, 'fvg_bottom_prices'])):
                    active_fvgs.append({
                        'type': 'bullish',
                        'top': top,
                        'bottom': bottom,
                        'age': i - j
                    })

            # Get bearish FVGs
            if df.at[j, 'fvg_bear_present'] == 1:
                for idx, (top, bottom) in enumerate(zip(df.at[j, 'fvg_top_prices'], df.at[j, 'fvg_bottom_prices'])):
                    active_fvgs.append({
                        'type': 'bearish',
                        'top': top,
                        'bottom': bottom,
                        'age': i - j
                    })

        # Check for overlapping FVGs
        if len(active_fvgs) >= 2:
            overlaps = []

            for idx1, fvg1 in enumerate(active_fvgs):
                for idx2, fvg2 in enumerate(active_fvgs[idx1+1:], idx1+1):
                    # Check if the FVGs overlap
                    if (fvg1['bottom'] <= fvg2['top'] and fvg1['top'] >= fvg2['bottom']) or \
                       (fvg2['bottom'] <= fvg1['top'] and fvg2['top'] >= fvg1['bottom']):
                        # Calculate overlap area
                        overlap_bottom = max(fvg1['bottom'], fvg2['bottom'])
                        overlap_top = min(fvg1['top'], fvg2['top'])
                        overlap_size = overlap_top - overlap_bottom

                        if overlap_size > 0:
                            overlaps.append({
                                'bottom': overlap_bottom,
                                'top': overlap_top,
                                'size': overlap_size,
                                'types': [fvg1['type'], fvg2['type']]
                            })

            # Store strongest overlap
            if overlaps:
                strongest_overlap = max(overlaps, key=lambda x: x['size'])
                confluence_zones.append({
                    'idx': i,
                    'bottom': strongest_overlap['bottom'],
                    'top': strongest_overlap['top'],
                    'size': strongest_overlap['size'],
                    'types': strongest_overlap['types']
                })

                # Set confluence strength based on overlap size and number of overlapping FVGs
                confluence_strength[i] = strongest_overlap['size'] * len(strongest_overlap['types'])

    # Add confluence strength to dataframe
    df_out['fvg_confluence_strength'] = confluence_strength

    # Add normalized confluence strength (0-1)
    max_strength = np.max(confluence_strength) if np.max(confluence_strength) > 0 else 1.0
    df_out['fvg_confluence_strength_normalized'] = confluence_strength / max_strength

    return df_out, confluence_zones