import unittest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Add the project root to the path to enable imports
sys.path.append(str(Path(__file__).parent.parent))

from smc_trading.features.swing_points import calculate_atr_vectorized, directional_change_adaptive_optimized, \
    mark_swing_points
from smc_trading.features.order_blocks import detect_bos_events_optimized, add_order_block_features
from smc_trading.features.fair_value_gaps import find_fair_value_gaps_optimized, add_fair_value_gap_features
from smc_trading.features.structure import add_advanced_smc_features_optimized
from smc_trading.features.feature_set import extract_all_smc_features
from smc_trading.data.processing import clean_price_data, add_basic_indicators
from smc_trading.data.collection import get_historical_data
from smc_trading.visualization.charting import plot_combined_analysis


class TestFeatureQuality(unittest.TestCase):
    """Test the quality and correctness of the features"""

    @classmethod
    def setUpClass(cls):
        """Download and prepare test data"""
        # Initialize MT5
        if not mt5.initialize():
            print(f"Failed to initialize MT5: {mt5.last_error()}")
            raise RuntimeError("MT5 initialization failed")

        print("MT5 initialized successfully")

        # Download 3 months of data for more robust testing
        cls.symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        cls.timeframes = [mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4]
        timeframe_names = {mt5.TIMEFRAME_H1: "H1", mt5.TIMEFRAME_H4: "H4"}

        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # 90 days of data

        cls.data = {}
        cls.feature_data = {}

        for symbol in cls.symbols:
            cls.data[symbol] = {}
            cls.feature_data[symbol] = {}

            for tf in cls.timeframes:
                print(f"Downloading {symbol} {timeframe_names[tf]} data...")
                df = get_historical_data(symbol, start_date, end_date, tf, mt5)

                if df is not None and len(df) > 0:
                    # Clean and prepare data
                    df = clean_price_data(df)

                    # Store raw data
                    cls.data[symbol][tf] = df

                    # Extract features
                    print(f"Extracting features for {symbol} {timeframe_names[tf]}...")
                    df_features = extract_all_smc_features(df)

                    # Store feature data
                    cls.feature_data[symbol][tf] = df_features
                else:
                    print(f"Failed to download {symbol} {timeframe_names[tf]} data")

        # Create output directory for visualizations
        cls.output_dir = Path(__file__).parent / "test_output"
        cls.output_dir.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """Shutdown MT5 after tests"""
        mt5.shutdown()
        print("MT5 shutdown complete")

    def test_swing_point_detection_accuracy(self):
        """Test if swing points are detected at logical price extremes"""
        for symbol in self.symbols:
            for tf in self.timeframes:
                df = self.feature_data[symbol][tf]

                # Check if swing highs are at local maxima
                for i in range(1, len(df) - 1):
                    if df.loc[i, 'is_swing_high'] == 1:
                        # Check if this really is a local high (with some margin)
                        window_size = 5
                        start_idx = max(0, i - window_size)
                        end_idx = min(len(df), i + window_size + 1)
                        window = df.iloc[start_idx:end_idx]

                        current_high = df.loc[i, 'high']
                        window_max = window['high'].max()

                        self.assertAlmostEqual(current_high, window_max, delta=0.0001,
                                               msg=f"Swing high at index {i} is not at local maximum")

                # Check if swing lows are at local minima
                for i in range(1, len(df) - 1):
                    if df.loc[i, 'is_swing_low'] == 1:
                        window_size = 5
                        start_idx = max(0, i - window_size)
                        end_idx = min(len(df), i + window_size + 1)
                        window = df.iloc[start_idx:end_idx]

                        current_low = df.loc[i, 'low']
                        window_min = window['low'].min()

                        self.assertAlmostEqual(current_low, window_min, delta=0.0001,
                                               msg=f"Swing low at index {i} is not at local minimum")

    def test_order_block_structure(self):
        """Test that order blocks have correct structure"""
        for symbol in self.symbols:
            for tf in self.timeframes:
                df = self.feature_data[symbol][tf]

                # Get original data for swing point detection
                orig_df = self.data[symbol][tf]
                high = orig_df['high'].values
                low = orig_df['low'].values
                close = orig_df['close'].values

                tops, bottoms, atr = directional_change_adaptive_optimized(
                    close, high, low,
                    atr_period=14,
                    atr_multiplier=1.5
                )

                # Get BOS events and order blocks
                bos_events, order_blocks = detect_bos_events_optimized(
                    mark_swing_points(orig_df, tops, bottoms)
                )

                # Test bullish order blocks
                for ob_idx, swing_idx, break_idx, _ in order_blocks['bullish']:
                    # Verify order block is a bearish candle (close < open)
                    self.assertLess(df.loc[ob_idx, 'close'], df.loc[ob_idx, 'open'],
                                    msg=f"Bullish order block at {ob_idx} is not a bearish candle")

                    # Verify order block is between swing high and BOS
                    self.assertLess(swing_idx, ob_idx,
                                    msg=f"Order block at {ob_idx} is not after swing high at {swing_idx}")
                    self.assertLess(ob_idx, break_idx,
                                    msg=f"Order block at {ob_idx} is not before BOS at {break_idx}")

                # Test bearish order blocks
                for ob_idx, swing_idx, break_idx, _ in order_blocks['bearish']:
                    # Verify order block is a bullish candle (close > open)
                    self.assertGreater(df.loc[ob_idx, 'close'], df.loc[ob_idx, 'open'],
                                       msg=f"Bearish order block at {ob_idx} is not a bullish candle")

                    # Verify order block is between swing low and BOS
                    self.assertLess(swing_idx, ob_idx,
                                    msg=f"Order block at {ob_idx} is not after swing low at {swing_idx}")
                    self.assertLess(ob_idx, break_idx,
                                    msg=f"Order block at {ob_idx} is not before BOS at {break_idx}")

    def test_fair_value_gap_identification(self):
        """Test that FVGs represent actual gaps in price"""
        for symbol in self.symbols:
            for tf in self.timeframes:
                df = self.data[symbol][tf]

                # Find Fair Value Gaps
                fvg_events = find_fair_value_gaps_optimized(df)

                # Test bullish FVGs
                for start_idx, end_idx, top_price, bottom_price, _, _ in fvg_events['bullish']:
                    # Verify gap: low of current candle is higher than high of candle before previous one
                    self.assertGreater(df.loc[end_idx, 'low'], df.loc[start_idx, 'high'],
                                       msg=f"Bullish FVG at {end_idx} does not have a true gap")

                    # Verify top and bottom prices are correct
                    self.assertAlmostEqual(top_price, df.loc[end_idx, 'low'], delta=0.0001)
                    self.assertAlmostEqual(bottom_price, df.loc[start_idx, 'high'], delta=0.0001)

                # Test bearish FVGs
                for start_idx, end_idx, top_price, bottom_price, _, _ in fvg_events['bearish']:
                    # Verify gap: high of current candle is lower than low of candle before previous one
                    self.assertLess(df.loc[end_idx, 'high'], df.loc[start_idx, 'low'],
                                    msg=f"Bearish FVG at {end_idx} does not have a true gap")

                    # Verify top and bottom prices are correct
                    self.assertAlmostEqual(top_price, df.loc[start_idx, 'low'], delta=0.0001)
                    self.assertAlmostEqual(bottom_price, df.loc[end_idx, 'high'], delta=0.0001)

                # Check if bearish FVGs are being detected
                print(f"{symbol} {tf} bearish FVGs: {len(fvg_events['bearish'])}")

    def test_feature_predictive_power(self):
        """Test if features have predictive power for future price moves"""
        for symbol in self.symbols:
            for tf in self.timeframes:
                df = self.feature_data[symbol][tf]

                # Skip if we don't have enough data
                if len(df) < 50:
                    continue

                # Create a simple feature test: when bullish OB present and below price,
                # check if price tends to move down to touch the OB
                df['close_shift_10'] = df['close'].shift(-10)  # Look 10 bars ahead
                df['close_shift_20'] = df['close'].shift(-20)  # Look 20 bars ahead

                # Cases where bullish OB present and below price
                bullish_ob_below = (
                        (df['bullish_ob_present'] == 1) &
                        (df['ob_bull_distance_pct'] < 0)
                )

                if bullish_ob_below.sum() > 5:  # Only test if we have enough samples
                    # Calculate how often price moves down to touch the OB within 10 and 20 bars
                    price_moves_down_10 = (df.loc[bullish_ob_below, 'close_shift_10'] <
                                           df.loc[bullish_ob_below, 'close'])

                    price_moves_down_20 = (df.loc[bullish_ob_below, 'close_shift_20'] <
                                           df.loc[bullish_ob_below, 'close'])

                    success_rate_10 = price_moves_down_10.mean()
                    success_rate_20 = price_moves_down_20.mean()

                    # Print success rates
                    print(f"{symbol} {tf} Bullish OB below success rate (10 bars): {success_rate_10:.2f}")
                    print(f"{symbol} {tf} Bullish OB below success rate (20 bars): {success_rate_20:.2f}")

                    # Create visualization
                    plt.figure(figsize=(12, 8))

                    # Plot price
                    plt.subplot(2, 1, 1)
                    plt.plot(df.index, df['close'])

                    # Mark where bullish OBs below price are
                    ob_indices = df[bullish_ob_below].index
                    plt.scatter(ob_indices, df.loc[ob_indices, 'close'],
                                color='green', marker='^', s=100)

                    plt.title(f"{symbol} {tf} Bullish OB Below Price")
                    plt.ylabel("Price")

                    # Plot success rate bar chart
                    plt.subplot(2, 1, 2)
                    plt.bar(['10 bars', '20 bars'], [success_rate_10, success_rate_20])
                    plt.axhline(0.5, linestyle='--', color='red')
                    plt.ylim(0, 1)
                    plt.title("Success Rate (price moves down)")
                    plt.ylabel("Probability")

                    # Save plot
                    plt.tight_layout()
                    plt.savefig(self.output_dir / f"{symbol}_{tf}_bullish_ob_test.png")
                    plt.close()

    def test_pattern_visualization(self):
        """Visual verification of detected patterns"""
        for symbol in self.symbols:
            for tf in self.timeframes:
                df = self.data[symbol][tf]
                df_features = self.feature_data[symbol][tf]

                # Extract components needed for visualization
                high = df['high'].values
                low = df['low'].values
                close = df['close'].values

                tops, bottoms, atr = directional_change_adaptive_optimized(
                    close, high, low,
                    atr_period=14,
                    atr_multiplier=1.5
                )

                marked_df = mark_swing_points(df, tops, bottoms)
                bos_events, order_blocks = detect_bos_events_optimized(marked_df)
                fvg_events = find_fair_value_gaps_optimized(df)

                # Create visualization for a sample window
                window_start = len(df) - 100  # Last 100 bars
                window_end = len(df)

                try:
                    fig = plot_combined_analysis(
                        df, tops, bottoms, atr, bos_events, order_blocks, fvg_events,
                        start_idx=window_start, end_idx=window_end
                    )

                    # Save visualization
                    plt.savefig(self.output_dir / f"{symbol}_{tf}_pattern_visualization.png")
                    plt.close(fig)
                except Exception as e:
                    print(f"Error creating visualization for {symbol} {tf}: {e}")

    def test_feature_distributions(self):
        """Analyze feature distributions to identify outliers or issues"""
        for symbol in self.symbols:
            for tf in self.timeframes:
                df = self.feature_data[symbol][tf]

                # Select key features to analyze
                key_features = [
                    'ob_bull_distance_pct', 'ob_bear_distance_pct',
                    'fvg_bull_distance_pct', 'fvg_bear_distance_pct',
                    'structure_trend', 'ob_fvg_confluence'
                ]

                # Filter to features that exist in the dataset
                features_to_analyze = [f for f in key_features if f in df.columns]

                if not features_to_analyze:
                    continue

                try:
                    # Create distribution plots
                    plt.figure(figsize=(15, 10))

                    for i, feature in enumerate(features_to_analyze):
                        # Skip if all values are NaN
                        if df[feature].isna().all():
                            continue

                        plt.subplot(2, 3, i + 1)

                        # For categorical features, use countplot
                        if feature == 'structure_trend':
                            sns.countplot(x=df[feature].dropna())
                        else:
                            # For continuous features, use histogram
                            sns.histplot(df[feature].dropna(), kde=True)

                        plt.title(feature)

                        # For distance percentages, add basic stats
                        if '_distance_pct' in feature:
                            mean = df[feature].mean()
                            median = df[feature].median()
                            plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
                            plt.axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}')
                            plt.legend()

                    plt.tight_layout()
                    plt.savefig(self.output_dir / f"{symbol}_{tf}_feature_distributions.png")
                    plt.close()

                except Exception as e:
                    print(f"Error creating distribution plots for {symbol} {tf}: {e}")

    def test_feature_correlations(self):
        """Analyze feature correlations to identify relationships"""
        for symbol in self.symbols:
            for tf in self.timeframes:
                df = self.feature_data[symbol][tf]

                # Select key features for correlation analysis
                key_features = [
                    'ob_bull_distance_pct', 'ob_bear_distance_pct',
                    'fvg_bull_distance_pct', 'fvg_bear_distance_pct',
                    'structure_trend', 'ob_fvg_confluence',
                    'atr', 'atr_z_score', 'pattern_freshness'
                ]

                # Filter to features that exist in the dataset
                features_to_analyze = [f for f in key_features if f in df.columns]

                if len(features_to_analyze) < 2:
                    continue

                try:
                    # Create correlation matrix
                    corr_matrix = df[features_to_analyze].corr().round(2)

                    # Plot correlation heatmap
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                    plt.title(f"{symbol} {tf} Feature Correlations")
                    plt.tight_layout()
                    plt.savefig(self.output_dir / f"{symbol}_{tf}_feature_correlations.png")
                    plt.close()

                except Exception as e:
                    print(f"Error creating correlation plot for {symbol} {tf}: {e}")

    def test_features_against_future_returns(self):
        """Test features against future returns to see if they have predictive power"""
        for symbol in self.symbols:
            for tf in self.timeframes:
                df = self.feature_data[symbol][tf].copy()

                # Calculate future returns for different horizons
                for horizon in [1, 5, 10, 20]:
                    df[f'future_return_{horizon}'] = (df['close'].shift(-horizon) / df['close'] - 1) * 100

                # Select key features
                key_features = [
                    'bullish_ob_present', 'bearish_ob_present',
                    'fvg_bull_present', 'fvg_bear_present',
                    'structure_trend', 'bos_confirmation',
                    'ob_fvg_confluence', 'pattern_freshness'
                ]

                # Filter to features that exist in the dataset
                features_to_analyze = [f for f in key_features if f in df.columns]

                if not features_to_analyze:
                    continue

                try:
                    # Create a summary report
                    plt.figure(figsize=(15, 10))

                    for i, feature in enumerate(features_to_analyze[:4]):  # Limit to 4 features
                        if df[feature].nunique() <= 1:
                            continue

                        plt.subplot(2, 2, i + 1)

                        # For binary features
                        if df[feature].nunique() <= 2:
                            feature_1 = df[df[feature] == 1]['future_return_10'].dropna()
                            feature_0 = df[df[feature] == 0]['future_return_10'].dropna()

                            # Skip if not enough data
                            if len(feature_1) < 5 or len(feature_0) < 5:
                                continue

                            # Create boxplot
                            data = [feature_1, feature_0]
                            plt.boxplot(data, labels=['Present', 'Absent'])
                            plt.title(f"{feature} vs 10-bar Future Return")
                            plt.ylabel("Return %")

                            # Add mean values
                            mean_1 = feature_1.mean()
                            mean_0 = feature_0.mean()
                            plt.scatter([1, 2], [mean_1, mean_0], color='red', marker='*', s=100)
                            plt.text(1, mean_1, f"{mean_1:.2f}%", ha='right')
                            plt.text(2, mean_0, f"{mean_0:.2f}%", ha='left')

                        # For multi-category features
                        else:
                            # Discretize the feature if too many values
                            if df[feature].nunique() > 5:
                                df[f'{feature}_bin'] = pd.qcut(df[feature], 5, labels=False, duplicates='drop')
                                grouped = df.groupby(f'{feature}_bin')['future_return_10'].mean()
                            else:
                                grouped = df.groupby(feature)['future_return_10'].mean()

                            grouped.plot(kind='bar')
                            plt.title(f"{feature} vs 10-bar Future Return")
                            plt.ylabel("Mean Return %")

                    plt.tight_layout()
                    plt.savefig(self.output_dir / f"{symbol}_{tf}_feature_vs_returns.png")
                    plt.close()

                except Exception as e:
                    print(f"Error creating returns analysis for {symbol} {tf}: {e}")


if __name__ == '__main__':
    unittest.main()