"""Chart visualization for price data and SMC patterns"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_combined_analysis(df, tops, bottoms, atr, bos_events, order_blocks, fvg_events,
                           start_idx=0, end_idx=None, atr_period=14, atr_multiplier=1.5):
    """
    Create a comprehensive chart with price data, swing points, BOS events, OBs, and FVGs

    Args:
        df: The DataFrame with price data
        tops: List of top swing points [detection_index, max_index, price]
        bottoms: List of bottom swing points [detection_index, min_index, price]
        atr: Array of ATR values
        bos_events: Dictionary with bullish and bearish BOS events
        order_blocks: Dictionary with bullish and bearish order blocks
        fvg_events: Dictionary with bullish and bearish FVG events
        start_idx: Start index for the plot
        end_idx: End index for the plot
        atr_period: ATR period used for calculation
        atr_multiplier: ATR multiplier used for calculation
    """
    if end_idx is None:
        end_idx = len(df)

    # Take subset of data to display
    df_subset = df.iloc[start_idx:end_idx].reset_index(drop=True)
    atr_subset = atr[start_idx:end_idx]

    # Increase DPI for higher resolution
    plt.rcParams['figure.dpi'] = 300

    # Create figure with white background and 2 subplots (price and ATR)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   facecolor='white')

    # Set background color for both axes
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')

    # Define colors
    bullish_color = '#2E7D32'  # Dark green
    bearish_color = '#C62828'  # Dark red
    bos_bullish_color = 'green'
    bos_bearish_color = 'red'
    bullish_ob_color = (144 / 255, 238 / 255, 144 / 255, 0.5)  # Light green (more transparent)
    bearish_ob_color = (255 / 255, 182 / 255, 193 / 255, 0.5)  # Light pink (more transparent)
    bullish_fvg_color = (0 / 255, 255 / 255, 0 / 255, 0.3)  # Green with transparency
    bearish_fvg_color = (255 / 255, 0 / 255, 0 / 255, 0.3)  # Red with transparency
    mitigated_color = (128 / 255, 128 / 255, 128 / 255, 0.5)  # Gray with transparency

    # Map original indices to plot indices
    idx_map = {orig_idx: plot_idx for plot_idx, orig_idx in enumerate(range(start_idx, end_idx))}

    # Draw order blocks first (behind everything else)
    # Plot bullish order blocks
    for ob_idx, swing_idx, break_idx, mitigation_idx in order_blocks['bullish']:
        if start_idx <= ob_idx < end_idx:
            # Convert to plot indices
            plot_ob_idx = idx_map.get(ob_idx)

            # Define the end of the order block (either mitigation point or end of view)
            mitigation_plot_idx = None
            if mitigation_idx < end_idx:
                mitigation_plot_idx = idx_map.get(mitigation_idx)
            else:
                mitigation_plot_idx = len(df_subset) - 1

            if plot_ob_idx is not None and mitigation_plot_idx is not None:
                # Draw rectangle from order block candle to mitigation point
                rect = patches.Rectangle(
                    (plot_ob_idx, df.iloc[ob_idx]['low']),
                    mitigation_plot_idx - plot_ob_idx,  # Extend to mitigation point
                    df.iloc[ob_idx]['high'] - df.iloc[ob_idx]['low'],
                    linewidth=0, edgecolor='none', facecolor=bullish_ob_color, zorder=1
                )
                ax1.add_patch(rect)

                # Add OB label
                mid_x = plot_ob_idx + 2  # Just a bit to the right
                mid_y = (df.iloc[ob_idx]['high'] + df.iloc[ob_idx]['low']) / 2
                ax1.annotate('OB', (mid_x, mid_y), ha='center', va='center',
                             color=bos_bullish_color, fontsize=8, fontweight='bold')

    # Plot bearish order blocks
    for ob_idx, swing_idx, break_idx, mitigation_idx in order_blocks['bearish']:
        if start_idx <= ob_idx < end_idx:
            # Convert to plot indices
            plot_ob_idx = idx_map.get(ob_idx)

            # Define the end of the order block (either mitigation point or end of view)
            mitigation_plot_idx = None
            if mitigation_idx < end_idx:
                mitigation_plot_idx = idx_map.get(mitigation_idx)
            else:
                mitigation_plot_idx = len(df_subset) - 1

            if plot_ob_idx is not None and mitigation_plot_idx is not None:
                # Draw rectangle from order block candle to mitigation point
                rect = patches.Rectangle(
                    (plot_ob_idx, df.iloc[ob_idx]['low']),
                    mitigation_plot_idx - plot_ob_idx,  # Extend to mitigation point
                    df.iloc[ob_idx]['high'] - df.iloc[ob_idx]['low'],
                    linewidth=0, edgecolor='none', facecolor=bearish_ob_color, zorder=1
                )
                ax1.add_patch(rect)

                # Add OB label
                mid_x = plot_ob_idx + 2  # Just a bit to the right
                mid_y = (df.iloc[ob_idx]['high'] + df.iloc[ob_idx]['low']) / 2
                ax1.annotate('OB', (mid_x, mid_y), ha='center', va='center',
                             color=bos_bearish_color, fontsize=8, fontweight='bold')

    # Plot Fair Value Gaps (similar to order blocks)
    # Bullish FVGs
    for start_idx_fvg, end_idx_fvg, top_price, bottom_price, mitigated, mitigation_idx in fvg_events['bullish']:
        if start_idx <= start_idx_fvg < end_idx and start_idx <= end_idx_fvg < end_idx:
            # Convert to plot indices
            plot_start_idx = idx_map.get(start_idx_fvg)
            plot_end_idx = idx_map.get(end_idx_fvg)

            # Determine fill color based on mitigation status
            fill_color = mitigated_color if mitigated else bullish_fvg_color

            # Define the end point of the FVG visualization
            mitigation_plot_idx = None
            if mitigated and mitigation_idx is not None and mitigation_idx < end_idx:
                mitigation_plot_idx = idx_map.get(mitigation_idx)
            else:
                mitigation_plot_idx = len(df_subset) - 1

            if plot_start_idx is not None and plot_end_idx is not None and mitigation_plot_idx is not None:
                # Draw rectangle for the FVG - start at the middle candle (i-1) where the gap actually occurs
                rect = patches.Rectangle(
                    (plot_start_idx + 1, bottom_price),  # Starting at i-1 (one candle to the right of i-2)
                    mitigation_plot_idx - (plot_start_idx + 1),  # Extend to mitigation point or end
                    top_price - bottom_price,
                    linewidth=1, edgecolor='green', facecolor=fill_color, zorder=1, linestyle='-'
                )
                ax1.add_patch(rect)

                # Add FVG label
                ax1.annotate('FVG', (plot_start_idx + 3, (top_price + bottom_price) / 2),
                             ha='left', va='center', color='green', fontsize=8, fontweight='bold')

    # Bearish FVGs
    for start_idx_fvg, end_idx_fvg, top_price, bottom_price, mitigated, mitigation_idx in fvg_events['bearish']:
        if start_idx <= start_idx_fvg < end_idx and start_idx <= end_idx_fvg < end_idx:
            # Convert to plot indices
            plot_start_idx = idx_map.get(start_idx_fvg)
            plot_end_idx = idx_map.get(end_idx_fvg)

            # Determine fill color based on mitigation status
            fill_color = mitigated_color if mitigated else bearish_fvg_color

            # Define the end point of the FVG visualization
            mitigation_plot_idx = None
            if mitigated and mitigation_idx is not None and mitigation_idx < end_idx:
                mitigation_plot_idx = idx_map.get(mitigation_idx)
            else:
                mitigation_plot_idx = len(df_subset) - 1

            if plot_start_idx is not None and plot_end_idx is not None and mitigation_plot_idx is not None:
                # Draw rectangle for the FVG - start at the middle candle (i-1) where the gap actually occurs
                rect = patches.Rectangle(
                    (plot_start_idx + 1, bottom_price),  # Starting at i-1 (one candle to the right of i-2)
                    mitigation_plot_idx - (plot_start_idx + 1),  # Extend to mitigation point or end
                    top_price - bottom_price,
                    linewidth=1, edgecolor='red', facecolor=fill_color, zorder=1, linestyle='-'
                )
                ax1.add_patch(rect)

                # Add FVG label
                ax1.annotate('FVG', (plot_start_idx + 3, (top_price + bottom_price) / 2),
                             ha='left', va='center', color='red', fontsize=8, fontweight='bold')

    # Plot candlesticks on top subplot
    for i in range(len(df_subset)):
        # Calculate candle position and size
        x = i
        open_price = df_subset['open'].iloc[i]
        close_price = df_subset['close'].iloc[i]
        high_price = df_subset['high'].iloc[i]
        low_price = df_subset['low'].iloc[i]

        # Determine candle color
        color = bullish_color if close_price >= open_price else bearish_color

        # Draw the candle body
        ax1.bar(x, abs(close_price - open_price), 0.6,
                bottom=min(open_price, close_price), color=color, zorder=3)

        # Draw the wick
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
    ax2.plot(range(len(atr_subset)), atr_subset, color='#1565C0', linewidth=1.5)  # Darker blue

    # Fill the area under the ATR curve
    ax2.fill_between(range(len(atr_subset)), 0, atr_subset, color='#1565C0', alpha=0.2)

    # Add a horizontal line for average ATR value
    avg_atr = np.mean(atr_subset)
    ax2.axhline(y=avg_atr, color='#795548', linestyle='--', linewidth=1)  # Brown line
    ax2.text(len(atr_subset) * 0.8, avg_atr * 1.1, f'Avg ATR: {avg_atr:.6f}',
             color='black', fontsize=10)

    # Highlight ATR threshold levels
    threshold_line = atr_multiplier * avg_atr
    ax2.axhline(y=threshold_line, color='#006064', linestyle='-.', linewidth=1)  # Teal line
    ax2.text(len(atr_subset) * 0.8, threshold_line * 1.1,
             f'Threshold ({atr_multiplier}x): {threshold_line:.6f}',
             color='black', fontsize=10)

    # Add titles and styling
    ax1.set_title('SMC Analysis: Swing Points, BOS, Order Blocks, and Fair Value Gaps',
                  color='black', fontsize=20, fontweight='bold')
    ax2.set_title('Average True Range (ATR) Volatility Metric', color='black', fontsize=16)

    # Add subtle grid for better readability
    ax1.grid(True, alpha=0.15, linestyle='-')
    ax2.grid(True, alpha=0.15, linestyle='-')

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

    return fig