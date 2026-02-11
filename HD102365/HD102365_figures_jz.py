# HD102365_figures.py
#
# Interactive plots on a server:
#   - Set SAVE_INTERACTIVE_HTML = True below; install mpld3 (pip install mpld3).
#     Script will also write .html files. Download them and open in a browser
#     for zoom, pan, and (where supported) hover.
#   - Or use X11 forwarding: ssh -X user@server, then set backend and use
#     plt.show() instead of/in addition to savefig (window opens on your machine).

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

import matplotlib as mpl

# Set True to also save interactive HTML (requires: pip install mpld3)
SAVE_INTERACTIVE_HTML = False

try:
    import mpld3
except ImportError:
    mpld3 = None

THESIS_FONTSIZE = 17
SMALL = 9


def _save_interactive_html(fig, png_path):
    """If SAVE_INTERACTIVE_HTML is True and mpld3 is available, save an HTML copy."""
    if not SAVE_INTERACTIVE_HTML or mpld3 is None:
        return
    html_path = png_path.replace(".png", ".html")
    try:
        mpld3.save_html(fig, html_path)
        print(f"  Interactive: {html_path}")
    except Exception as e:
        print(f"  Warning: could not save HTML: {e}")


mpl.rcParams.update({
    "font.size": THESIS_FONTSIZE,
    "axes.titlesize": THESIS_FONTSIZE + 2,
    "axes.labelsize": THESIS_FONTSIZE,
    "xtick.labelsize": THESIS_FONTSIZE - 1,
    "ytick.labelsize": THESIS_FONTSIZE - 1,
    "legend.fontsize": THESIS_FONTSIZE - 6,
    "figure.titlesize": THESIS_FONTSIZE + 3,
    "savefig.dpi": 300,
})

def load_dat_files(data_dir):
    """Load data directly from .dat files created by the HD102365 processing script"""
    
    all_data = []
    
    # Look for all HD102365*.dat files
    dat_files = [f for f in os.listdir(data_dir) if f.startswith('HD102365') and f.endswith('.dat')]
    dat_files.sort()
    
    if not dat_files:
        raise FileNotFoundError(f"No HD102365*.dat files found in {data_dir}")
    
    print(f"Found {len(dat_files)} .dat files")
    
    # Group files by instrument and measurement type
    # File format: HD102365_{instrument}_{measurement}.dat
    file_groups = {}
    for filename in dat_files:
        parts = filename.replace('HD102365_', '').replace('.dat', '').split('_')
        
        if len(parts) == 2:
            instrument = parts[0]
            measurement = parts[1]
            
            if instrument not in file_groups:
                file_groups[instrument] = {}
            file_groups[instrument][measurement] = filename
    
    # Process each instrument
    for instrument in sorted(file_groups.keys()):
        print(f"Processing {instrument}...")
        inst_data = None
        
        for measurement, filename in sorted(file_groups[instrument].items()):
            filepath = os.path.join(data_dir, filename)
            
            try:
                # Read: time, value, error, jitter_flag, offset_flag, subset_flag
                data = np.loadtxt(filepath)
                
                # Create base DataFrame
                df = pd.DataFrame({
                    'Time [BJD]': data[:, 0],
                    'offset_flag': data[:, 4].astype(int),
                    'Instrument': instrument
                })
                
                # Add measurement-specific columns
                if measurement == 'RV':
                    df['RV [m/s]'] = data[:, 1]
                    df['RV Err. [m/s]'] = data[:, 2]
                elif measurement == 'BIS':
                    df['BIS [m/s]'] = data[:, 1]
                    df['BIS Err. [m/s]'] = data[:, 2]
                elif measurement == 'FWHM':
                    df['CCF FWHM [m/s]'] = data[:, 1]
                    df['CCF FWHM Err. [m/s]'] = data[:, 2]
                elif measurement == 'Contrast':
                    df['CCF Contrast'] = data[:, 1]
                    df['CCF Contrast Err.'] = data[:, 2]
                elif measurement == 'EWHa':
                    df['EW H-alpha'] = data[:, 1]
                    df['EW H-alpha Err.'] = data[:, 2]
                elif measurement == 'SHK':
                    df['S_HK Index'] = data[:, 1]
                    df['S_HK Err.'] = data[:, 2]
                elif measurement == 'RHK':
                    df['RHK'] = data[:, 1]
                    df['RHK Err.'] = data[:, 2]
                
                # Merge with existing instrument data
                if inst_data is None:
                    inst_data = df
                else:
                    inst_data = pd.merge(
                        inst_data, df,
                        on=['Time [BJD]', 'offset_flag', 'Instrument'],
                        how='outer'
                    )
                
            except Exception as e:
                print(f"  Error loading {filepath}: {e}")
                continue
        
        if inst_data is not None:
            all_data.append(inst_data)
            print(f"  Loaded {len(inst_data)} points")
    
    # Combine all instruments
    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal: {len(df_all)} points from {len(all_data)} instruments")
        return df_all
    else:
        raise ValueError("No data loaded!")


def spectral_window_power(t, freq):
    """
    Spectral window (squared magnitude of Fourier transform of the sampling).
    Returns power in [0, 1], same length as freq. Use same time series t and
    frequency grid freq as the data periodogram.
    """
    # W(f) = (1/N^2) * | sum_n exp(2*pi*i*f*t_n) |^2
    exp_terms = np.exp(2j * np.pi * np.outer(freq, t))
    window_power = np.abs(exp_terms.sum(axis=1)) ** 2 / (len(t) ** 2)
    return window_power


def plot_activity_with_periodograms(instrument, inst_df, fig_dir):
    """
    Plot activity indicators with Lomb-Scargle periodograms.
    Left column: Activity data, Right column: Periodograms
    Only includes rows where data exists.
    """
    
    # Define plot information: (column_name, error_column, y_label)
    plot_info = [
        ("RV [m/s]", "RV Err. [m/s]", "RV [m/s]"),
        ("CCF Contrast", "CCF Contrast Err.", "CCF Contrast"),
        ("CCF FWHM [m/s]", "CCF FWHM Err. [m/s]", "CCF FWHM [m/s]"),
        ("BIS [m/s]", "BIS Err. [m/s]", "BIS [m/s]"),
        ("EW H-alpha", "EW H-alpha Err.", "EW H-alpha"),
        ("S_HK Index", "S_HK Err.", "S_HK Index"),
        ("RHK", "RHK Err.", "RHK Index"),
    ]
    
    # Filter to only include indicators with data
    valid_plots = []
    for col, err_col, ylabel in plot_info:
        if col in inst_df.columns and not inst_df[col].isna().all():
            valid_data = inst_df[inst_df[col].notna()]
            if len(valid_data) > 0:
                valid_plots.append((col, err_col, ylabel))
                print(f"  {col}: {len(valid_data)} points")
    
    if len(valid_plots) == 0:
        print(f"No valid data for {instrument}, skipping plot.")
        return
    
    # Create figure with only the necessary rows
    n_rows = len(valid_plots)
    fig, axes = plt.subplots(n_rows, 2, figsize=(18, 3*n_rows))
    
    # Handle case where there's only one row (axes won't be 2D)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Use skyblue color for data points
    data_color = "skyblue"
    
    for i, (col, err_col, ylabel) in enumerate(valid_plots):
        ax_data = axes[i, 0]      # Left column for data
        ax_period = axes[i, 1]    # Right column for periodogram
        
        # Filter valid data
        valid_data = inst_df[inst_df[col].notna()].copy()
        
        # === LEFT SIDE: DATA PLOT ===
        # Center data by median
        median_val = valid_data[col].median()
        y_centered = valid_data[col] - median_val
        
        # Handle error bars
        if err_col in valid_data.columns:
            error_values = valid_data[err_col]
        else:
            error_values = None
        
        ax_data.errorbar(
            valid_data["Time [BJD]"], y_centered,
            yerr=error_values, fmt="o",
            color=data_color, label=instrument,
            alpha=0.7, markersize=5
        )
        
        ax_data.set_ylabel(f"{ylabel} - median")
        ax_data.grid(True, alpha=0.3)
        ax_data.legend(loc='best')
        
        # === RIGHT SIDE: LOMB-SCARGLE PERIODOGRAM ===
        try:
            # Prepare data for periodogram
            time_data = valid_data["Time [BJD]"].values
            y_data = valid_data[col].values
            
            # Handle error values
            if err_col in valid_data.columns:
                dy_data = valid_data[err_col].values
            else:
                dy_data = np.ones_like(y_data)
            
            # Remove NaN values
            mask = ~(np.isnan(time_data) | np.isnan(y_data) | np.isnan(dy_data))
            t = time_data[mask]
            y = y_data[mask]
            dy = dy_data[mask]
            
            if len(t) > 5:  # Need sufficient points for periodogram
                # Fix non-positive errors
                m = dy > 0
                if not m.all():
                    repl = np.median(dy[m]) if m.any() else 1.0
                    dy[~m] = repl
                
                span = t.max() - t.min()
                if span > 1:  # Need reasonable time span
                    # Define period range
                    min_period = 1.1
                    max_period = max(2.0, 0.8 * span)
                    f_min = 1.0 / max_period
                    f_max = 1.0 / min_period
                    
                    # Create frequency grid
                    N = 15000
                    freq = np.linspace(f_min, f_max, N)
                    
                    # Compute Lomb-Scargle periodogram
                    ls = LombScargle(t, y, dy)
                    power = ls.power(freq)
                    
                    # === FALSE ALARM PROBABILITY LEVELS ===
                    fap_levels = [0.001, 0.01, 0.1]  # 0.1%, 1%, 10%
                    fap_colors = ['red', 'orange', 'green']
                    fap_labels = ['0.1%', '1%', '10%']
                    fap_powers = []
                    
                    try:
                        for fap in fap_levels:
                            fap_power = ls.false_alarm_level(fap, method='baluev')
                            fap_powers.append(fap_power)
                        print(
                            f"    FAP levels: 0.1%={fap_powers[0]:.3f}, "
                            f"1%={fap_powers[1]:.3f}, 10%={fap_powers[2]:.3f}"
                        )
                    except Exception as fap_error:
                        print(f"    Warning: Could not compute FAP levels: {fap_error}")
                        fap_powers = []
                    
                    # === PEAK DETECTION ===
                    power_threshold = np.percentile(power, 85)
                    max_power = np.max(power)
                    
                    if power_threshold > 0.8 * max_power:
                        power_threshold = 0.5 * max_power
                    
                    min_period_separation = 0.01
                    min_freq_separation = int(
                        len(freq) * min_period_separation / np.log10(f_max/f_min)
                    )
                    
                    try:
                        peak_indices, peak_properties = find_peaks(
                            power,
                            height=float(power_threshold),
                            distance=max(1, min_freq_separation),
                            prominence=float(power_threshold) * 0.1
                        )
                        
                        if len(peak_indices) == 0:
                            lower_threshold = np.percentile(power, 70)
                            peak_indices, peak_properties = find_peaks(
                                power,
                                height=float(lower_threshold),
                                distance=max(1, min_freq_separation),
                                prominence=float(lower_threshold) * 0.05
                            )
                            
                    except Exception as peak_error:
                        print(f"    Peak detection error: {peak_error}")
                        peak_indices = [np.argmax(power)] if len(power) > 0 else []
                        peak_indices = np.array(peak_indices)
                    
                    # === PROCESS DETECTED PEAKS ===
                    if len(peak_indices) > 0:
                        peak_periods = 1.0 / freq[peak_indices]
                        peak_powers_raw = power[peak_indices]
                        
                        # Sort by power (highest first)
                        sorted_indices = np.argsort(peak_powers_raw)[::-1]
                        peak_periods = peak_periods[sorted_indices][:3]  # Top 3 peaks
                        peak_powers_sorted = peak_powers_raw[sorted_indices][:3]
                        
                        # Calculate FAP for each detected peak
                        peak_faps = []
                        peak_significance = []
                        
                        for peak_power in peak_powers_sorted:
                            try:
                                peak_fap = ls.false_alarm_probability(
                                    peak_power, method='baluev'
                                )
                                peak_faps.append(peak_fap)
                                
                                if len(fap_powers) >= 3:
                                    if peak_power >= fap_powers[0]:
                                        significance = "highly significant"
                                    elif peak_power >= fap_powers[1]:
                                        significance = "significant"
                                    elif peak_power >= fap_powers[2]:
                                        significance = "marginally significant"
                                    else:
                                        significance = "not significant"
                                else:
                                    significance = "unknown"
                                peak_significance.append(significance)
                                
                            except Exception as e:
                                peak_faps.append(np.nan)
                                peak_significance.append("unknown")
                        
                        print(f"    Detected peaks:")
                        for j, (period, power_val, fap, sig) in enumerate(
                            zip(peak_periods, peak_powers_sorted, peak_faps, peak_significance)
                        ):
                            ordinal = ['1st', '2nd', '3rd'][j]
                            if not np.isnan(fap):
                                print(
                                    f"      {ordinal}: {period:.2f}d, "
                                    f"Power={power_val:.4f}, FAP={fap:.2e} ({sig})"
                                )
                            else:
                                print(
                                    f"      {ordinal}: {period:.2f}d, "
                                    f"Power={power_val:.4f}"
                                )
                                
                    else:
                        peak_periods = []
                        peak_powers_sorted = []
                        peak_faps = []
                        peak_significance = []
                    
                    # Convert to periods for plotting
                    periods = 1.0 / freq
                    
                    # Spectral window: Fourier transform of the sampling pattern
                    power_window = spectral_window_power(t, freq)
                    power_window_scaled = power_window / np.max(power_window) * np.max(power)
                    ax_period.semilogx(
                        periods, -power_window_scaled,
                        color='red', linestyle='-', alpha=0.8, linewidth=1.5,
                        label='Window'
                    )
                    # Plot periodogram
                    ax_period.semilogx(periods, power, 'k-', linewidth=1)
                    
                    # Plot FALSE ALARM PROBABILITY reference lines
                    for fap_power, fap_color, fap_label in zip(
                        fap_powers, fap_colors, fap_labels
                    ):
                        ax_period.axhline(
                            fap_power, color=fap_color, linestyle='--',
                            alpha=0.7, linewidth=1.5,
                            label=f'{fap_label} FAP'
                        )
                    
                    # Plot detected peaks with their FAP values
                    peak_colors = ['purple', 'blue', 'cyan']
                    peak_styles = ['-', '--', ':']
                    
                    for j, (peak_period, peak_power, peak_fap, significance) in enumerate(
                        zip(peak_periods, peak_powers_sorted, peak_faps, peak_significance)
                    ):
                        if j < len(peak_colors):
                            ordinal = ['1st', '2nd', '3rd'][j]
                            if not np.isnan(peak_fap):
                                label = (
                                    f'{ordinal}: {peak_period:.1f}d (FAP={peak_fap:.1e})'
                                )
                            else:
                                label = f'{ordinal}: {peak_period:.1f}d'
                            
                            ax_period.axvline(
                                peak_period, color=peak_colors[j],
                                ls=peak_styles[j], lw=2, alpha=0.8,
                                label=label
                            )
                    
                    # Add reference lines for common periods
                    reference_periods = [1, 7, 14, 28, 100, 365]
                    for ref_period in reference_periods:
                        if min_period <= ref_period <= max_period:
                            ax_period.axvline(
                                ref_period, color='blue', alpha=0.3,
                                linestyle=':', linewidth=1
                            )
                            ax_period.text(
                                ref_period, ax_period.get_ylim()[1]*0.85,
                                f'{ref_period}d', rotation=90, ha='right',
                                va='top', fontsize=SMALL, alpha=0.6, color='blue'
                            )
                    
                    ax_period.set_xlabel("Period [days]")
                    ax_period.set_ylabel("LS Power")
                    ax_period.set_title(f"{ylabel} Periodogram")
                    ax_period.grid(True, alpha=0.3)
                    
                    # Add legend (FAP lines + peaks)
                    ax_period.legend(fontsize=SMALL, loc='upper right')
                    
                    # Add statistics text
                    if len(peak_periods) > 0:
                        peak_info = []
                        for j, (p, pow, fap) in enumerate(
                            zip(peak_periods, peak_powers_sorted, peak_faps)
                        ):
                            if not np.isnan(fap):
                                peak_info.append(
                                    f'Peak {j+1}: {p:.1f}d (FAP={fap:.1e})'
                                )
                            else:
                                peak_info.append(
                                    f'Peak {j+1}: {p:.1f}d (P={pow:.3f})'
                                )
                        peak_text = '\n'.join(peak_info)
                        stats_text = f'N={len(t)} points\nSpan={span:.1f}d\n{peak_text}'
                    else:
                        stats_text = (
                            f'N={len(t)} points\nSpan={span:.1f}d\nNo significant peaks'
                        )
                    
                    ax_period.text(
                        0.02, 0.75, stats_text,
                        transform=ax_period.transAxes, fontsize=SMALL,
                        verticalalignment='top', alpha=0.9,
                        bbox=dict(
                            boxstyle="round,pad=0.4", facecolor="white",
                            alpha=0.9, edgecolor='gray', linewidth=0.5
                        )
                    )
                    
                else:
                    ax_period.text(
                        0.5, 0.5, f"Insufficient time span\n({span:.1f} days)",
                        ha='center', va='center', transform=ax_period.transAxes
                    )
            else:
                ax_period.text(
                    0.5, 0.5, f"Insufficient data\n({len(t)} points)",
                    ha='center', va='center', transform=ax_period.transAxes
                )
                
        except Exception as e:
            ax_period.text(
                0.5, 0.5, f"Error computing\nperiodogram:\n{str(e)[:50]}...",
                ha='center', va='center', transform=ax_period.transAxes, fontsize=SMALL
            )
            print(f"    Error: {str(e)}")
    
    # Set x-labels for bottom row
    axes[-1, 0].set_xlabel("Time [BJD]")
    axes[-1, 1].set_xlabel("Period [days]")
    
    # Set overall title
    fig.suptitle(
        f"HD102365 - {instrument} - Activity Indicators & Lomb-Scargle Periodograms")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    fig_path = os.path.join(
        fig_dir, f"HD102365_{instrument}_activity_LS_periodograms.png"
    )
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    _save_interactive_html(fig, fig_path)
    plt.close(fig)
    print(f"  Saved: {fig_path}")

# ---------------------------------------------------------------------
# OLD non-ESPRESSO combined RV function
# ---------------------------------------------------------------------
def plot_combined_rv_non_espresso_with_periodogram(df_all, fig_dir):
    """
    Combined RV time series + periodogram for NON-ESPRESSO instruments only.
    """

    if "Instrument" not in df_all.columns:
        print("No 'Instrument' column found in df_all, cannot filter ESPRESSO.")
        return

    mask_non_espresso = ~df_all["Instrument"].astype(str).str.contains(
        "ESPRESSO", case=False, na=False
    )
    df_non_espresso = df_all[mask_non_espresso].copy()

    if df_non_espresso.empty:
        print(
            "All data are ESPRESSO or no non-ESPRESSO RVs available, "
            "skipping combined RV plot."
        )
        return

    if "RV [m/s]" not in df_non_espresso.columns:
        print("No 'RV [m/s]' column found, skipping combined RV plot.")
        return

    rv_df = df_non_espresso[~df_non_espresso["RV [m/s]"].isna()].copy()
    if rv_df.empty:
        print("No non-ESPRESSO RV data available for combined RV plot.")
        return

    print(
        f"\nCreating combined RV time series and periodogram (non-ESPRESSO only)..."
    )
    print(
        f"  Total RV points: {len(rv_df)} from "
        f"{rv_df['Instrument'].nunique()} instruments"
    )
    print(f"  Instruments: {sorted(rv_df['Instrument'].dropna().unique())}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    ax_ts, ax_ls = axes

    # ---- LEFT PANEL: TIME SERIES ----
    instruments = sorted(rv_df["Instrument"].dropna().unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(instruments))))
    color_map = dict(zip(instruments, colors))

    for inst in instruments:
        inst_data = rv_df[rv_df["Instrument"] == inst].copy()
        median_inst = inst_data["RV [m/s]"].median()
        y_centered = inst_data["RV [m/s]"] - median_inst

        if "RV Err. [m/s]" in inst_data.columns:
            yerr = inst_data["RV Err. [m/s]"].values
        else:
            yerr = None

        ax_ts.errorbar(
            inst_data["Time [BJD]"],
            y_centered,
            yerr=yerr,
            fmt="o",
            color=color_map[inst],
            label=inst,
            alpha=0.7,
            markersize=5,
            capsize=2,
        )

    ax_ts.set_xlabel("Time [BJD]")
    ax_ts.set_ylabel("RV [m/s] (per-instrument median subtracted)")
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend(loc="best", ncol=2)
    ax_ts.set_title("HD102365 - Combined RVs (All Non-ESPRESSO Instruments)")

    # ---- RIGHT PANEL: LS (same as your original) ----
    time_all = rv_df["Time [BJD]"].values
    rv_all = rv_df["RV [m/s]"].values

    if "RV Err. [m/s]" in rv_df.columns:
        err_all = rv_df["RV Err. [m/s]"].values
    else:
        err_all = np.ones_like(rv_all)

    mask = ~(np.isnan(time_all) | np.isnan(rv_all) | np.isnan(err_all))
    t = time_all[mask]
    y = rv_all[mask]
    dy = err_all[mask]

    if len(t) < 5:
        ax_ls.text(
            0.5,
            0.5,
            f"Insufficient data for periodogram\n(N={len(t)} points)",
            ha="center",
            va="center",
            transform=ax_ls.transAxes,
        )
        print(f"  Not enough points for LS periodogram (N={len(t)})")
    else:
        bad = dy <= 0
        if bad.any():
            repl = np.median(dy[~bad]) if (~bad).any() else 1.0
            dy[bad] = repl

        span = t.max() - t.min()
        print(f"  Time span for LS: {span:.1f} days")

        if span <= 1.0:
            ax_ls.text(
                0.5,
                0.5,
                f"Insufficient time span for periodogram\n(span={span:.1f} days)",
                ha="center",
                va="center",
                transform=ax_ls.transAxes,
            )
        else:
            min_period = 1.1
            max_period = max(2.0, 0.8 * span)
            f_min = 1.0 / max_period
            f_max = 1.0 / min_period

            N = 20000
            freq = np.linspace(f_min, f_max, N)

            ls = LombScargle(t, y, dy)
            power = ls.power(freq)

            fap_levels = [0.001, 0.01, 0.1]
            fap_colors = ["red", "orange", "green"]
            fap_labels = ["0.1%", "1%", "10%"]
            fap_powers = []

            try:
                for fap in fap_levels:
                    fap_powers.append(ls.false_alarm_level(fap, method="baluev"))
                print(
                    "  FAP levels (power): "
                    + ", ".join(
                        f"{lab}={p:.3f}" for lab, p in zip(fap_labels, fap_powers)
                    )
                )
            except Exception as e:
                print(f"  Warning: could not compute FAP levels: {e}")
                fap_powers = []

            power_threshold = np.percentile(power, 85)
            max_power = np.max(power)
            if power_threshold > 0.8 * max_power:
                power_threshold = 0.5 * max_power

            min_period_separation = 0.01
            if f_max > f_min:
                min_freq_separation = int(
                    len(freq) * min_period_separation / np.log10(f_max / f_min)
                )
            else:
                min_freq_separation = 1

            try:
                peak_indices, peak_props = find_peaks(
                    power,
                    height=float(power_threshold),
                    distance=max(1, min_freq_separation),
                    prominence=float(power_threshold) * 0.1,
                )
                if len(peak_indices) == 0:
                    lower_threshold = np.percentile(power, 70)
                    peak_indices, peak_props = find_peaks(
                        power,
                        height=float(lower_threshold),
                        distance=max(1, min_freq_separation),
                        prominence=float(lower_threshold) * 0.05,
                    )
            except Exception as e:
                print(f"  Peak detection error: {e}")
                peak_indices = (
                    np.array([np.argmax(power)]) if len(power) > 0 else np.array([])
                )

            if len(peak_indices) > 0:
                peak_periods = 1.0 / freq[peak_indices]
                peak_powers_raw = power[peak_indices]
                order = np.argsort(peak_powers_raw)[::-1]
                peak_periods = peak_periods[order][:3]
                peak_powers_sorted = peak_powers_raw[order][:3]

                peak_faps = []
                peak_significance = []
                for pwr in peak_powers_sorted:
                    try:
                        pfap = ls.false_alarm_probability(pwr, method="baluev")
                        peak_faps.append(pfap)
                        if len(fap_powers) >= 3:
                            if pwr >= fap_powers[0]:
                                sig = "highly significant"
                            elif pwr >= fap_powers[1]:
                                sig = "significant"
                            elif pwr >= fap_powers[2]:
                                sig = "marginal"
                            else:
                                sig = "not significant"
                        else:
                            sig = "unknown"
                        peak_significance.append(sig)
                    except Exception:
                        peak_faps.append(np.nan)
                        peak_significance.append("unknown")

                print("  Detected peaks (combined non-ESPRESSO RV):")
                ord_names = ["1st", "2nd", "3rd"]
                for j, (P, Pow, FAP, sig) in enumerate(
                    zip(peak_periods, peak_powers_sorted, peak_faps, peak_significance)
                ):
                    if not np.isnan(FAP):
                        print(
                            f"    {ord_names[j]} peak: {P:.2f} d, "
                            f"Power={Pow:.4f}, FAP={FAP:.2e} ({sig})"
                        )
                    else:
                        print(
                            f"    {ord_names[j]} peak: {P:.2f} d, Power={Pow:.4f}"
                        )
            else:
                peak_periods = []
                peak_powers_sorted = []
                peak_faps = []
                peak_significance = []
                print("  No significant peaks found in combined non-ESPRESSO RVs.")

            periods = 1.0 / freq
            # Spectral window: Fourier transform of the sampling pattern
            power_window = spectral_window_power(t, freq)
            power_window_scaled = power_window / np.max(power_window) * np.max(power)
            ax_ls.semilogx(
                periods, -power_window_scaled,
                color="red", linestyle="-", alpha=0.8, linewidth=1.5,
                label="Window",
            )
            ax_ls.semilogx(periods, power, "k-", linewidth=1)

            for fap_power, col, lab in zip(fap_powers, fap_colors, fap_labels):
                ax_ls.axhline(
                    fap_power,
                    color=col,
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1.5,
                    label=f"{lab} FAP",
                )

            peak_colors = ["purple", "blue", "cyan"]
            peak_styles = ["-", "--", ":"]
            for j, (P, Pow, FAP, sig) in enumerate(
                zip(peak_periods, peak_powers_sorted, peak_faps, peak_significance)
            ):
                if j >= len(peak_colors):
                    break
                if not np.isnan(FAP):
                    label = (
                        f"{['1st','2nd','3rd'][j]} peak: {P:.1f}d (FAP={FAP:.1e})"
                    )
                else:
                    label = f"{['1st','2nd','3rd'][j]} peak: {P:.1f}d"
                ax_ls.axvline(
                    P,
                    color=peak_colors[j],
                    linestyle=peak_styles[j],
                    linewidth=2,
                    alpha=0.8,
                    label=label,
                )

            reference_periods = [1, 7, 14, 28, 100, 365]
            for refP in reference_periods:
                if min_period <= refP <= max_period:
                    ax_ls.axvline(
                        refP, color="blue", linestyle=":", alpha=0.25, linewidth=1
                    )
                    ax_ls.text(
                        refP,
                        ax_ls.get_ylim()[1] * 0.85,
                        f"{refP}d",
                        rotation=90,
                        ha="right",
                        va="top",
                        fontsize=SMALL,
                        alpha=0.6,
                        color="blue",
                    )

            ax_ls.set_xlabel("Period [days]")
            ax_ls.set_ylabel("LS Power")
            ax_ls.set_title("HD102365 - Combined RV Lomb–Scargle (Non-ESPRESSO)")
            ax_ls.grid(True, alpha=0.3)
            ax_ls.legend(loc="upper right")

            if len(peak_periods) > 0:
                lines = []
                for j, (P, Pow, FAP) in enumerate(
                    zip(peak_periods, peak_powers_sorted, peak_faps)
                ):
                    if not np.isnan(FAP):
                        lines.append(f"Peak {j+1}: {P:.1f} d (FAP={FAP:.1e})")
                    else:
                        lines.append(f"Peak {j+1}: {P:.1f} d (P={Pow:.3f})")
                peak_text = "\n".join(lines)
            else:
                peak_text = "No significant peaks"

            stats_text = f"N={len(t)} points\nSpan={span:.1f} d\n" + peak_text
            ax_ls.text(
                0.02,
                0.75,
                stats_text,
                transform=ax_ls.transAxes,
                fontsize=SMALL,
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="gray",
                    linewidth=0.5,
                ),
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(
        "HD102365 - Combined RVs (All Non-ESPRESSO Instruments)")

    outpath = os.path.join(fig_dir, "HD102365_all_non_ESPRESSO_RV_combined_LS.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    _save_interactive_html(fig, outpath)
    plt.close(fig)
    print(f"  Saved combined non-ESPRESSO RV figure: {outpath}")

# ---------------------------------------------------------------------
# NEW: same idea as above but INCLUDING ESPRESSO
# ---------------------------------------------------------------------
def plot_combined_rv_with_periodogram(df_all, fig_dir):
    """
    Plot a combined RV time series (ALL instruments, including ESPRESSO)
    and a single Lomb-Scargle periodogram using all RV data.
    """

    if "RV [m/s]" not in df_all.columns:
        print("No 'RV [m/s]' column found, skipping combined RV plot.")
        return

    rv_df = df_all[~df_all["RV [m/s]"].isna()].copy()
    if rv_df.empty:
        print("No RV data available for combined RV plot.")
        return

    print("\nCreating combined RV time series and periodogram (ALL instruments)...")
    print(
        f"  Total RV points: {len(rv_df)} from "
        f"{rv_df['Instrument'].nunique()} instruments"
    )
    print(f"  Instruments: {sorted(rv_df['Instrument'].dropna().unique())}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    ax_ts, ax_ls = axes

    # ---- LEFT PANEL: TIME SERIES ----
    instruments = sorted(rv_df["Instrument"].dropna().unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(instruments))))
    color_map = dict(zip(instruments, colors))

    for inst in instruments:
        inst_data = rv_df[rv_df["Instrument"] == inst].copy()
        median_inst = inst_data["RV [m/s]"].median()
        y_centered = inst_data["RV [m/s]"] - median_inst

        if "RV Err. [m/s]" in inst_data.columns:
            yerr = inst_data["RV Err. [m/s]"].values
        else:
            yerr = None

        ax_ts.errorbar(
            inst_data["Time [BJD]"],
            y_centered,
            yerr=yerr,
            fmt="o",
            color=color_map[inst],
            label=inst,
            alpha=0.7,
            markersize=5,
            capsize=2,
        )

    ax_ts.set_xlabel("Time [BJD]")
    ax_ts.set_ylabel("RV [m/s] (per-instrument median subtracted)")
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend(loc="best", ncol=2)
    ax_ts.set_title("HD102365 - Combined RVs (All Instruments)")

    # ---- RIGHT PANEL: LOMB-SCARGLE (ALL data together) ----
    time_all = rv_df["Time [BJD]"].values
    rv_all = rv_df["RV [m/s]"].values

    if "RV Err. [m/s]" in rv_df.columns:
        err_all = rv_df["RV Err. [m/s]"].values
    else:
        err_all = np.ones_like(rv_all)

    mask = ~(np.isnan(time_all) | np.isnan(rv_all) | np.isnan(err_all))
    t = time_all[mask]
    y = rv_all[mask]
    dy = err_all[mask]

    if len(t) < 5:
        ax_ls.text(
            0.5,
            0.5,
            f"Insufficient data for periodogram\n(N={len(t)} points)",
            ha="center",
            va="center",
            transform=ax_ls.transAxes,
        )
        print(f"  Not enough points for LS periodogram (N={len(t)})")
    else:
        bad = dy <= 0
        if bad.any():
            repl = np.median(dy[~bad]) if (~bad).any() else 1.0
            dy[bad] = repl

        span = t.max() - t.min()
        print(f"  Time span for LS: {span:.1f} days")

        if span <= 1.0:
            ax_ls.text(
                0.5,
                0.5,
                f"Insufficient time span for periodogram\n(span={span:.1f} days)",
                ha="center",
                va="center",
                transform=ax_ls.transAxes,
            )
        else:
            min_period = 1.1
            max_period = max(2.0, 0.8 * span)
            f_min = 1.0 / max_period
            f_max = 1.0 / min_period

            N = 20000
            freq = np.linspace(f_min, f_max, N)

            ls = LombScargle(t, y, dy)
            power = ls.power(freq)

            fap_levels = [0.001, 0.01, 0.1]
            fap_colors = ["red", "orange", "green"]
            fap_labels = ["0.1%", "1%", "10%"]
            fap_powers = []

            try:
                for fap in fap_levels:
                    fap_powers.append(ls.false_alarm_level(fap, method="baluev"))
                print(
                    "  FAP levels (power): "
                    + ", ".join(
                        f"{lab}={p:.3f}" for lab, p in zip(fap_labels, fap_powers)
                    )
                )
            except Exception as e:
                print(f"  Warning: could not compute FAP levels: {e}")
                fap_powers = []

            power_threshold = np.percentile(power, 85)
            max_power = np.max(power)
            if power_threshold > 0.8 * max_power:
                power_threshold = 0.5 * max_power

            min_period_separation = 0.01
            if f_max > f_min:
                min_freq_separation = int(
                    len(freq) * min_period_separation / np.log10(f_max / f_min)
                )
            else:
                min_freq_separation = 1

            try:
                peak_indices, peak_props = find_peaks(
                    power,
                    height=float(power_threshold),
                    distance=max(1, min_freq_separation),
                    prominence=float(power_threshold) * 0.1,
                )
                if len(peak_indices) == 0:
                    lower_threshold = np.percentile(power, 70)
                    peak_indices, peak_props = find_peaks(
                        power,
                        height=float(lower_threshold),
                        distance=max(1, min_freq_separation),
                        prominence=float(lower_threshold) * 0.05,
                    )
            except Exception as e:
                print(f"  Peak detection error: {e}")
                peak_indices = (
                    np.array([np.argmax(power)]) if len(power) > 0 else np.array([])
                )

            if len(peak_indices) > 0:
                peak_periods = 1.0 / freq[peak_indices]
                peak_powers_raw = power[peak_indices]
                order = np.argsort(peak_powers_raw)[::-1]
                peak_periods = peak_periods[order][:3]
                peak_powers_sorted = peak_powers_raw[order][:3]

                peak_faps = []
                peak_significance = []
                for pwr in peak_powers_sorted:
                    try:
                        pfap = ls.false_alarm_probability(pwr, method="baluev")
                        peak_faps.append(pfap)
                        if len(fap_powers) >= 3:
                            if pwr >= fap_powers[0]:
                                sig = "highly significant"
                            elif pwr >= fap_powers[1]:
                                sig = "significant"
                            elif pwr >= fap_powers[2]:
                                sig = "marginal"
                            else:
                                sig = "not significant"
                        else:
                            sig = "unknown"
                        peak_significance.append(sig)
                    except Exception:
                        peak_faps.append(np.nan)
                        peak_significance.append("unknown")

                print("  Detected peaks (combined ALL-instrument RV):")
                ord_names = ["1st", "2nd", "3rd"]
                for j, (P, Pow, FAP, sig) in enumerate(
                    zip(peak_periods, peak_powers_sorted, peak_faps, peak_significance)
                ):
                    if not np.isnan(FAP):
                        print(
                            f"    {ord_names[j]} peak: {P:.2f} d, "
                            f"Power={Pow:.4f}, FAP={FAP:.2e} ({sig})"
                        )
                    else:
                        print(
                            f"    {ord_names[j]} peak: {P:.2f} d, Power={Pow:.4f}"
                        )
            else:
                peak_periods = []
                peak_powers_sorted = []
                peak_faps = []
                peak_significance = []
                print("  No significant peaks found in combined ALL-instrument RVs.")

            periods = 1.0 / freq
            # Spectral window: Fourier transform of the sampling pattern
            power_window = spectral_window_power(t, freq)
            power_window_scaled = power_window / np.max(power_window) * np.max(power)
            ax_ls.semilogx(
                periods, -power_window_scaled,
                color="red", linestyle="-", alpha=0.8, linewidth=1.5,
                label="Window",
            )
            ax_ls.semilogx(periods, power, "k-", linewidth=1)

            for fap_power, col, lab in zip(fap_powers, fap_colors, fap_labels):
                ax_ls.axhline(
                    fap_power,
                    color=col,
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1.5,
                    label=f"{lab} FAP",
                )

            peak_colors = ["purple", "blue", "cyan"]
            peak_styles = ["-", "--", ":"]
            for j, (P, Pow, FAP, sig) in enumerate(
                zip(peak_periods, peak_powers_sorted, peak_faps, peak_significance)
            ):
                if j >= len(peak_colors):
                    break
                if not np.isnan(FAP):
                    label = (
                        f"{['1st','2nd','3rd'][j]} peak: {P:.1f}d (FAP={FAP:.1e})"
                    )
                else:
                    label = f"{['1st','2nd','3rd'][j]} peak: {P:.1f}d"
                ax_ls.axvline(
                    P,
                    color=peak_colors[j],
                    linestyle=peak_styles[j],
                    linewidth=2,
                    alpha=0.8,
                    label=label,
                )

            reference_periods = [1, 7, 14, 28, 100, 365]
            for refP in reference_periods:
                if min_period <= refP <= max_period:
                    ax_ls.axvline(
                        refP, color="blue", linestyle=":", alpha=0.25, linewidth=1
                    )
                    ax_ls.text(
                        refP,
                        ax_ls.get_ylim()[1] * 0.85,
                        f"{refP}d",
                        rotation=90,
                        ha="right",
                        va="top",
                        fontsize=SMALL,
                        alpha=0.6,
                        color="blue",
                    )

            ax_ls.set_xlabel("Period [days]")
            ax_ls.set_ylabel("LS Power")
            ax_ls.set_title("HD102365 - Combined RV Lomb–Scargle (All Instruments)")
            ax_ls.grid(True, alpha=0.3)
            ax_ls.legend(loc="upper right")

            if len(peak_periods) > 0:
                lines = []
                for j, (P, Pow, FAP) in enumerate(
                    zip(peak_periods, peak_powers_sorted, peak_faps)
                ):
                    if not np.isnan(FAP):
                        lines.append(f"Peak {j+1}: {P:.1f} d (FAP={FAP:.1e})")
                    else:
                        lines.append(f"Peak {j+1}: {P:.1f} d (P={Pow:.3f})")
                peak_text = "\n".join(lines)
            else:
                peak_text = "No significant peaks"

            stats_text = f"N={len(t)} points\nSpan={span:.1f} d\n" + peak_text
            ax_ls.text(
                0.02,
                0.75,
                stats_text,
                transform=ax_ls.transAxes,
                fontsize=SMALL,
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="gray",
                    linewidth=0.5,
                ),
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle("HD102365 - Combined RVs (All Instruments)")

    outpath = os.path.join(fig_dir, "HD102365_all_instruments_RV_combined_LS.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    _save_interactive_html(fig, outpath)
    plt.close(fig)
    print(f"  Saved combined ALL-instruments RV figure: {outpath}")

def main():
    """Main function"""
    
    data_dir = "/work2/lbuc/iara/GitHub/PyORBIT_iara/HD102365/processed_data"
    fig_dir = "/work2/lbuc/iara/GitHub/PyORBIT_iara/HD102365/figures"
    
    os.makedirs(fig_dir, exist_ok=True)
    
    # Load data and create plots
    df_all = load_dat_files(data_dir)
    
    print("\nCreating periodogram plots...")
    for instrument, inst_df in df_all.groupby("Instrument"):
        print(f"\n{instrument}:")
        plot_activity_with_periodograms(instrument, inst_df, fig_dir)
    
    # Non-ESPRESSO combined (old behaviour, but with honest name/file)
    plot_combined_rv_non_espresso_with_periodogram(df_all, fig_dir)
    # NEW: ALL instruments (incl. ESPRESSO)
    plot_combined_rv_with_periodogram(df_all, fig_dir)

    print("\nDone!")

if __name__ == "__main__":
    main()
