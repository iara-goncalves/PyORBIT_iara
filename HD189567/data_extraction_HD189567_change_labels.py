from glob import glob
import os
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

# Specify where all the data is
espresso_data_dir = "/work2/lbuc/iara/Data/Fits_ESPRESSO"
harps_data_dir = "/work2/lbuc/iara/Data/Fits_HARPS"

# BJD threshold for HARPS instrument correction
HARPS_CORRECTION_BJD = 2457176.5

def extract_espresso_fits_data(fits_file):
    """
    Extract the required ESO QC parameters from an ESPRESSO FITS file header
    """
    try:
        with fits.open(fits_file) as hdul:
            header = hdul[0].header
            
            # Extract the required parameters
            data = {
                'BJD': header.get('HIERARCH ESO QC BJD', np.nan),
                'RV': header.get('HIERARCH ESO QC CCF RV', np.nan),
                'RV_ERROR': header.get('HIERARCH ESO QC CCF RV ERROR', np.nan),
                'CCF_FWHM': header.get('HIERARCH ESO QC CCF FWHM', np.nan),
                'CCF_FWHM_ERROR': header.get('HIERARCH ESO QC CCF FWHM ERROR', np.nan),
                'CCF_CONTRAST': header.get('HIERARCH ESO QC CCF CONTRAST', np.nan),
                'CCF_CONTRAST_ERROR': header.get('HIERARCH ESO QC CCF CONTRAST ERROR', np.nan),
                'CCF_BIS': header.get('HIERARCH ESO QC CCF BIS SPAN', np.nan),
                'CCF_BIS_ERROR': header.get('HIERARCH ESO QC CCF BIS SPAN ERROR', np.nan),
                'instrument': 'ESPRESSO',
                'jitter': 0,  # ESPRESSO jitter flag = 0
                'instrument_flag': 0  # ESPRESSO = 0
            }
            
            # Convert km/s to m/s immediately after extraction
            km_to_m_columns = ['RV', 'RV_ERROR', 'CCF_FWHM', 'CCF_FWHM_ERROR', 'CCF_BIS', 'CCF_BIS_ERROR']
            for col in km_to_m_columns:
                if not np.isnan(data[col]):
                    data[col] = data[col] * 1000  # Convert km/s to m/s
            
        return data
    
    except Exception as e:
        print(f"Error reading ESPRESSO {fits_file}: {e}")
        return None

def extract_harps_fits_data(fits_file, error_ratios):
    """
    Extract the required DRS parameters from a HARPS FITS file header
    """
    try:
        with fits.open(fits_file) as hdul:
            header = hdul[0].header
            
            # Get BJD first to determine instrument flag
            bjd = header.get('HIERARCH ESO DRS BJD', np.nan)
            
            # Extract the required parameters
            data = {
                'BJD': bjd,
                'RV': header.get('HIERARCH ESO DRS CCF RV', np.nan),
                'RV_ERROR': header.get('HIERARCH ESO DRS CCF NOISE', np.nan),  # CCF NOISE is RV error
                'CCF_FWHM': header.get('HIERARCH ESO DRS CCF FWHM', np.nan),
                'CCF_CONTRAST': header.get('HIERARCH ESO DRS CCF CONTRAST', np.nan),
                'CCF_BIS': header.get('HIERARCH ESO DRS BIS SPAN', np.nan),
                'instrument': 'HARPS',
                'jitter': 1,  # HARPS jitter flag = 1
                'instrument_flag': 1 if bjd < HARPS_CORRECTION_BJD else 2  # HARPS pre-correction = 1, post-correction = 2
            }
            
            # Convert km/s to m/s for RV, RV_ERROR, CCF_FWHM, and CCF_BIS
            km_to_m_columns = ['RV', 'RV_ERROR', 'CCF_FWHM', 'CCF_BIS']
            for col in km_to_m_columns:
                if not np.isnan(data[col]):
                    data[col] = data[col] * 1000  # Convert km/s to m/s
            
            # Estimate errors for FWHM, CONTRAST, and BIS using ratios from ESPRESSO
            if not np.isnan(data['RV_ERROR']):
                data['CCF_FWHM_ERROR'] = data['RV_ERROR'] * error_ratios['fwhm_ratio']
                data['CCF_CONTRAST_ERROR'] = data['RV_ERROR'] * error_ratios['contrast_ratio']
                data['CCF_BIS_ERROR'] = data['RV_ERROR'] * error_ratios['bis_ratio']
            else:
                data['CCF_FWHM_ERROR'] = np.nan
                data['CCF_CONTRAST_ERROR'] = np.nan
                data['CCF_BIS_ERROR'] = np.nan
            
        return data
    
    except Exception as e:
        print(f"Error reading HARPS {fits_file}: {e}")
        return None

def calculate_error_ratios(espresso_files):
    """
    Calculate error ratios from the first ESPRESSO file to estimate HARPS errors
    """
    if not espresso_files:
        print("No ESPRESSO files found for error ratio calculation")
        return {'fwhm_ratio': 1.0, 'contrast_ratio': 1.0, 'bis_ratio': 1.0}
    
    try:
        with fits.open(espresso_files[0]) as hdul:
            header = hdul[0].header
            
            rv_error = header.get('HIERARCH ESO QC CCF RV ERROR', np.nan) * 1000  # Convert to m/s
            fwhm_error = header.get('HIERARCH ESO QC CCF FWHM ERROR', np.nan) * 1000  # Convert to m/s
            contrast_error = header.get('HIERARCH ESO QC CCF CONTRAST ERROR', np.nan)
            bis_error = header.get('HIERARCH ESO QC CCF BIS SPAN ERROR', np.nan) * 1000  # Convert to m/s
            
            # Calculate ratios (error_indicator / rv_error)
            ratios = {
                'fwhm_ratio': fwhm_error / rv_error if not np.isnan(fwhm_error) and not np.isnan(rv_error) and rv_error != 0 else 1.0,
                'contrast_ratio': contrast_error / rv_error if not np.isnan(contrast_error) and not np.isnan(rv_error) and rv_error != 0 else 1.0,
                'bis_ratio': bis_error / rv_error if not np.isnan(bis_error) and not np.isnan(rv_error) and rv_error != 0 else 1.0
            }
            
            print(f"Error ratios calculated from {os.path.basename(espresso_files[0])}:")
            print(f"  FWHM/RV error ratio: {ratios['fwhm_ratio']:.3f}")
            print(f"  CONTRAST/RV error ratio: {ratios['contrast_ratio']:.3f}")
            print(f"  BIS/RV error ratio: {ratios['bis_ratio']:.3f}")
            
            return ratios
    
    except Exception as e:
        print(f"Error calculating ratios: {e}")
        return {'fwhm_ratio': 1.0, 'contrast_ratio': 1.0, 'bis_ratio': 1.0}

def identify_outliers(data, sigma_threshold=3):
    """
    Identify outliers using sigma clipping
    """
    mean = np.nanmean(data)
    std = np.nanstd(data)
    outliers = np.abs(data - mean) > sigma_threshold * std
    return outliers

def identify_outliers_by_dates(df, sigma_threshold=3):
    """
    Identify outlier dates using the same method as your ESSP code:
    - Use median instead of mean
    - Use standard deviation (ddof=1)
    - Accumulate outliers across ALL activity indicators
    - Return dates to exclude from ALL indicators
    """
    # Columns to test for outliers (matching your approach)
    cols_to_clip = ['RV', 'CCF_FWHM', 'CCF_CONTRAST', 'CCF_BIS']
    
    # Initialize outlier mask - use pandas Series to maintain index alignment
    outlier_mask = pd.Series(False, index=df.index)
    
    print(f"\n{'='*60}")
    print(f"OUTLIER DETECTION (sigma_threshold={sigma_threshold})")
    print(f"{'='*60}")
    
    outliers_by_column = {}  # Track outliers found in each column
    
    # Check each column and accumulate outliers
    for col in cols_to_clip:
        if col not in df.columns:
            print(f"  ❌ Column {col} not found, skipping")
            continue
            
        # Get non-null values for statistics calculation
        valid_data = df[col].dropna()
        if len(valid_data) == 0:
            print(f"  ❌ Column {col} has no valid data, skipping")
            continue
            
        # Use median and std (matching your code exactly)
        median = valid_data.median()
        std = valid_data.std(ddof=1)  # ddof=1 matches your code
        
        if std == 0 or np.isnan(std):
            print(f"  ❌ Column {col} has zero or NaN std, skipping")
            continue
            
        # Find outliers for this column - use the full df[col] to maintain index alignment
        col_outliers = (np.abs(df[col] - median) > (sigma_threshold * std))
        
        # Get outlier information for this column
        col_outlier_indices = col_outliers.fillna(False)
        outliers_in_col = col_outlier_indices.sum()
        
        if outliers_in_col > 0:
            outlier_bjds = df.loc[col_outlier_indices, 'BJD'].values
            outlier_values = df.loc[col_outlier_indices, col].values
            outlier_instruments = df.loc[col_outlier_indices, 'instrument'].values
            
            outliers_by_column[col] = {
                'count': outliers_in_col,
                'bjds': outlier_bjds,
                'values': outlier_values,
                'instruments': outlier_instruments
            }
            
            print(f"  📊 Column {col}:")
            print(f"      Median: {median:.3f}, Std: {std:.3f}")
            print(f"      Threshold: ±{sigma_threshold * std:.3f}")
            print(f"      Outliers found: {outliers_in_col}")
            
            # Show details of each outlier
            for j, (bjd, val, inst) in enumerate(zip(outlier_bjds, outlier_values, outlier_instruments)):
                deviation = abs(val - median) / std
                print(f"        {j+1}. BJD {bjd:.6f} ({inst}): {val:.3f} ({deviation:.1f}σ)")
        else:
            print(f"  ✅ Column {col}: median={median:.3f}, std={std:.3f} - No outliers found")
        
        # Accumulate outliers (OR operation, matching your |= operator)
        outlier_mask = outlier_mask | col_outliers.fillna(False)
    
    # Get the BJD dates of outliers
    outlier_indices = outlier_mask[outlier_mask].index
    outlier_dates = set(df.loc[outlier_indices, 'BJD'].values)
    
    print(f"\n{'='*60}")
    print(f"OUTLIER SUMMARY")
    print(f"{'='*60}")
    
    if len(outlier_dates) > 0:
        print(f"🚨 Total unique outlier dates identified: {len(outlier_dates)}")
        print(f"📅 Outlier BJD dates: {sorted(list(outlier_dates))}")
        
        # Show which instruments contributed outliers
        outlier_instruments = df.loc[outlier_indices, 'instrument'].value_counts()
        print(f"📡 Outliers by instrument:")
        for inst, count in outlier_instruments.items():
            print(f"    {inst}: {count} observations")
        
        # Show breakdown by column
        print(f"📈 Outliers by activity indicator:")
        for col, info in outliers_by_column.items():
            print(f"    {col}: {info['count']} outliers")
        
        # Show which dates appear as outliers in multiple indicators
        bjd_counts = {}
        for col, info in outliers_by_column.items():
            for bjd in info['bjds']:
                bjd_counts[bjd] = bjd_counts.get(bjd, 0) + 1
        
        multi_indicator_outliers = {bjd: count for bjd, count in bjd_counts.items() if count > 1}
        if multi_indicator_outliers:
            print(f"⚠️  Dates flagged as outliers in multiple indicators:")
            for bjd, count in sorted(multi_indicator_outliers.items()):
                inst = df[df['BJD'] == bjd]['instrument'].iloc[0]
                print(f"    BJD {bjd:.6f} ({inst}): flagged in {count} indicators")
    else:
        print(f"✅ No outliers detected in any activity indicator!")
        print(f"📊 All {len(df)} observations passed the {sigma_threshold}σ threshold")
    
    print(f"{'='*60}\n")
    
    return outlier_dates



def save_dat_files(df, output_dir, instrument_name, exclude_outliers=True):
    """
    Save individual .dat files for each activity indicator for a specific instrument
    Format: BJD, value, error, jitter, instrument_flag
    EXCLUDES OUTLIERS from the saved files
    """
    indicators = {
        'RV': {
            'columns': ['BJD', 'RV', 'RV_ERROR', 'jitter', 'instrument_flag']
        },
        'CCF_FWHM': {
            'columns': ['BJD', 'CCF_FWHM', 'CCF_FWHM_ERROR', 'jitter', 'instrument_flag']
        },
        'CCF_CONTRAST': {
            'columns': ['BJD', 'CCF_CONTRAST', 'CCF_CONTRAST_ERROR', 'jitter', 'instrument_flag']
        },
        'CCF_BIS': {
            'columns': ['BJD', 'CCF_BIS', 'CCF_BIS_ERROR', 'jitter', 'instrument_flag']
        }
    }
    
    sigma_threshold = 3
    
    # Get instrument-specific data
    instrument_df = df[df['instrument'] == instrument_name].copy()
    
    if exclude_outliers:
        # Identify outlier dates using your method
        outlier_dates = identify_outliers_by_dates(instrument_df, sigma_threshold)
        
        # Filter out outlier dates from the entire instrument dataset
        instrument_df_clean = instrument_df[~instrument_df['BJD'].isin(outlier_dates)].copy()
        
        print(f"\n{instrument_name} outlier removal (EXCLUDING outliers from .dat files):")
        print(f"  - Total observations before filtering: {len(instrument_df)}")
        print(f"  - Outlier dates identified: {len(outlier_dates)}")
        print(f"  - Total observations after filtering: {len(instrument_df_clean)}")
        if len(outlier_dates) > 0:
            print(f"  - Outlier BJD dates: {sorted(list(outlier_dates))}")
    else:
        instrument_df_clean = instrument_df.copy()
        print(f"\n{instrument_name} - INCLUDING all data (no outlier removal)")
        print(f"  - Total observations: {len(instrument_df_clean)}")
    
    for indicator, info in indicators.items():
        # Create subset DataFrame for this indicator
        subset_df = instrument_df_clean[info['columns']].copy()
        
        # Remove rows with NaN values in the main parameter
        main_param = info['columns'][1]  # Second column is the main parameter
        subset_df = subset_df.dropna(subset=[info['columns'][0], main_param, info['columns'][2]])
        
        if len(subset_df) > 0:
            # Save as .dat file
            output_file = os.path.join(output_dir, f"{indicator}.dat")
            
            # Save with tab separation, no header
            save_data = subset_df.values
            np.savetxt(output_file, save_data, 
                      delimiter='\t', 
                      fmt=['%.6f', '%.6f', '%.6f', '%d', '%d'])
            
            status = "INLIERS ONLY" if exclude_outliers else "ALL DATA"
            print(f"Saved {instrument_name} {indicator}.dat with {len(subset_df)} records ({status})")
            
            # Print instrument flag distribution for HARPS
            if instrument_name == 'HARPS':
                flag_counts = subset_df['instrument_flag'].value_counts().sort_index()
                print(f"  - Instrument flags: {dict(flag_counts)}")
                pre_correction = len(subset_df[subset_df['instrument_flag'] == 1])
                post_correction = len(subset_df[subset_df['instrument_flag'] == 2])
                print(f"  - Pre-correction (flag=1): {pre_correction}, Post-correction (flag=2): {post_correction}")
        else:
            print(f"Warning: No valid data for {instrument_name} {indicator}")


def save_combined_dat_files(df, output_dir, exclude_outliers=True):
    """
    Save combined .dat files with both ESPRESSO and HARPS data
    Format: BJD, value, error, jitter, instrument_flag
    EXCLUDES OUTLIERS from the saved files
    """
    indicators = {
        'RV': {
            'columns': ['BJD', 'RV', 'RV_ERROR', 'jitter', 'instrument_flag']
        },
        'CCF_FWHM': {
            'columns': ['BJD', 'CCF_FWHM', 'CCF_FWHM_ERROR', 'jitter', 'instrument_flag']
        },
        'CCF_CONTRAST': {
            'columns': ['BJD', 'CCF_CONTRAST', 'CCF_CONTRAST_ERROR', 'jitter', 'instrument_flag']
        },
        'CCF_BIS': {
            'columns': ['BJD', 'CCF_BIS', 'CCF_BIS_ERROR', 'jitter', 'instrument_flag']
        }
    }
    
    sigma_threshold = 3
    
    if exclude_outliers:
        # Identify outlier dates using your method on the entire dataset
        outlier_dates = identify_outliers_by_dates(df, sigma_threshold)
        
        # Filter out outlier dates from the entire dataset
        df_clean = df[~df['BJD'].isin(outlier_dates)].copy()
        
        print(f"\nCombined data outlier removal (EXCLUDING outliers from .dat files):")
        print(f"  - Total observations before filtering: {len(df)}")
        print(f"  - Outlier dates identified: {len(outlier_dates)}")
        print(f"  - Total observations after filtering: {len(df_clean)}")
        if len(outlier_dates) > 0:
            print(f"  - Outlier BJD dates: {sorted(list(outlier_dates))}")
    else:
        df_clean = df.copy()
        print(f"\nCombined data - INCLUDING all data (no outlier removal)")
        print(f"  - Total observations: {len(df_clean)}")
    
    for indicator, info in indicators.items():
        # Create combined DataFrame for all instruments
        combined_df = df_clean[info['columns']].copy()
        
        # Remove rows with NaN values in the main parameter
        main_param = info['columns'][1]  # Second column is the main parameter
        combined_df = combined_df.dropna(subset=[info['columns'][0], main_param, info['columns'][2]])
        
        if len(combined_df) > 0:
            # Sort by BJD
            combined_df = combined_df.sort_values('BJD').reset_index(drop=True)
            
            # Save as .dat file
            output_file = os.path.join(output_dir, f"{indicator}_combined.dat")
            
            # Save with tab separation, no header
            save_data = combined_df.values
            np.savetxt(output_file, save_data, 
                      delimiter='\t', 
                      fmt=['%.6f', '%.6f', '%.6f', '%d', '%d'])
            
            status = "INLIERS ONLY" if exclude_outliers else "ALL DATA"
            print(f"Saved combined {indicator}_combined.dat with {len(combined_df)} records ({status})")
            
            # Print instrument flag distribution
            flag_counts = combined_df['instrument_flag'].value_counts().sort_index()
            flag_names = {0: 'ESPRESSO', 1: 'HARPS pre-correction', 2: 'HARPS post-correction'}
            print(f"  - Instrument flag distribution:")
            for flag, count in flag_counts.items():
                print(f"    Flag {flag} ({flag_names.get(flag, 'Unknown')}): {count} records")
        else:
            print(f"Warning: No valid data for combined {indicator}")

def load_dat_files(espresso_dir, harps_dir):
    """
    Load data from .dat files in both directories
    """
    indicators = ['RV', 'CCF_FWHM', 'CCF_CONTRAST', 'CCF_BIS']
    combined_data = []
    
    for indicator in indicators:
        # Load ESPRESSO data
        espresso_file = os.path.join(espresso_dir, f"{indicator}.dat")
        if os.path.exists(espresso_file):
            try:
                esp_data = np.loadtxt(espresso_file)
                if esp_data.ndim == 1:
                    esp_data = esp_data.reshape(1, -1)
                for row in esp_data:
                    combined_data.append({
                        'BJD': row[0],
                        indicator: row[1],
                        f"{indicator}_ERROR": row[2],
                        'jitter': int(row[3]),
                        'instrument_flag': int(row[4]),
                        'instrument': 'ESPRESSO'
                    })
                print(f"Loaded ESPRESSO {indicator}: {len(esp_data)} points")
            except Exception as e:
                print(f"Error loading ESPRESSO {indicator}: {e}")
        
        # Load HARPS data
        harps_file = os.path.join(harps_dir, f"{indicator}.dat")
        if os.path.exists(harps_file):
            try:
                harps_data = np.loadtxt(harps_file)
                if harps_data.ndim == 1:
                    harps_data = harps_data.reshape(1, -1)
                for row in harps_data:
                    combined_data.append({
                        'BJD': row[0],
                        indicator: row[1],
                        f"{indicator}_ERROR": row[2],
                        'jitter': int(row[3]),
                        'instrument_flag': int(row[4]),
                        'instrument': 'HARPS'
                    })
                print(f"Loaded HARPS {indicator}: {len(harps_data)} points")
            except Exception as e:
                print(f"Error loading HARPS {indicator}: {e}")
    
    if combined_data:
        df = pd.DataFrame(combined_data)
        # Fill NaN for missing indicators
        all_indicators = ['RV', 'CCF_FWHM', 'CCF_CONTRAST', 'CCF_BIS']
        all_error_cols = [f"{ind}_ERROR" for ind in all_indicators]
        for col in all_indicators + all_error_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        # Group by BJD and instrument, then merge
        df_final = df.groupby(['BJD', 'instrument']).first().reset_index()
        df_final = df_final.sort_values('BJD').reset_index(drop=True)
        
        return df_final
    else:
        return pd.DataFrame()

def plot_activity_with_periodograms_combined(df, output_dir, sigma_threshold=3):
    """
    Plot combined ESPRESSO and HARPS activity indicators with Lomb-Scargle periodograms using only inliers.
    Left column: Activity data, Right column: Periodograms
    Shows instrument correction boundary for HARPS data
    """
    # Remove outliers for periodogram analysis
    df_clean = df.copy()
    
    # Identify and mark outliers for each indicator
    indicators = ['RV', 'CCF_FWHM', 'CCF_CONTRAST', 'CCF_BIS']
    
    for indicator in indicators:
        valid_mask = ~df_clean[indicator].isna()
        if valid_mask.sum() > 0:
            outliers = identify_outliers(df_clean.loc[valid_mask, indicator].values, sigma_threshold)
            df_clean.loc[valid_mask, f'{indicator}_outlier'] = outliers
        else:
            df_clean[f'{indicator}_outlier'] = False
    
    # Create the plot
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle('HD 189567 - Combined ESPRESSO & HARPS Activity Indicators with Periodograms\n(HARPS correction at BJD 2457176.5)', fontsize=16, fontweight='bold')
    
    colors = {'ESPRESSO': 'red', 'HARPS': 'blue'}
    
    for i, indicator in enumerate(indicators):
        # Left column: Activity data
        ax_data = axes[i, 0]
        
        for instrument in ['ESPRESSO', 'HARPS']:
            mask = (df['instrument'] == instrument) & (~df[indicator].isna())
            if mask.sum() > 0:
                # For HARPS, separate pre and post correction
                if instrument == 'HARPS':
                    # Pre-correction HARPS (flag = 1)
                    pre_mask = mask & (df['instrument_flag'] == 1)
                    if pre_mask.sum() > 0:
                        inlier_mask = pre_mask & (~df_clean[f'{indicator}_outlier'].fillna(False))
                        if inlier_mask.sum() > 0:
                            ax_data.errorbar(df.loc[inlier_mask, 'BJD'], 
                                           df.loc[inlier_mask, indicator],
                                           yerr=df.loc[inlier_mask, f'{indicator}_ERROR'],
                                           fmt='o', color='darkblue', alpha=0.7, 
                                           markersize=4, capsize=2, label='HARPS pre-correction')
                    
                    # Post-correction HARPS (flag = 2)
                    post_mask = mask & (df['instrument_flag'] == 2)
                    if post_mask.sum() > 0:
                        inlier_mask = post_mask & (~df_clean[f'{indicator}_outlier'].fillna(False))
                        if inlier_mask.sum() > 0:
                            ax_data.errorbar(df.loc[inlier_mask, 'BJD'], 
                                           df.loc[inlier_mask, indicator],
                                           yerr=df.loc[inlier_mask, f'{indicator}_ERROR'],
                                           fmt='s', color='lightblue', alpha=0.7, 
                                           markersize=4, capsize=2, label='HARPS post-correction')
                else:
                    # ESPRESSO data
                    inlier_mask = mask & (~df_clean[f'{indicator}_outlier'].fillna(False))
                    if inlier_mask.sum() > 0:
                        ax_data.errorbar(df.loc[inlier_mask, 'BJD'], 
                                       df.loc[inlier_mask, indicator],
                                       yerr=df.loc[inlier_mask, f'{indicator}_ERROR'],
                                       fmt='o', color=colors[instrument], alpha=0.7, 
                                       markersize=4, capsize=2, label=f'{instrument}')
                
                # Plot outliers
                outlier_mask = mask & (df_clean[f'{indicator}_outlier'].fillna(False))
                if outlier_mask.sum() > 0:
                    ax_data.scatter(df.loc[outlier_mask, 'BJD'], 
                                  df.loc[outlier_mask, indicator],
                                  marker='x', color=colors[instrument], alpha=0.5, 
                                  s=50, label=f'{instrument} (outliers)')
        
        # Add vertical line for HARPS correction
        ax_data.axvline(HARPS_CORRECTION_BJD, color='gray', linestyle='--', alpha=0.5, 
                       label='HARPS correction')
        
        ax_data.set_xlabel('BJD')
        ax_data.set_ylabel(f'{indicator} (m/s)' if indicator != 'CCF_CONTRAST' else f'{indicator}')
        ax_data.set_title(f'{indicator}')
        ax_data.legend()
        ax_data.grid(True, alpha=0.3)
        
        # Right column: Periodogram
        ax_period = axes[i, 1]
        
        # Combine inlier data from both instruments for periodogram
        combined_inliers = df_clean[(~df_clean[indicator].isna()) & 
                                   (~df_clean[f'{indicator}_outlier'].fillna(False))]
        
        if len(combined_inliers) > 3:
            try:
                bjd = combined_inliers['BJD'].values
                values = combined_inliers[indicator].values
                errors = combined_inliers[f'{indicator}_ERROR'].values
                
                # Create frequency array
                frequency = np.linspace(0.01, 2, 10000)  # 0.5 to 100 days
                
                # Calculate Lomb-Scargle periodogram
                ls = LombScargle(bjd, values, errors)
                power = ls.power(frequency)
                
                # Plot periodogram
                period = 1/frequency
                ax_period.semilogx(period, power, 'k-', alpha=0.8)
                
                # Find and mark significant peaks
                peaks, _ = find_peaks(power, height=0.1, distance=100)
                if len(peaks) > 0:
                    peak_periods = period[peaks]
                    peak_powers = power[peaks]
                    
                    # Sort by power and take top 3
                    sorted_indices = np.argsort(peak_powers)[::-1][:3]
                    top_peaks = peaks[sorted_indices]
                    
                    for peak_idx in top_peaks:
                        ax_period.axvline(period[peak_idx], color='red', linestyle='--', alpha=0.7)
                        ax_period.text(period[peak_idx], power[peak_idx], 
                                     f'{period[peak_idx]:.1f}d', 
                                     rotation=90, ha='right', va='bottom', fontsize=8)
                
                ax_period.set_xlabel('Period (days)')
                ax_period.set_ylabel('Power')
                ax_period.set_title(f'{indicator} - Periodogram')
                ax_period.grid(True, alpha=0.3)
                ax_period.set_xlim(0.5, 100)
                
            except Exception as e:
                ax_period.text(0.5, 0.5, f'Error: {str(e)}', 
                             transform=ax_period.transAxes, ha='center', va='center')
                ax_period.set_title(f'{indicator} - Periodogram (Error)')
        else:
            ax_period.text(0.5, 0.5, 'Insufficient data for periodogram', 
                         transform=ax_period.transAxes, ha='center', va='center')
            ax_period.set_title(f'{indicator} - Periodogram (No data)')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(output_dir, 'HD189567_combined_activity_indicators_with_periodograms.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved: {plot_filename}")
    
    plt.show()

def main():
    print("Starting HD 189567 data extraction and analysis...")
    print("=" * 60)
    print(f"HARPS instrument correction threshold: BJD {HARPS_CORRECTION_BJD}")
    
    # Find all FITS files
    espresso_fits_files = sorted(glob(os.path.join(espresso_data_dir, "*.fits")))
    harps_fits_files = sorted(glob(os.path.join(harps_data_dir, "*bis_G2_A.fits")))
    
    print(f"Found {len(espresso_fits_files)} ESPRESSO FITS files")
    print(f"Found {len(harps_fits_files)} HARPS FITS files")
    
    if len(espresso_fits_files) == 0 and len(harps_fits_files) == 0:
        print("No FITS files found! Please check the directory paths.")
        return
    
    # Calculate error ratios from ESPRESSO data
    error_ratios = calculate_error_ratios(espresso_fits_files)
    
    # Extract data from all files
    all_data = []
    
    # Process ESPRESSO files
    print(f"\nProcessing {len(espresso_fits_files)} ESPRESSO files...")
    for i, fits_file in enumerate(espresso_fits_files):
        if i % 10 == 0:
            print(f"  Processing ESPRESSO file {i+1}/{len(espresso_fits_files)}")
        
        data = extract_espresso_fits_data(fits_file)
        if data:
            all_data.append(data)
    
    # Process HARPS files
    print(f"\nProcessing {len(harps_fits_files)} HARPS files...")
    harps_pre_correction = 0
    harps_post_correction = 0
    
    for i, fits_file in enumerate(harps_fits_files):
        if i % 10 == 0:
            print(f"  Processing HARPS file {i+1}/{len(harps_fits_files)}")
        
        data = extract_harps_fits_data(fits_file, error_ratios)
        if data:
            all_data.append(data)
            if data['instrument_flag'] == 1:
                harps_pre_correction += 1
            else:
                harps_post_correction += 1
    
    print(f"HARPS data distribution:")
    print(f"  Pre-correction (BJD < {HARPS_CORRECTION_BJD}): {harps_pre_correction} files")
    print(f"  Post-correction (BJD >= {HARPS_CORRECTION_BJD}): {harps_post_correction} files")
    
    if not all_data:
        print("No valid data extracted from any files!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    df = df.sort_values('BJD').reset_index(drop=True)
    
    print(f"\nTotal records extracted: {len(df)}")
    print(f"ESPRESSO records: {len(df[df['instrument'] == 'ESPRESSO'])}")
    print(f"HARPS records: {len(df[df['instrument'] == 'HARPS'])}")
    print(f"BJD range: {df['BJD'].min():.2f} to {df['BJD'].max():.2f}")
    
    # Print instrument flag distribution
    print(f"\nInstrument flag distribution:")
    flag_counts = df['instrument_flag'].value_counts().sort_index()
    flag_names = {0: 'ESPRESSO', 1: 'HARPS pre-correction', 2: 'HARPS post-correction'}
    for flag, count in flag_counts.items():
        print(f"  Flag {flag} ({flag_names.get(flag, 'Unknown')}): {count} records")
    
    # Create output directories
    espresso_output_dir = os.path.join(espresso_data_dir, "data_files")
    harps_output_dir = os.path.join(harps_data_dir, "data_files")
    # CHANGED: Save combined files to GitHub directory instead
    combined_output_dir = "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD189567/data_HD189567"
    
    os.makedirs(espresso_output_dir, exist_ok=True)
    os.makedirs(harps_output_dir, exist_ok=True)
    os.makedirs(combined_output_dir, exist_ok=True)
    
    print(f"\nCreated ESPRESSO output directory: {espresso_output_dir}")
    print(f"Created HARPS output directory: {harps_output_dir}")
    print(f"Created combined output directory: {combined_output_dir}")

    # Save separate .dat files for each instrument
    print("\nSaving ESPRESSO .dat files...")
    save_dat_files(df, espresso_output_dir, 'ESPRESSO')
    
    print("\nSaving HARPS .dat files...")
    save_dat_files(df, harps_output_dir, 'HARPS')
    
    # Save combined .dat files to GitHub directory
    print("\nSaving combined .dat files...")
    save_combined_dat_files(df, combined_output_dir)
    
    # Load the data back and create combined plots
    print("\nLoading data for combined plotting...")
    combined_df = load_dat_files(espresso_output_dir, harps_output_dir)
    
    if len(combined_df) > 0:
        print(f"Loaded {len(combined_df)} combined records for plotting")
        
        # Create plots directory in GitHub folder
        plots_dir = "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD189567/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate combined plots
        print("\nGenerating combined activity plots with periodograms...")
        plot_activity_with_periodograms_combined(combined_df, plots_dir)
        
        print(f"\nAnalysis complete!")
        print(f"ESPRESSO .dat files saved in: {espresso_output_dir}")
        print(f"HARPS .dat files saved in: {harps_output_dir}")
        print(f"Combined .dat files saved in: {combined_output_dir}")
        print(f"Combined plots saved in: {plots_dir}")
        
        print(f"\n.dat file format: BJD, value, error, jitter, instrument_flag")
        print(f"Instrument flags: 0=ESPRESSO, 1=HARPS pre-correction, 2=HARPS post-correction")
        print(f"Jitter flags: 0=ESPRESSO, 1=HARPS")
        print(f"\nCombined files: RV_combined.dat, CCF_FWHM_combined.dat, CCF_CONTRAST_combined.dat, CCF_BIS_combined.dat")
    else:
        print("No data available for plotting!")

if __name__ == "__main__":
    main()