#!/usr/bin/env python3
"""
PyORBIT Log File Parser and Analyzer for HD102365 (emcee)

HD102365-specific adaptation of the generic emcee analyzer:
- Restricts to configuration_file_emcee_run_HD102365*.log across multiple datasets:
    * HD102365
    * HD102365_UCLES_only
    * HD102365_all_instr_ESPRESSO_GP
- Uses HD102365 filename scheme for Dataset / Planets / Configuration
- Exports model comparison tables, styled HTML
- Model comparison grouped by:
    Dataset: HD102365, HD102365_UCLES_only, HD102365_all_instr_ESPRESSO_GP
    ConfigGroup:
        - all_instr, espresso_only, no_espresso (original HD102365)
        - UCLES_only
        - all_instr_ESPRESSO_GP
    GPType: gp, no_gp (gp and no_gp are NEVER compared directly)
  Within each (Dataset, ConfigGroup, GPType), 0p/1p/2p/3p models are compared by BIC.
"""

import re
import os
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime


def calculate_t0_from_mean_long(mean_long_deg, omega_deg, period_days, reference_epoch=0.0):
    """
    Calculate time of periastron (t0) from mean longitude.
    
    Formula: t0 = t_ref - (mean_long - omega) / n
    where n = 2π/P is the mean motion
    """
    mean_long_rad = np.deg2rad(mean_long_deg)
    omega_rad = np.deg2rad(omega_deg)
    mean_anomaly_rad = mean_long_rad - omega_rad
    n = 2.0 * np.pi / period_days
    dt = mean_anomaly_rad / n
    t0 = reference_epoch - dt
    return t0


def parse_log_file(filepath):
    """
    Parses a PyORBIT emcee log file to extract Median BIC, convergence status,
    orbital parameters, and activity parameters.

    IMPORTANT: emcee logs contain multiple "Statistics on the model parameters obtained..."
    blocks; for HD102365 you want the THIRD one (index 2).
    """
    median_bic = None
    gelman_rubin_values = []
    orbital_parameters = {}
    activity_parameters = {}

    try:
        with open(filepath, 'r') as f:
            content = f.read()

            # Median BIC
            bic_match = re.search(
                r'Median BIC\s+\(using likelihood\)\s*=\s*(-?[\d\.]+)', content
            )
            if bic_match:
                median_bic = float(bic_match.group(1))

            # Gelman-Rubin
            gr_pattern = (
                r'Gelman-Rubin:\s+(\d+)\s+([\d\.]+)\s+'
                r'([a-zA-Z_][a-zA-Z_0-9]*(?:_[a-zA-Z_][a-zA-Z_0-9]*)*)\s*$'
            )
            gr_matches = re.findall(gr_pattern, content, re.MULTILINE)

            gr_dict = {}
            for match in gr_matches:
                param_name = match[2]
                gr_value = float(match[1])
                gr_dict[param_name] = gr_value
                gelman_rubin_values.append(gr_value)

            # --- Select the 3rd stats block ---
            lines = content.split('\n')
            stats_header = "Statistics on the model parameters obtained from the posteriors samples"
            stats_indices = [i for i, line in enumerate(lines) if stats_header in line]

            if len(stats_indices) >= 3:
                stats_start_idx = stats_indices[2]  # third block (0-based)
            elif len(stats_indices) > 0:
                # fallback: last block if fewer than 3 exist
                stats_start_idx = stats_indices[-1]
            else:
                stats_start_idx = -1

            if stats_start_idx != -1:
                current_planet = None
                in_activity_section = False

                # robust number (supports scientific notation)
                num = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'

                # stop markers: end of this stats section (varies between runs)
                stop_markers = (
                    "Statistics on the derived parameters",
                    "Parameters corresponding to",
                    "*** Information criteria",
                    "Gelman-Rubin:",
                )

                for i in range(stats_start_idx, len(lines)):
                    line = lines[i]

                    if any(m in line for m in stop_markers) and i > stats_start_idx:
                        break

                    # Planet header
                    planet_match = re.search(r'----- common model:\s+([a-z])\s*$', line)
                    if planet_match:
                        current_planet = planet_match.group(1)
                        in_activity_section = False
                        orbital_parameters.setdefault(current_planet, {})
                        continue

                    # Activity header
                    if "----- common model:  activity" in line:
                        in_activity_section = True
                        current_planet = None
                        continue

                    s = line.strip()
                    if not s:
                        continue

                    # Preferred: "param median err_minus err_plus ... (15-84 p)"
                    m4 = re.match(
                        rf'^([A-Za-z_]+)\s+({num})\s+({num})\s+({num}).*\(15-84 p\)',
                        s
                    )
                    # Fallback: "param value"
                    m2 = re.match(rf'^([A-Za-z_]+)\s+({num})\s*$', s)

                    if m4:
                        param_name = m4.group(1)
                        median_val = float(m4.group(2))
                        err_minus_raw = float(m4.group(3))  # often negative
                        err_plus_raw = float(m4.group(4))   # often positive
                        err_minus = abs(err_minus_raw)
                        err_plus = abs(err_plus_raw)
                    elif m2:
                        param_name = m2.group(1)
                        median_val = float(m2.group(2))
                        err_minus = None
                        err_plus = None
                    else:
                        continue

                    if in_activity_section:
                        full_param_name = f'activity_{param_name}'
                        gr_value = gr_dict.get(full_param_name, None)
                        activity_parameters[param_name] = {
                            'value': median_val,
                            'err_minus': err_minus,
                            'err_plus': err_plus,
                            'gelman_rubin': gr_value
                        }
                    elif current_planet:
                        full_param_name = f'{current_planet}_{param_name}'
                        gr_value = gr_dict.get(full_param_name, None)
                        orbital_parameters[current_planet][param_name] = {
                            'value': median_val,
                            'err_minus': err_minus,
                            'err_plus': err_plus,
                            'gelman_rubin': gr_value
                        }

                # sre_coso / sre_sino: GR-only if present in GR list but not in stats block
                for planet in list(orbital_parameters.keys()):
                    coso_key = f'{planet}_sre_coso'
                    if coso_key in gr_dict:
                        orbital_parameters[planet].setdefault('sre_coso', {})
                        orbital_parameters[planet]['sre_coso']['gelman_rubin'] = gr_dict[coso_key]

                    sino_key = f'{planet}_sre_sino'
                    if sino_key in gr_dict:
                        orbital_parameters[planet].setdefault('sre_sino', {})
                        orbital_parameters[planet]['sre_sino']['gelman_rubin'] = gr_dict[sino_key]

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while reading {filepath}: {e}")
        return None

    if median_bic is None:
        return None

    # Convergence stats
    if gelman_rubin_values:
        converged_count = sum(1 for gr in gelman_rubin_values if gr < 1.1)
        total_count = len(gelman_rubin_values)
        convergence_pct = (converged_count / total_count * 100) if total_count > 0 else 0
        max_gr = max(gelman_rubin_values)
    else:
        convergence_pct = 0
        max_gr = None

    basename = os.path.basename(filepath)
    cleaned_name = basename.replace('configuration_file_emcee_run_', '').replace('.log', '')

    dataset = 'Unknown'
    config_name = cleaned_name

    if cleaned_name.startswith('HD102365_UCLES_only_'):
        dataset = 'HD102365_UCLES_only'
        config_name = cleaned_name[len('HD102365_UCLES_only_'):]
    elif cleaned_name.startswith('HD102365_all_instr_ESPRESSO_GP_'):
        dataset = 'HD102365_all_instr_ESPRESSO_GP'
        config_name = cleaned_name[len('HD102365_all_instr_ESPRESSO_GP_'):]
    elif cleaned_name.startswith('HD102365_'):
        dataset = 'HD102365'
        config_name = cleaned_name[len('HD102365_'):]
    else:
        pass

    pm = re.search(r'_(\dp)_emcee', config_name)
    num_planets = pm.group(1) if pm else 'N/A'

    directory_name = os.path.dirname(os.path.abspath(filepath))

    return {
        'Configuration': config_name,
        'Dataset': dataset,
        'Planets': num_planets,
        'Median BIC': median_bic,
        'Convergence %': convergence_pct,
        'Max GR': max_gr,
        'Orbital Parameters': orbital_parameters,
        'Activity Parameters': activity_parameters,
        'File': basename,
        'Directory': directory_name
    }


def _derive_config_group_fields(df):
    """
    Add ConfigGroup / GPType / ModelName / ModelLabel columns to df based on
    Dataset + Configuration.

    Dataset:
      - 'HD102365'                        -> original grids
      - 'HD102365_UCLES_only'            -> UCLES-only grids
      - 'HD102365_all_instr_ESPRESSO_GP' -> all-instr, ESPRESSO-only GP grids

    ConfigGroup:
      - all_instr, espresso_only, no_espresso (original HD102365)
      - UCLES_only
      - all_instr_ESPRESSO_GP
    """
    def split_row(row):
        config_name = row['Configuration']
        dataset = row['Dataset']

        group = 'other'
        gp_flag = 'unknown'
        nplanets = 'Np'

        if dataset == 'HD102365':
            # Original naming has explicit prefixes
            if config_name.startswith('all_instr_'):
                group = 'all_instr'
            elif config_name.startswith('espresso_only_'):
                group = 'espresso_only'
            elif config_name.startswith('no_espresso_'):
                group = 'no_espresso'
        elif dataset == 'HD102365_UCLES_only':
            group = 'UCLES_only'
        elif dataset == 'HD102365_all_instr_ESPRESSO_GP':
            group = 'all_instr_ESPRESSO_GP'

        # Planets: look for '_<digit>p_emcee' OR '<digit>p_emcee'
        m_planets = re.search(r'_(\dp)_emcee', config_name)
        if not m_planets:
            m_planets = re.search(r'^(\dp)_emcee', config_name)
        if m_planets:
            nplanets = m_planets.group(1)

        # GPType:
        # Original configs contain '_gp_' / '_no_gp_'
        # UCLES-only configs look like 'gp_1p_emcee' / 'no_gp_2p_emcee'
        if '_no_gp_' in config_name or config_name.startswith('no_gp_'):
            gp_flag = 'no_gp'
        elif '_gp_' in config_name or config_name.startswith('gp_'):
            gp_flag = 'gp'
        else:
            # For the ESPRESSO-GP-only grid, everything is GP by construction
            if dataset == 'HD102365_all_instr_ESPRESSO_GP':
                gp_flag = 'gp'

        model_label = f"{gp_flag}_{nplanets}"
        full_model_name = f"{gp_flag}_{nplanets}_emcee"
        return pd.Series([group, gp_flag, full_model_name, model_label])

    df[['ConfigGroup', 'GPType', 'ModelName', 'ModelLabel']] = df.apply(
        split_row, axis=1
    )

    # Keep only the desired groups
    df = df[df['ConfigGroup'].isin(
        ['all_instr', 'espresso_only', 'no_espresso', 'UCLES_only', 'all_instr_ESPRESSO_GP']
    )].copy()
    return df


def analyze_and_display(log_files, search_directory=None, fit_type="multiple"):
    """
    Analyze emcee log files and print formatted comparison tables.
    Export:
      - styled HTML report,
      - compact best-models CSV (now includes parameter error bars).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "results_HD102365")
    os.makedirs(output_dir, exist_ok=True)

    def fmt_val_err(pdict, ndigits=4):
        """
        pdict: {'value': v, 'err_minus': em, 'err_plus': ep, ...}
        -> 'v (-em, +ep)' if errors exist else 'v'
        """
        if not isinstance(pdict, dict) or pdict.get('value') is None:
            return '-'
        v = pdict.get('value')
        em = pdict.get('err_minus', None)
        ep = pdict.get('err_plus', None)

        try:
            v = float(v)
        except Exception:
            return str(v)

        if em is None or ep is None:
            return f"{v:.{ndigits}f}"

        try:
            em = abs(float(em))
            ep = abs(float(ep))
        except Exception:
            return f"{v:.{ndigits}f}"

        return f"{v:.{ndigits}f} (-{em:.{ndigits}f}, +{ep:.{ndigits}f})"

    def get_triplet(pdict):
        """Return (value, err_minus, err_plus) as strings for CSV."""
        if not isinstance(pdict, dict):
            return ('', '', '')
        v = pdict.get('value', '')
        em = pdict.get('err_minus', '')
        ep = pdict.get('err_plus', '')
        return (
            '' if v is None else str(v),
            '' if em is None else str(em),
            '' if ep is None else str(ep),
        )

    all_data = []
    failed_files = []

    print(f"Processing {len(log_files)} log files...")
    for log_file in log_files:
        data = parse_log_file(log_file)
        if data:
            all_data.append(data)
        else:
            failed_files.append(log_file)

    if failed_files:
        print(f"\nWarning: {len(failed_files)} files could not be parsed (missing Median BIC data):")
        for f in failed_files:
            print(f"  - {f}")
        print()

    if not all_data:
        print("No data could be extracted from any log files found.")
        return

    df = pd.DataFrame(all_data)
    df = _derive_config_group_fields(df)

    datasets = sorted(df['Dataset'].unique())

    print("--- Model Comparison by Dataset, ConfigGroup and GPType "
          "(all_instr / espresso_only / no_espresso / UCLES_only / all_instr_ESPRESSO_GP; gp / no_gp) ---\n")

    export_data = []

    for dataset in datasets:
        dataset_df = df[df['Dataset'] == dataset]

        print(f"\n{'='*80}")
        print(f"DATASET: {dataset}")
        print(f"{'='*80}\n")

        top_groups = dataset_df.groupby('ConfigGroup')

        for cfg_group_name, cfg_group_df in top_groups:
            print(f"=== ConfigGroup: {cfg_group_name} ===\n")

            gp_groups = cfg_group_df.groupby('GPType')

            for gp_type, gp_df in gp_groups:
                print(f"--- GPType: {gp_type} ---\n")

                grouped = gp_df.groupby('Configuration')
                best_models_info = []

                for config_name, group in grouped:
                    print(f"Configuration: {config_name}\n")

                    group = group.copy()
                    group['Planets'] = pd.Categorical(
                        group['Planets'], categories=['0p', '1p', '2p', '3p'], ordered=True
                    )
                    group = group.sort_values('Planets')

                    min_bic_idx = group['Median BIC'].idxmin()
                    min_bic_value = group.loc[min_bic_idx, 'Median BIC']

                    display_group = group[['Planets', 'Median BIC', 'Convergence %', 'Max GR']].copy()
                    display_group['ΔBIC'] = group['Median BIC'] - min_bic_value

                    group_copy = group.copy()
                    group_copy['Preferred_BIC'] = group_copy.index == min_bic_idx
                    group_copy['ΔBIC'] = group_copy['Median BIC'] - min_bic_value
                    group_copy['Dataset'] = dataset
                    group_copy['ConfigGroup'] = cfg_group_name
                    group_copy['GPType'] = gp_type
                    group_copy['Configuration'] = config_name

                    export_group = group_copy[['Dataset', 'ConfigGroup', 'GPType',
                                               'Configuration', 'Planets',
                                               'Median BIC', 'ΔBIC',
                                               'Convergence %', 'Max GR',
                                               'Preferred_BIC',
                                               'Orbital Parameters', 'Activity Parameters',
                                               'File', 'Directory']]
                    export_data.append(export_group)

                    print(display_group.to_string(index=False))
                    print(f"\nBest BIC in this Configuration: {group.loc[min_bic_idx, 'Planets']} model "
                          f"(BIC = {group.loc[min_bic_idx, 'Median BIC']:.2f})")
                    print("\n" + "-"*80 + "\n")

                    best_model_data = group.loc[min_bic_idx]
                    best_models_info.append({
                        'Configuration': config_name,
                        'ConfigGroup': cfg_group_name,
                        'GPType': gp_type,
                        'Best Model': best_model_data['Planets'],
                        'BIC': best_model_data['Median BIC'],
                        'Convergence %': best_model_data['Convergence %'],
                        'Max GR': best_model_data['Max GR'],
                        'Orbital Parameters': best_model_data['Orbital Parameters'],
                        'Activity Parameters': best_model_data['Activity Parameters']
                    })

                if best_models_info:
                    print(f"{'-'*180}")
                    print(f"ORBITAL & ACTIVITY PARAMETERS SUMMARY FOR {dataset} / {cfg_group_name} / {gp_type}")
                    print(f"{'-'*180}\n")

                    print("TABLE 1: ORBITAL PARAMETERS\n")
                    print(f"{'Config':<30} {'Model':<8} {'Planet':<8} "
                          f"{'P (days)':<30} {'P_GR':<8} "
                          f"{'K (m/s)':<30} {'K_GR':<8} "
                          f"{'mean_long':<12} {'ml_GR':<8} "
                          f"{'e':<30} {'ω (deg)':<10} "
                          f"{'coso_GR':<10} {'sino_GR':<10}")
                    print("-" * 180)

                    for model_info in best_models_info:
                        config_name = model_info['Configuration']
                        best_model = model_info['Best Model']
                        orbital_params = model_info['Orbital Parameters']

                        planets = sorted(orbital_params.keys()) if orbital_params else []

                        if not planets:
                            print(f"{config_name:<30} {best_model:<8} {'-':<8} "
                                  f"{'-':<30} {'-':<8} "
                                  f"{'-':<30} {'-':<8} "
                                  f"{'-':<12} {'-':<8} "
                                  f"{'-':<30} {'-':<10} "
                                  f"{'-':<10} {'-':<10}")
                        else:
                            for planet in planets:
                                planet_params = orbital_params[planet]

                                p_val = fmt_val_err(planet_params.get('P'), ndigits=4)
                                p_gr = planet_params.get('P', {}).get('gelman_rubin')
                                p_gr_str = f"{p_gr:.3f}" if p_gr is not None else '-'

                                k_val = fmt_val_err(planet_params.get('K'), ndigits=4)
                                k_gr = planet_params.get('K', {}).get('gelman_rubin')
                                k_gr_str = f"{k_gr:.3f}" if k_gr is not None else '-'

                                ml_v = planet_params.get('mean_long', {}).get('value')
                                ml_val = f"{ml_v:.2f}" if ml_v is not None else '-'
                                ml_gr = planet_params.get('mean_long', {}).get('gelman_rubin')
                                ml_gr_str = f"{ml_gr:.3f}" if ml_gr is not None else '-'

                                e_str = fmt_val_err(planet_params.get('e'), ndigits=4)

                                omega_v = planet_params.get('omega', {}).get('value')
                                omega_str = f"{omega_v:.2f}" if omega_v is not None else '-'

                                coso_gr = planet_params.get('sre_coso', {}).get('gelman_rubin')
                                sino_gr = planet_params.get('sre_sino', {}).get('gelman_rubin')
                                coso_gr_str = f"{coso_gr:.3f}" if coso_gr is not None else '-'
                                sino_gr_str = f"{sino_gr:.3f}" if sino_gr is not None else '-'

                                print(f"{config_name:<30} {best_model:<8} {planet:<8} "
                                      f"{p_val:<30} {p_gr_str:<8} "
                                      f"{k_val:<30} {k_gr_str:<8} "
                                      f"{ml_val:<12} {ml_gr_str:<8} "
                                      f"{e_str:<30} {omega_str:<10} "
                                      f"{coso_gr_str:<10} {sino_gr_str:<10}")

                    print()

                    print("TABLE 2: ACTIVITY PARAMETERS\n")
                    print(f"{'Config':<30} {'Model':<8} "
                          f"{'Prot (days)':<30} {'Prot_GR':<10} "
                          f"{'Pdec (days)':<30} {'Pdec_GR':<10} "
                          f"{'Oamp':<30} {'Oamp_GR':<10}")
                    print("-" * 170)

                    for model_info in best_models_info:
                        config_name = model_info['Configuration']
                        best_model = model_info['Best Model']
                        activity_params = model_info['Activity Parameters'] or {}

                        prot_val = fmt_val_err(activity_params.get('Prot'), ndigits=4)
                        prot_gr = activity_params.get('Prot', {}).get('gelman_rubin')
                        prot_gr_str = f"{prot_gr:.3f}" if prot_gr is not None else '-'

                        pdec_val = fmt_val_err(activity_params.get('Pdec'), ndigits=4)
                        pdec_gr = activity_params.get('Pdec', {}).get('gelman_rubin')
                        pdec_gr_str = f"{pdec_gr:.3f}" if pdec_gr is not None else '-'

                        oamp_val = fmt_val_err(activity_params.get('Oamp'), ndigits=4)
                        oamp_gr = activity_params.get('Oamp', {}).get('gelman_rubin')
                        oamp_gr_str = f"{oamp_gr:.3f}" if oamp_gr is not None else '-'

                        print(f"{config_name:<30} {best_model:<8} "
                              f"{prot_val:<30} {prot_gr_str:<10} "
                              f"{pdec_val:<30} {pdec_gr_str:<10} "
                              f"{oamp_val:<30} {oamp_gr_str:<10}")

                    print("\n" + "="*180 + "\n")

    if export_data:
        combined_df = pd.concat(export_data, ignore_index=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # HTML report
        html_filename = f"HD102365_emcee_model_comparison_{timestamp}.html"
        html_filepath = os.path.join(output_dir, html_filename)

        html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>PyORBIT Model Comparison Results (HD102365 emcee)</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
               margin: 0; padding: 20px; background-color: #fafafa; color: #2c3e50; }
        .container { max-width: 100%; margin: 0 auto; padding: 0 20px; }
        h1 { text-align: center; color: #2c3e50; font-weight: 600; margin-bottom: 30px; font-size: 32px; }
        .dataset-section { background-color: white; margin: 30px 0; border-radius: 12px;
                           box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow: hidden; }
        .dataset-header { background-color: #2c3e50; color: white; padding: 20px 30px; font-size: 22px; font-weight: 600; }
        .config-section { margin: 0; padding: 25px 30px; border-bottom: 1px solid #ecf0f1; }
        .config-section:last-child { border-bottom: none; }
        h2 { color: #34495e; margin: 20px 0 10px 0; font-size: 18px; font-weight: 600; }
        h3 { color: #2c3e50; margin: 15px 0 8px 0; font-size: 15px; font-weight: 600;
             background-color: #ecf0f1; padding: 8px 12px; border-radius: 5px; }
        .gp-label { font-size: 13px; font-weight: 600; margin-top: 5px; margin-bottom: 3px; color: #2c3e50; }
        table { border-collapse: collapse; margin: 15px 0; width: 100%; background-color: white; font-size: 13px; }
        th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid #ecf0f1; }
        th { background-color: #f8f9fa; color: #2c3e50; font-weight: 600; font-size: 13px;
             text-transform: uppercase; letter-spacing: 0.5px; }
        td { color: #34495e; font-size: 14px; }
        tr:hover td { background-color: #f8f9fa; }
        .highlight-bic { background-color: #e8f5e9 !important; font-weight: 600; color: #2e7d32; }
        .convergence-good { color: #2e7d32; font-weight: 600; }
        .convergence-warning { color: #f57c00; font-weight: 600; }
        .convergence-bad { color: #c62828; font-weight: 600; }
        .params-summary-section { margin: 20px; padding: 20px; background-color: #f8f9fa; border-radius: 8px;
                                 border-left: 4px solid #3498db; }
        .params-table { border-collapse: collapse; margin: 10px 0; width: 100%; background-color: white;
                        border: 2px solid #3498db; font-size: 12px; }
        .params-table th { background-color: #3498db; color: white; padding: 8px 6px; text-align: center;
                           font-weight: 600; border: 1px solid #2980b9; font-size: 11px; }
        .params-table td { padding: 6px 6px; border: 1px solid #bdc3c7; text-align: center; }
        .params-table tr:nth-child(even) { background-color: #f8f9fa; }
        .params-table tr:hover { background-color: #e8f4f8; }
        .gr-converged { color: #2e7d32; font-weight: 600; }
        .gr-not-converged { color: #c62828; font-weight: 600; }
        .legend { margin: 0 0 30px 0; padding: 20px 25px; background-color: white; border-radius: 12px;
                  box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .legend h3 { margin: 0 0 15px 0; color: #2c3e50; font-size: 18px; font-weight: 600; background: none; padding: 0; }
        .legend-item { margin: 8px 0; color: #34495e; font-size: 14px; line-height: 1.6; }
        .badge { display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 13px; font-weight: 600; }
        .badge-green { background-color: #e8f5e9; color: #2e7d32; }
    </style>
</head>
<body>
    <div class="container">
        <h1>PyORBIT Model Comparison Results (HD102365 emcee)</h1>

        <div class="legend">
            <h3>Legend</h3>
            <div class="legend-item"><span class="badge badge-green">Green highlight</span> = Best BIC within a given (Dataset, ConfigGroup, GPType)</div>
            <div class="legend-item"><strong>ConfigGroup:</strong> all_instr, espresso_only, no_espresso, UCLES_only, all_instr_ESPRESSO_GP</div>
            <div class="legend-item"><strong>GPType:</strong> gp or no_gp; gp and no_gp are not compared directly.</div>
            <div class="legend-item"><strong>Convergence %:</strong> Percentage of parameters with Gelman-Rubin &lt; 1.1</div>
            <div class="legend-item"><strong>Max GR:</strong> Maximum Gelman-Rubin value across all parameters</div>
        </div>
"""

        def format_gr(gr_value):
            if gr_value is None:
                return '-', ''
            try:
                gr_float = float(gr_value)
                gr_class = 'gr-converged' if gr_float < 1.1 else 'gr-not-converged'
                return f'{gr_float:.3f}', gr_class
            except Exception:
                return '-', ''

        for dataset in datasets:
            html_content += f'        <div class="dataset-section">\n'
            html_content += f'            <div class="dataset-header">{dataset}</div>\n'

            dataset_df = df[df['Dataset'] == dataset]
            top_groups = dataset_df.groupby('ConfigGroup')

            for cfg_group_name, cfg_group_df in top_groups:
                html_content += f'            <div class="config-section">\n'
                html_content += f'                <h2>ConfigGroup: {cfg_group_name}</h2>\n'

                gp_groups = cfg_group_df.groupby('GPType')

                for gp_type, gp_df in gp_groups:
                    html_content += f'                <div class="gp-label">GPType: {gp_type}</div>\n'

                    grouped = gp_df.groupby('Configuration')
                    best_models_info = []

                    gp_min_bic = gp_df['Median BIC'].min()

                    html_content += '                <table>\n'
                    html_content += '                    <tr><th>Configuration</th><th>Planets</th><th>Median BIC</th><th>ΔBIC (GPType)</th><th>Convergence %</th><th>Max GR</th></tr>\n'

                    for config_name, group in grouped:
                        group = group.copy()
                        group['Planets'] = pd.Categorical(
                            group['Planets'], categories=['0p', '1p', '2p', '3p'], ordered=True
                        )
                        group = group.sort_values('Planets')
                        min_bic_idx = group['Median BIC'].idxmin()
                        row = group.loc[min_bic_idx]

                        delta_bic = row['Median BIC'] - gp_min_bic
                        bic_class = 'highlight-bic' if row['Median BIC'] == gp_min_bic else ''

                        conv_pct = row['Convergence %']
                        if conv_pct == 100:
                            conv_class = 'convergence-good'
                        elif conv_pct >= 90:
                            conv_class = 'convergence-warning'
                        else:
                            conv_class = 'convergence-bad'

                        max_gr_str = f"{row['Max GR']:.4f}" if row['Max GR'] is not None else "-"

                        html_content += '                    <tr>\n'
                        html_content += f'                        <td>{config_name}</td>\n'
                        html_content += f'                        <td>{row["Planets"]}</td>\n'
                        html_content += f'                        <td class="{bic_class}">{row["Median BIC"]:.2f}</td>\n'
                        html_content += f'                        <td class="{bic_class}">{delta_bic:.2f}</td>\n'
                        html_content += f'                        <td class="{conv_class}">{conv_pct:.1f}%</td>\n'
                        html_content += f'                        <td>{max_gr_str}</td>\n'
                        html_content += '                    </tr>\n'

                        best_models_info.append({
                            'Configuration': config_name,
                            'Best Model': row['Planets'],
                            'BIC': row['Median BIC'],
                            'Orbital Parameters': row['Orbital Parameters'],
                            'Activity Parameters': row['Activity Parameters']
                        })

                    html_content += '                </table>\n'

                    if best_models_info:
                        html_content += '                <div class="params-summary-section">\n'
                        html_content += f'                    <h3>Parameters Summary (Best Models in GPType = {gp_type})</h3>\n'

                        # Orbital params (with errors)
                        html_content += '                    <table class="params-table">\n'
                        html_content += '                        <tr><th>Config</th><th>Model</th><th>Planet</th>'
                        html_content += '<th>P (days)</th><th>P_GR</th><th>K (m/s)</th><th>K_GR</th>'
                        html_content += '<th>mean_long (°)</th><th>ml_GR</th>'
                        html_content += '<th>e</th><th>ω (°)</th>'
                        html_content += '<th>coso_GR</th><th>sino_GR</th></tr>\n'

                        for model_info in best_models_info:
                            config_name = model_info['Configuration']
                            best_model = model_info['Best Model']
                            orbital_params = model_info['Orbital Parameters']

                            planets = sorted(orbital_params.keys()) if orbital_params else []

                            if not planets:
                                html_content += '                        <tr>\n'
                                html_content += f'                            <td>{config_name}</td><td>{best_model}</td><td>-</td>\n'
                                html_content += '                            <td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>\n'
                                html_content += '                            <td>-</td><td>-</td><td>-</td><td>-</td>\n'
                                html_content += '                        </tr>\n'
                            else:
                                for i, planet in enumerate(planets):
                                    planet_params = orbital_params[planet]

                                    p_val = fmt_val_err(planet_params.get('P'), ndigits=4)
                                    p_gr, p_gr_class = format_gr(planet_params.get('P', {}).get('gelman_rubin'))

                                    k_val = fmt_val_err(planet_params.get('K'), ndigits=4)
                                    k_gr, k_gr_class = format_gr(planet_params.get('K', {}).get('gelman_rubin'))

                                    ml_v = planet_params.get('mean_long', {}).get('value')
                                    ml_val = f"{ml_v:.2f}" if ml_v is not None else '-'
                                    ml_gr, ml_gr_class = format_gr(planet_params.get('mean_long', {}).get('gelman_rubin'))

                                    e_str = fmt_val_err(planet_params.get('e'), ndigits=4)

                                    omega_v = planet_params.get('omega', {}).get('value')
                                    omega_str = f"{omega_v:.2f}" if omega_v is not None else '-'

                                    coso_gr, coso_gr_class = format_gr(planet_params.get('sre_coso', {}).get('gelman_rubin'))
                                    sino_gr, sino_gr_class = format_gr(planet_params.get('sre_sino', {}).get('gelman_rubin'))

                                    html_content += '                        <tr>\n'
                                    if i == 0:
                                        html_content += f'                            <td rowspan="{len(planets)}">{config_name}</td>\n'
                                        html_content += f'                            <td rowspan="{len(planets)}">{best_model}</td>\n'
                                    html_content += f'                            <td>{planet}</td>\n'
                                    html_content += f'                            <td>{p_val}</td><td class="{p_gr_class}">{p_gr}</td>\n'
                                    html_content += f'                            <td>{k_val}</td><td class="{k_gr_class}">{k_gr}</td>\n'
                                    html_content += f'                            <td>{ml_val}</td><td class="{ml_gr_class}">{ml_gr}</td>\n'
                                    html_content += f'                            <td>{e_str}</td><td>{omega_str}</td>\n'
                                    html_content += f'                            <td class="{coso_gr_class}">{coso_gr}</td><td class="{sino_gr_class}">{sino_gr}</td>\n'
                                    html_content += '                        </tr>\n'

                        html_content += '                    </table>\n'

                        # Activity params (with errors)
                        html_content += '                    <table class="params-table">\n'
                        html_content += '                        <tr><th>Config</th><th>Model</th>'
                        html_content += '<th>Prot (days)</th><th>Prot_GR</th><th>Pdec (days)</th><th>Pdec_GR</th>'
                        html_content += '<th>Oamp</th><th>Oamp_GR</th></tr>\n'

                        for model_info in best_models_info:
                            config_name = model_info['Configuration']
                            best_model = model_info['Best Model']
                            activity_params = model_info['Activity Parameters'] or {}

                            prot_val = fmt_val_err(activity_params.get('Prot'), ndigits=4)
                            prot_gr, prot_gr_class = format_gr(activity_params.get('Prot', {}).get('gelman_rubin'))

                            pdec_val = fmt_val_err(activity_params.get('Pdec'), ndigits=4)
                            pdec_gr, pdec_gr_class = format_gr(activity_params.get('Pdec', {}).get('gelman_rubin'))

                            oamp_val = fmt_val_err(activity_params.get('Oamp'), ndigits=4)
                            oamp_gr, oamp_gr_class = format_gr(activity_params.get('Oamp', {}).get('gelman_rubin'))

                            html_content += '                        <tr>\n'
                            html_content += f'                            <td>{config_name}</td><td>{best_model}</td>\n'
                            html_content += f'                            <td>{prot_val}</td><td class="{prot_gr_class}">{prot_gr}</td>\n'
                            html_content += f'                            <td>{pdec_val}</td><td class="{pdec_gr_class}">{pdec_gr}</td>\n'
                            html_content += f'                            <td>{oamp_val}</td><td class="{oamp_gr_class}">{oamp_gr}</td>\n'
                            html_content += '                        </tr>\n'

                        html_content += '                    </table>\n'
                        html_content += '                </div>\n'

                html_content += '            </div>\n'
            html_content += '        </div>\n'

        html_content += """
    </div>
</body>
</html>
"""
        with open(html_filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # ---------- Compact best-models CSV (now with errors) ----------
        cfg_best = combined_df[combined_df['Preferred_BIC']].copy()

        idx = (
            cfg_best
            .groupby(['Dataset', 'ConfigGroup', 'GPType'])['Median BIC']
            .idxmin()
        )
        final_best = cfg_best.loc[idx].sort_values(['Dataset', 'ConfigGroup', 'GPType'])

        best_rows = []

        for _, row in final_best.iterrows():
            dataset = row['Dataset']
            cfg_group = row['ConfigGroup']
            gptype = row['GPType']
            config = row['Configuration']
            model_planets = row['Planets']
            bic = row['Median BIC']
            dBIC = row['ΔBIC']
            filename = row['File']
            directory = row['Directory']

            orb = row['Orbital Parameters']
            act = row['Activity Parameters']

            planet_names = []

            p_list, p_em_list, p_ep_list = [], [], []
            k_list, k_em_list, k_ep_list = [], [], []
            e_list, e_em_list, e_ep_list = [], [], []

            if isinstance(orb, dict) and orb:
                for planet_label in sorted(orb.keys()):
                    pdata = orb[planet_label]
                    planet_names.append(planet_label)

                    p_v, p_em, p_ep = get_triplet(pdata.get('P', {}))
                    k_v, k_em, k_ep = get_triplet(pdata.get('K', {}))
                    e_v, e_em, e_ep = get_triplet(pdata.get('e', {}))

                    p_list.append(p_v);   p_em_list.append(p_em);   p_ep_list.append(p_ep)
                    k_list.append(k_v);   k_em_list.append(k_em);   k_ep_list.append(k_ep)
                    e_list.append(e_v);   e_em_list.append(e_em);   e_ep_list.append(e_ep)

            if isinstance(act, dict) and act:
                prot_val = act.get('Prot', {}).get('value', np.nan)
                pdec_val = act.get('Pdec', {}).get('value', np.nan)
                oamp_val = act.get('Oamp', {}).get('value', np.nan)
            else:
                prot_val = pdec_val = oamp_val = np.nan

            best_rows.append({
                'Dataset': dataset,
                'ConfigGroup': cfg_group,
                'GPType': gptype,
                'Configuration': config,
                'Best_Model_Planets': model_planets,
                'Median BIC': bic,
                'ΔBIC': dBIC,

                'Planets': ','.join(planet_names) if planet_names else '',

                'P_days': ','.join(p_list) if p_list else '',
                'P_err_minus': ','.join(p_em_list) if p_em_list else '',
                'P_err_plus': ','.join(p_ep_list) if p_ep_list else '',

                'K_mps': ','.join(k_list) if k_list else '',
                'K_err_minus': ','.join(k_em_list) if k_em_list else '',
                'K_err_plus': ','.join(k_ep_list) if k_ep_list else '',

                'e': ','.join(e_list) if e_list else '',
                'e_err_minus': ','.join(e_em_list) if e_em_list else '',
                'e_err_plus': ','.join(e_ep_list) if e_ep_list else '',

                'Prot_days': prot_val,
                'Pdec_days': pdec_val,
                'Oamp': oamp_val,

                'File': filename,
                'Directory': directory,
            })

        best_df = pd.DataFrame(best_rows)
        best_csv_name = f"HD102365_emcee_best_models_summary_{timestamp}.csv"
        best_csv_path = os.path.join(output_dir, best_csv_name)
        best_df.to_csv(best_csv_path, index=False)

        print(f"\n{'='*80}")
        print("Results exported to:")
        print(f"  - HTML:            {html_filepath}")
        print(f"  - Best-model CSV:  {best_csv_path} (one row per Dataset × ConfigGroup × GPType)")
        print("\nNotes:")
        print("  - Comparison is done separately for each GPType (gp vs gp, no_gp vs no_gp).")
        print("  - Preferred_BIC=True marks the lowest-BIC model within each Configuration.")
        print("  - The best-model CSV keeps only the globally best configuration per (Dataset × ConfigGroup × GPType).")
        print(f"{'='*80}\n")


def main():
    """
    Main function to run the HD102365 emcee analyzer.

    Usage:
        python compare_emcee_models_HD102365.py [directory] [fit_type]

    directory: root folder containing emcee logs (e.g. ".." from HD102365/)
    fit_type:  'multiple' (default) or 'single'  [currently only controls printing]
    """
    if len(sys.argv) > 1:
        search_directory = sys.argv[1]
    else:
        search_directory = "."
    
    fit_type = "multiple"
    if len(sys.argv) > 2:
        fit_type_arg = sys.argv[2].lower()
        if fit_type_arg in ['multiple', 'single']:
            fit_type = fit_type_arg
        else:
            print(f"Warning: Invalid fit_type '{sys.argv[2]}'. Using default 'multiple'.")
            print("Valid options: 'multiple' or 'single'")
    
    if not os.path.isdir(search_directory):
        print(f"Error: Directory '{search_directory}' does not exist.")
        sys.exit(1)
    
    print(f"Searching for emcee log files in: {os.path.abspath(search_directory)}\n")
    
    all_log_files = glob.glob(os.path.join(search_directory, '**', '*.log'), recursive=True)
    
    # HD102365-specific: only emcee logs for this star (all three families)
    all_log_files = [
        f for f in all_log_files
        if os.path.basename(f).startswith('configuration_file_emcee_run_')
        and 'HD102365' in os.path.basename(f)
    ]
    
    file_groups = {}
    for log_file in all_log_files:
        basename = os.path.basename(log_file)
        if basename not in file_groups:
            file_groups[basename] = []
        file_groups[basename].append(log_file)
    
    log_files = []
    for basename, files in file_groups.items():
        if len(files) == 1:
            log_files.append(files[0])
        else:
            best_file = min(files, key=lambda f: f.count(os.sep))
            log_files.append(best_file)
            print(f"Note: Found {len(files)} copies of '{basename}', using: {best_file}")
    
    if not log_files:
        print("No log files found matching 'configuration_file_emcee_run_HD102365*.log'")
        sys.exit(1)
    
    print(f"\nFound {len(log_files)} unique log files to analyze (filtered from {len(all_log_files)} total).")
    print(f"Using fit_type: {fit_type}\n")
    
    analyze_and_display(log_files, search_directory=search_directory, fit_type=fit_type)


if __name__ == "__main__":
    main()
