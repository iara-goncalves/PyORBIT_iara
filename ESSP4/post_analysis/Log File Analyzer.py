import re
import os
import pandas as pd
from datetime import datetime

def parse_log_file(filepath):
    """
    Parses a PyORBIT log file to extract MAP BIC and convergence status.

    Args:
        filepath (str): The full path to the log file.

    Returns:
        dict: A dictionary containing the model name, BIC, and convergence status.
              Returns None if the required lines are not found.
    """
    map_bic = None
    gelman_rubin_values = []
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
            # Extract MAP BIC
            bic_match = re.search(r'MAP BIC\s+\(using likelihood\)\s*=\s*(-?[\d\.]+)', content)
            if bic_match:
                map_bic = float(bic_match.group(1))
            
            # Extract Gelman-Rubin values
            gr_pattern = r'Gelman-Rubin:\s+\d+\s+([\d\.]+)\s+'
            gelman_rubin_values = [float(x) for x in re.findall(gr_pattern, content)]

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while reading {filepath}: {e}")
        return None

    if map_bic is not None:
        # Calculate convergence statistics
        if gelman_rubin_values:
            converged_count = sum(1 for gr in gelman_rubin_values if gr < 1.1)
            total_count = len(gelman_rubin_values)
            convergence_pct = (converged_count / total_count * 100) if total_count > 0 else 0
            max_gr = max(gelman_rubin_values)
        else:
            convergence_pct = 0
            max_gr = None
        
        # Extract model name details from the filename
        basename = os.path.basename(filepath)
        # Remove the common prefix and .log extension
        cleaned_name = basename.replace('configuration_file_emcee_run_', '').replace('.log', '')
        
        # Extract dataset (DS1, DS2, etc.) and the rest
        match = re.match(r'(DS\d+)_(\dp)_(.*)', cleaned_name)
        if match:
            dataset = match.group(1)
            num_planets = match.group(2)
            config_name = match.group(3)
        else:
            # Fallback for other naming patterns
            dataset = 'Unknown'
            num_planets = 'N/A'
            config_name = cleaned_name

        return {
            'Configuration': config_name,
            'Dataset': dataset,
            'Planets': num_planets,
            'MAP BIC': map_bic,
            'Convergence %': convergence_pct,
            'Max GR': max_gr,
            'File': basename
        }
    return None

def analyze_and_display(log_files):
    """
    Analyzes a list of log files and prints a formatted comparison table.
    Also exports results to CSV with highlighting for preferred models.

    Args:
        log_files (list): A list of paths to the log files.
    """
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
        print(f"\nWarning: {len(failed_files)} files could not be parsed (missing MAP BIC data):")
        for f in failed_files:
            print(f"  - {f}")
        print()

    if not all_data:
        print("No data could be extracted from any log files found.")
        return

    # Create a DataFrame and group by configuration and dataset
    df = pd.DataFrame(all_data)
    
    # Group by dataset first, then by configuration
    datasets = sorted(df['Dataset'].unique())
    
    # --- Display Results ---
    print("--- Model Comparison by Dataset and Configuration ---\n")

    # Prepare data for CSV export
    export_data = []
    
    for dataset in datasets:
        dataset_df = df[df['Dataset'] == dataset]
        grouped = dataset_df.groupby('Configuration')
        
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset}")
        print(f"{'='*80}\n")
        
        for config_name, group in grouped:
            print(f"--- Configuration: {config_name} ---\n")
            
            # Sort by number of planets
            group['Planets'] = pd.Categorical(group['Planets'], categories=['0p', '1p', '2p', '3p'], ordered=True)
            group = group.sort_values('Planets')
            
            # Find the index of the minimum BIC for highlighting
            min_bic_idx = group['MAP BIC'].idxmin()
            min_bic_value = group.loc[min_bic_idx, 'MAP BIC']
            
            # Create display dataframe
            display_group = group[['Planets', 'MAP BIC', 'Convergence %', 'Max GR']].copy()
            # Add ΔBIC column to display
            display_group['ΔBIC'] = group['MAP BIC'] - min_bic_value
            
            # Add preferred model indicators and ΔBIC calculation
            group_copy = group.copy()
            group_copy['Preferred_BIC'] = group_copy.index == min_bic_idx
            group_copy['ΔBIC'] = group_copy['MAP BIC'] - min_bic_value
            group_copy['Dataset'] = dataset
            group_copy['Configuration'] = config_name
            
            # Reorder columns for export
            export_group = group_copy[['Dataset', 'Configuration', 'Planets', 'MAP BIC', 'ΔBIC',
                                      'Convergence %', 'Max GR', 'Preferred_BIC', 'File']]
            export_data.append(export_group)
            
            # Print simple table
            print(display_group.to_string(index=False))
            print(f"\nBest BIC: {group.loc[min_bic_idx, 'Planets']} model (BIC = {group.loc[min_bic_idx, 'MAP BIC']:.2f})")
            print("\n" + "-"*80 + "\n")

    # Export to CSV and HTML
    if export_data:
        combined_df = pd.concat(export_data, ignore_index=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export CSV for data analysis with UTF-8 encoding
        csv_filename = f"model_comparison_{timestamp}.csv"
        combined_df.to_csv(csv_filename, index=False, encoding='utf-8')
        
        # Export HTML with formatting for visual display
        html_filename = f"model_comparison_{timestamp}.html"
        
        # Create HTML content with modern, simple color scheme
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Model Comparison Results</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #fafafa;
            color: #2c3e50;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            font-weight: 600;
            margin-bottom: 30px;
            font-size: 32px;
        }
        .dataset-section {
            background-color: white;
            margin: 30px 0;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            overflow: hidden;
        }
        .dataset-header {
            background-color: #2c3e50;
            color: white;
            padding: 20px 30px;
            font-size: 22px;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        .config-section {
            margin: 0;
            padding: 25px 30px;
            border-bottom: 1px solid #ecf0f1;
        }
        .config-section:last-child {
            border-bottom: none;
        }
        h2 { 
            color: #34495e;
            margin: 0 0 15px 0;
            font-size: 18px;
            font-weight: 600;
        }
        table { 
            border-collapse: collapse; 
            margin: 15px 0;
            width: 100%;
            background-color: white;
        }
        th, td { 
            padding: 12px 16px; 
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }
        th { 
            background-color: #f8f9fa;
            color: #2c3e50;
            font-weight: 600;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        td {
            color: #34495e;
            font-size: 15px;
        }
        tr:last-child td {
            border-bottom: none;
        }
        tr:hover td {
            background-color: #f8f9fa;
        }
        .highlight-bic {
            background-color: #e8f5e9 !important;
            font-weight: 600;
            color: #2e7d32;
        }
        .convergence-good {
            color: #2e7d32;
            font-weight: 600;
        }
        .convergence-warning {
            color: #f57c00;
            font-weight: 600;
        }
        .convergence-bad {
            color: #c62828;
            font-weight: 600;
        }
        .summary {
            margin: 15px 0 0 0;
            padding: 15px 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            font-size: 15px;
            color: #34495e;
        }
        .summary strong {
            color: #2c3e50;
        }
        .legend {
            margin: 0 0 30px 0;
            padding: 20px 25px;
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .legend h3 {
            margin: 0 0 15px 0;
            color: #2c3e50;
            font-size: 18px;
            font-weight: 600;
        }
        .legend-item {
            margin: 8px 0;
            color: #34495e;
            font-size: 14px;
            line-height: 1.6;
        }
        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 13px;
            font-weight: 600;
        }
        .badge-green {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>PyORBIT Model Comparison Results</h1>
        
        <div class="legend">
            <h3>Legend</h3>
            <div class="legend-item"><span class="badge badge-green">Green highlight</span> = Best BIC in configuration</div>
            <div class="legend-item"><strong>Convergence %:</strong> Percentage of parameters with Gelman-Rubin < 1.1</div>
            <div class="legend-item"><strong>Max GR:</strong> Maximum Gelman-Rubin value across all parameters</div>
            <div class="legend-item">
                <span class="convergence-good">Green</span> = 100% converged | 
                <span class="convergence-warning">Orange</span> = 90-99% converged | 
                <span class="convergence-bad">Red</span> = <90% converged
            </div>
        </div>
"""
        
        # Process each dataset
        for dataset in datasets:
            html_content += f'        <div class="dataset-section">\n'
            html_content += f'            <div class="dataset-header">{dataset}</div>\n'
            
            dataset_df = df[df['Dataset'] == dataset]
            grouped = dataset_df.groupby('Configuration')
            
            for config_name, group in grouped:
                html_content += f'            <div class="config-section">\n'
                html_content += f'                <h2>Configuration: {config_name}</h2>\n'
                
                # Sort by number of planets
                group['Planets'] = pd.Categorical(group['Planets'], categories=['0p', '1p', '2p', '3p'], ordered=True)
                group = group.sort_values('Planets')
                
                min_bic_idx = group['MAP BIC'].idxmin()
                
                # Build table
                html_content += '                <table>\n'
                html_content += '                    <tr><th>Planets</th><th>MAP BIC</th><th>ΔBIC</th><th>Convergence %</th><th>Max GR</th></tr>\n'
                
                for idx, row in group.iterrows():
                    bic_class = 'highlight-bic' if idx == min_bic_idx else ''
                    
                    # Determine convergence color
                    conv_pct = row['Convergence %']
                    if conv_pct == 100:
                        conv_class = 'convergence-good'
                    elif conv_pct >= 90:
                        conv_class = 'convergence-warning'
                    else:
                        conv_class = 'convergence-bad'
                    
                    max_gr_str = f"{row['Max GR']:.4f}" if row['Max GR'] is not None else "N/A"
                    
                    # Calculate ΔBIC for this row
                    delta_bic = row['MAP BIC'] - group.loc[min_bic_idx, 'MAP BIC']
                    
                    html_content += f'                    <tr>\n'
                    html_content += f'                        <td>{row["Planets"]}</td>\n'
                    html_content += f'                        <td class="{bic_class}">{row["MAP BIC"]:.2f}</td>\n'
                    html_content += f'                        <td class="{bic_class}">{delta_bic:.2f}</td>\n'
                    html_content += f'                        <td class="{conv_class}">{conv_pct:.1f}%</td>\n'
                    html_content += f'                        <td>{max_gr_str}</td>\n'
                    html_content += f'                    </tr>\n'
                
                html_content += '                </table>\n'
                
                # Add summary
                html_content += f'                <div class="summary">\n'
                html_content += f'                    <strong>Best BIC:</strong> {group.loc[min_bic_idx, "Planets"]} model (BIC = {group.loc[min_bic_idx, "MAP BIC"]:.2f})\n'
                html_content += f'                </div>\n'
                html_content += f'            </div>\n'
            
            html_content += '        </div>\n'
        
        html_content += """
    </div>
</body>
</html>
"""
        
        # Write HTML file with UTF-8 encoding
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n{'='*80}")
        print(f"Results exported to:")
        print(f"  - CSV: {csv_filename} (for data analysis)")
        print(f"  - HTML: {html_filename} (for visual display with formatting)")
        print("\nNotes:")
        print("  - Preferred_BIC=True indicates the model with lowest BIC in that configuration")
        print("  - Convergence % shows percentage of parameters with Gelman-Rubin < 1.1")
        print("  - Open the HTML file in a web browser to see the formatted results")
        print(f"{'='*80}\n")

# --- Main Execution ---
if __name__ == "__main__":
    import sys
    import glob

    # Allow specifying a directory as command line argument
    if len(sys.argv) > 1:
        search_directory = sys.argv[1]
        if not os.path.exists(search_directory):
            print(f"Error: Directory '{search_directory}' does not exist.")
            sys.exit(1)
        if not os.path.isdir(search_directory):
            print(f"Error: '{search_directory}' is not a directory.")
            sys.exit(1)
        print(f"Searching for .log files in: {search_directory}")
    else:
        search_directory = "."
        print("Searching for .log files in current directory and subdirectories...")

    # Find all .log files in the specified directory and subdirectories
    all_log_files = glob.glob(os.path.join(search_directory, '**/*.log'), recursive=True)
    
    # More robust duplicate detection: prefer files that are not in deeply nested subdirectories
    # Group files by basename and select the best one from each group
    file_groups = {}
    
    for log_file in all_log_files:
        basename = os.path.basename(log_file)
        if basename not in file_groups:
            file_groups[basename] = []
        file_groups[basename].append(log_file)
    
    # Select the best file from each group (prefer less nested files)
    log_files = []
    for basename, files in file_groups.items():
        if len(files) == 1:
            # No duplicates, use the single file
            log_files.append(files[0])
        else:
            # Multiple files with same basename - choose the one with least nesting
            # Calculate depth for each file and choose the shallowest
            best_file = min(files, key=lambda f: f.count(os.sep))
            log_files.append(best_file)
    
    if not log_files:
        print("No .log files found in the current directory or subdirectories.")
    else:
        print(f"Found {len(log_files)} unique log files to analyze (filtered from {len(all_log_files)} total).\n")
        analyze_and_display(log_files)
