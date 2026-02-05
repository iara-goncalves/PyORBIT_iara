import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import traceback

# Create output directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Load the results
with open('HD189567/dynesty/results.p', 'rb') as f:
    results = pickle.load(f)

# Load the model container
with open('HD189567/dynesty/model_container.p', 'rb') as f:
    mc = pickle.load(f)

# Extract samples
samples = results.samples
print(f"Samples shape: {samples.shape}")

# Calculate weights from logwt
logwt = results.logwt
weights = np.exp(logwt - logwt.max())
weights /= weights.sum()

# Get the maximum likelihood sample
max_idx = np.argmax(results.logl)
theta_max = results.samples[max_idx]

# Get the weighted median parameters
theta_median = np.zeros(samples.shape[1])
for i in range(samples.shape[1]):
    # Calculate weighted median
    sorted_idx = np.argsort(samples[:, i])
    sorted_weights = weights[sorted_idx]
    cumulative_weights = np.cumsum(sorted_weights)
    cumulative_weights /= cumulative_weights[-1]  # Normalize
    theta_median[i] = np.interp(0.5, cumulative_weights, samples[sorted_idx, i])

# Print model container information
print("\nModel container information:")
print(f"Type: {type(mc)}")
print(f"Attributes: {dir(mc)}")

# Check bounds type
if hasattr(mc, 'bounds'):
    print(f"\nBounds type: {type(mc.bounds)}")
    if isinstance(mc.bounds, np.ndarray):
        print(f"Bounds shape: {mc.bounds.shape}")
        if len(mc.bounds.shape) == 2:
            print(f"Bounds: {mc.bounds}")

# Try different ways to get parameter names
param_names = []

# Option 1: Try mc.pam_names
if hasattr(mc, 'pam_names') and mc.pam_names:
    param_names = mc.pam_names
    print(f"Found {len(param_names)} parameters from mc.pam_names")

# Option 2: Try mc.variable_list
if not param_names and hasattr(mc, 'variable_list') and mc.variable_list:
    param_names = mc.variable_list
    print(f"Found {len(param_names)} parameters from mc.variable_list")

# Option 3: Try mc.variables
if not param_names and hasattr(mc, 'variables'):
    if hasattr(mc.variables, 'keys'):
        param_names = list(mc.variables.keys())
        print(f"Found {len(param_names)} parameters from mc.variables.keys()")

# Option 4: Check if there's a planets attribute and extract names
if not param_names and hasattr(mc, 'planets'):
    try:
        planet_params = []
        for planet in mc.planets:
            planet_params.extend([f"{planet}_P", f"{planet}_K", f"{planet}_e", f"{planet}_omega", f"{planet}_M0"])
        param_names = planet_params
        print(f"Created {len(param_names)} parameters from planet names")
    except:
        pass

# If all else fails, create generic names
if not param_names:
    param_names = [f"param_{i}" for i in range(samples.shape[1])]
    print(f"Created {len(param_names)} generic parameter names")

print("\nParameter names:", param_names)

print("\nMaximum likelihood parameters:")
for name, value in zip(param_names, theta_max):
    print(f"  {name}: {value}")

print("\nMedian parameters:")
for name, value in zip(param_names, theta_median):
    print(f"  {name}: {value}")

# Convert array to dictionary for PyORBIT
theta_dict = {name: value for name, value in zip(param_names, theta_median)}

# Get the dataset
dataset_names = list(mc.dataset_dict.keys())
print(f"\nAvailable datasets: {dataset_names}")

# Print information about the models
if hasattr(mc, 'models'):
    print("\nModels information:")
    print(f"Models type: {type(mc.models)}")
    print(f"Models keys: {list(mc.models.keys())}")

# Explore the dataset structure
print("\nExploring dataset structure:")
for dataset_name in dataset_names:
    print(f"\nDataset: {dataset_name}")
    dataset = mc.dataset_dict[dataset_name]
    print(f"  Type: {type(dataset)}")
    
    # Check if the dataset has the expected attributes
    has_x = hasattr(dataset, 'x')
    has_y = hasattr(dataset, 'y')
    has_e = hasattr(dataset, 'e')
    print(f"  Has x: {has_x}, Has y: {has_y}, Has e: {has_e}")
    
    if has_x and has_y and has_e:
        print(f"  x shape: {dataset.x.shape}")
        print(f"  y shape: {dataset.y.shape}")
        print(f"  e shape: {dataset.e.shape}")
        print(f"  x range: {np.min(dataset.x)} to {np.max(dataset.x)}")
        print(f"  y range: {np.min(dataset.y)} to {np.max(dataset.y)}")

# Create a simple plot of just the data
for dataset_name in dataset_names:
    try:
        # Get the observed data
        dataset = mc.dataset_dict[dataset_name]
        x_data = dataset.x
        y_data = dataset.y
        e_data = dataset.e
        
        # Create a simple plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(x_data, y_data, yerr=e_data, fmt='o', alpha=0.7)
        plt.xlabel('Time (days)')
        plt.ylabel('RV (m/s)')
        plt.title(f'HD189567')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'plots/HD189567_{dataset_name}_data.png', dpi=300)
        print(f"Data plot saved as plots/HD189567_{dataset_name}_data.png")
        
    except Exception as e:
        print(f"Error creating data plot for {dataset_name}: {str(e)}")
        traceback.print_exc()

# Try to create model plots for each dataset
for dataset_name in dataset_names:
    try:
        print(f"\nProcessing dataset for model: {dataset_name}")
        
        # Get the observed data
        dataset = mc.dataset_dict[dataset_name]
        x_data = dataset.x
        y_data = dataset.y
        e_data = dataset.e
        
        print(f"Data loaded: {len(x_data)} points")
        
        # Find the correct model key
        model_key = None
        for key in mc.models.keys():
            if dataset_name in key or key in dataset_name:
                model_key = key
                break
        
        if model_key is None:
            print(f"Could not find model for {dataset_name}. Available models: {list(mc.models.keys())}")
            continue
        
        print(f"Using model key: {model_key}")
        
        # Initialize the model
        model = mc.models[model_key]
        print(f"Model type: {type(model)}")
        print(f"Model attributes: {dir(model)}")
        
        # Check if model has the model method
        if not hasattr(model, 'model'):
            print(f"Model does not have 'model' method. Available methods: {[m for m in dir(model) if not m.startswith('_')]}")
            continue
        
        # Generate model using median parameters
        model_x = np.linspace(np.min(x_data), np.max(x_data), 1000)
        
        # Evaluate the model
        print("Evaluating model...")
        try:
            # Try calling model with dictionary
            y_model = model.model(theta_dict, x_data)
            y_curve = model.model(theta_dict, model_x)
        except Exception as e:
            print(f"Error with dictionary parameters: {str(e)}")
            # Try calling model with array
            try:
                y_model = model.model(theta_median, x_data)
                y_curve = model.model(theta_median, model_x)
            except Exception as e2:
                print(f"Error with array parameters: {str(e2)}")
                raise
        
        # Create the plot
        fig = plt.figure(figsize=(10, 6))
        gs = GridSpec(2, 1, height_ratios=[3, 1])
        
        # Top panel: data and model
        ax1 = plt.subplot(gs[0])
        ax1.errorbar(x_data, y_data, yerr=e_data, fmt='o', alpha=0.5, label='Data')
        ax1.plot(model_x, y_curve, 'r-', alpha=0.8, label='Model')
        ax1.set_ylabel('RV (m/s)')
        ax1.set_title(f'HD189567 {dataset_name} Model')
        ax1.legend()
        
        # Bottom panel: residuals
        ax2 = plt.subplot(gs[1])
        ax2.errorbar(x_data, y_data - y_model, yerr=e_data, fmt='o', alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.8)
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Residuals (m/s)')
        
        plt.tight_layout()
        plt.savefig(f'plots/HD189567_{dataset_name}_model.png', dpi=300)
        print(f"RV model plot saved as plots/HD189567_{dataset_name}_model.png")
        
    except Exception as e:
        print(f"Error creating plot for {dataset_name}: {str(e)}")
        traceback.print_exc()

# Function to phase fold data
def phase_fold(time, period, tc):
    phase = ((time - tc) / period) % 1.0
    return phase

# Create phase-folded plots for each planet
for planet in ['b', 'c', 'd']:
    try:
        # Get period and Tc from the results using parameter indices
        if planet == 'b':
            P = 2**theta_median[0]  # Convert from log2 to linear space
            Tc = theta_median[4]  # Tc for planet b
        elif planet == 'c':
            P = 2**theta_median[5]  # Convert from log2 to linear space
            Tc = theta_median[9]  # Tc for planet c
        else:  # planet d
            P = 2**theta_median[10]  # Convert from log2 to linear space
            Tc = theta_median[14]  # Tc for planet d

        # Get the correct model key for this planet
        model_key = f'radial_velocities_{planet}'
        
        # Check if model exists
        if model_key not in mc.models:
            print(f"Model {model_key} not found. Available models: {list(mc.models.keys())}")
            continue
            
        # Get the model
        model = mc.models[model_key]
        
        # Print model info for debugging
        print(f"Model type: {type(model)}")
        print(f"Model methods: {[m for m in dir(model) if not m.startswith('_')]}")
        
        # Create figure for phase-folded plot
        plt.figure(figsize=(8, 12))
        gs = GridSpec(2, 1, height_ratios=[3, 1])

        # Top panel: phase-folded data and model
        ax1 = plt.subplot(gs[0])
        
        # Colors for different datasets
        colors = {'HD189567_HARPS': 'blue', 'HD189567_ESPRESSO': 'red'}
        markers = {'HD189567_HARPS_offset0': 'o', 'HD189567_HARPS_offset1': 's', 'HD189567_ESPRESSO': '^'}
        
        # Get parameter indices from the log file
        HARPS_offset0_idx = 16  # ID as per the log file
        HARPS_offset1_idx = 17  # ID as per the log file
        ESPRESSO_offset_idx = 19  # ID as per the log file
        
        # Get the offset values from the parameter array
        HARPS_offset_0 = theta_median[HARPS_offset0_idx]
        HARPS_offset_1 = theta_median[HARPS_offset1_idx]
        ESPRESSO_offset = theta_median[ESPRESSO_offset_idx]
        
        print(f"Offset values from parameters:")
        print(f"  HARPS offset_0 (param_{HARPS_offset0_idx}): {HARPS_offset_0}")
        print(f"  HARPS offset_1 (param_{HARPS_offset1_idx}): {HARPS_offset_1}")
        print(f"  ESPRESSO offset (param_{ESPRESSO_offset_idx}): {ESPRESSO_offset}")
        
        # Store all data points for plotting
        all_phases = []
        all_rvs = []
        all_errs = []
        all_colors = []
        all_markers = []
        
        # Create phased data for each dataset
        for dataset_name in dataset_names:
            dataset = mc.dataset_dict[dataset_name]
            x_data = dataset.x
            y_data = dataset.y
            e_data = dataset.e
            
            # Check if this dataset has offset flags
            has_offset_flags = False
            offset_flags = None
            if hasattr(dataset, 'offset_flag'):
                has_offset_flags = True
                offset_flags = dataset.offset_flag
                print(f"Dataset {dataset_name} has offset flags: {np.unique(offset_flags)}")
            
            # For each data point, apply the appropriate offset
            y_adjusted = np.copy(y_data)
            
            if dataset_name == "HD189567_HARPS" and has_offset_flags:
                # Apply the offsets based on the flag
                for i in range(len(y_adjusted)):
                    if offset_flags[i] == 0:
                        y_adjusted[i] -= HARPS_offset_0
                    else:
                        y_adjusted[i] -= HARPS_offset_1
                        
                # Phase fold the data
                phases = phase_fold(x_data, P, Tc)
                
                # Plot separately for each offset flag
                for flag in np.unique(offset_flags):
                    mask = offset_flags == flag
                    marker = markers.get(f"{dataset_name}_offset{flag}", 'o')
                    label = f"{dataset_name} (offset {flag})"
                    
                    ax1.errorbar(phases[mask], y_adjusted[mask], yerr=e_data[mask], 
                                 fmt=marker, alpha=0.7, color=colors.get(dataset_name, 'green'),
                                 label=label)
                    
                    # Plot points at phase+1 for continuity
                    ax1.errorbar(phases[mask] + 1, y_adjusted[mask], yerr=e_data[mask], 
                                 fmt=marker, alpha=0.4, color=colors.get(dataset_name, 'green'))
                    
                    # Store for later use
                    all_phases.extend(phases[mask])
                    all_rvs.extend(y_adjusted[mask])
                    all_errs.extend(e_data[mask])
                    all_colors.extend([colors.get(dataset_name, 'green')] * np.sum(mask))
                    all_markers.extend([marker] * np.sum(mask))
            else:
                # For ESPRESSO dataset
                offset = ESPRESSO_offset if dataset_name == "HD189567_ESPRESSO" else 0.0
                
                print(f"{dataset_name} offset: {offset}")
                
                # Remove the offset from the data
                y_adjusted = y_data - offset
                
                # Phase fold the data
                phases = phase_fold(x_data, P, Tc)
                
                # Plot with dataset-specific color and marker
                marker = markers.get(dataset_name, 'o')
                ax1.errorbar(phases, y_adjusted, yerr=e_data, fmt=marker, alpha=0.7, 
                            label=f'{dataset_name}', color=colors.get(dataset_name, 'green'))
                
                # Plot points at phase+1 for continuity
                ax1.errorbar(phases + 1, y_adjusted, yerr=e_data, fmt=marker, alpha=0.4, 
                            color=colors.get(dataset_name, 'green'))
                
                # Store for later use
                all_phases.extend(phases)
                all_rvs.extend(y_adjusted)
                all_errs.extend(e_data)
                all_colors.extend([colors.get(dataset_name, 'green')] * len(phases))
                all_markers.extend([marker] * len(phases))

        # Generate a simple sine curve for the model using the period and semi-amplitude
        if planet == 'b':
            K = 2**theta_median[1]  # Convert from log2 to linear space
        elif planet == 'c':
            K = 2**theta_median[6]  # Convert from log2 to linear space
        else:  # planet d
            K = 2**theta_median[11]  # Convert from log2 to linear space
        
        # For eccentric orbits, we should calculate e and omega
        e = 0.0
        omega = 0.0
        
        if planet == "b":
            # For planet b, compute e and omega from sre_coso and sre_sino
            coso_idx = 2  # ID as per the log file
            sino_idx = 3  # ID as per the log file
            sre_coso = theta_median[coso_idx]
            sre_sino = theta_median[sino_idx]
            e = sre_coso**2 + sre_sino**2
            if e > 0:
                omega = np.degrees(np.arctan2(sre_sino, sre_coso))
        elif planet == "c":
            # For planet c, get e and omega directly
            e_idx = 7  # ID as per the log file
            omega_idx = 8  # ID as per the log file
            e = theta_median[e_idx]
            omega = theta_median[omega_idx]
        else:  # planet d
            # For planet d, compute e and omega from sre_coso and sre_sino
            coso_idx = 12  # ID as per the log file
            sino_idx = 13  # ID as per the log file
            sre_coso = theta_median[coso_idx]
            sre_sino = theta_median[sino_idx]
            e = sre_coso**2 + sre_sino**2
            if e > 0:
                omega = np.degrees(np.arctan2(sre_sino, sre_coso))
        
        print(f"Planet {planet}: P={P}, K={K}, e={e}, omega={omega}")
        
        # Generate the RV model curve (simple Keplerian)
        phase_points = np.linspace(0, 1, 1000)
        rv_curve = np.zeros(len(phase_points))
        
        for i, phase in enumerate(phase_points):
            # Convert phase to mean anomaly
            M = 2 * np.pi * phase
            
            # Solve Kepler's equation for eccentric anomaly
            E = M
            if e > 0:
                for _ in range(10):  # Iterative solution
                    E = M + e * np.sin(E)
            
            # Calculate true anomaly
            if e < 1.0:
                true_anomaly = 2 * np.arctan2(np.sqrt(1+e) * np.sin(E/2), np.sqrt(1-e) * np.cos(E/2))
            else:
                true_anomaly = 0
            
            # Calculate RV
            omega_rad = np.radians(omega)
            rv_curve[i] = K * (np.cos(true_anomaly + omega_rad) + e * np.cos(omega_rad))
        
        # Plot the model curve
        ax1.plot(phase_points, rv_curve, 'k-', linewidth=2, alpha=0.8, label='Model')
        ax1.plot(phase_points + 1, rv_curve, 'k-', linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Phase')
        ax1.set_ylabel('RV (m/s)')
        ax1.set_title(f'HD189567 Planet {planet} Phase-folded RV')
        ax1.set_xlim(0, 2)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom panel: residuals
        ax2 = plt.subplot(gs[1])
        
        # Calculate residuals
        model_phases = np.array(all_phases)
        model_rvs = np.zeros(len(model_phases))
        
        for i, phase in enumerate(model_phases):
            # Convert phase to mean anomaly
            M = 2 * np.pi * phase
            
            # Solve Kepler's equation for eccentric anomaly
            E = M
            if e > 0:
                for _ in range(10):  # Iterative solution
                    E = M + e * np.sin(E)
            
            # Calculate true anomaly
            if e < 1.0:
                true_anomaly = 2 * np.arctan2(np.sqrt(1+e) * np.sin(E/2), np.sqrt(1-e) * np.cos(E/2))
            else:
                true_anomaly = 0
            
            # Calculate RV
            omega_rad = np.radians(omega)
            model_rvs[i] = K * (np.cos(true_anomaly + omega_rad) + e * np.cos(omega_rad))
        
        residuals = np.array(all_rvs) - model_rvs
        
        # Plot residuals
        for i, phase in enumerate(model_phases):
            ax2.errorbar(phase, residuals[i], yerr=all_errs[i], fmt=all_markers[i], 
                         alpha=0.7, color=all_colors[i])
            # Plot at phase+1 for continuity
            ax2.errorbar(phase + 1, residuals[i], yerr=all_errs[i], fmt=all_markers[i], 
                         alpha=0.4, color=all_colors[i])
        
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.8)
        ax2.set_xlabel('Phase')
        ax2.set_ylabel('Residuals (m/s)')
        ax2.set_xlim(0, 2)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'plots/HD189567_{planet}_phase_folded.png', dpi=300)
        print(f"Phase-folded plot saved as plots/HD189567_{planet}_phase_folded.png")
        
    except Exception as e:
        print(f"Error creating phase-folded plot for planet {planet}: {str(e)}")
        traceback.print_exc()

# Add code to plot GLS periodogram of offset-corrected data
print("\nCalculating GLS periodogram of the offset-corrected data...")

try:
    # Import necessary packages
    from astropy.timeseries import LombScargle
    
    # Get parameter indices from the log file
    HARPS_offset0_idx = 16  # ID as per the log file
    HARPS_offset1_idx = 17  # ID as per the log file
    ESPRESSO_offset_idx = 19  # ID as per the log file
    
    # Get the offset values from the parameter array
    HARPS_offset_0 = theta_median[HARPS_offset0_idx]
    HARPS_offset_1 = theta_median[HARPS_offset1_idx]
    ESPRESSO_offset = theta_median[ESPRESSO_offset_idx]
    
    print(f"Using offset values:")
    print(f"  HARPS offset_0: {HARPS_offset_0}")
    print(f"  HARPS offset_1: {HARPS_offset_1}")
    print(f"  ESPRESSO offset: {ESPRESSO_offset}")
    
    # Combine all data with offsets corrected
    all_times = []
    all_rvs = []
    all_rv_errs = []
    
    # Process each dataset to remove offsets
    for dataset_name in dataset_names:
        dataset = mc.dataset_dict[dataset_name]
        x_data = dataset.x
        y_data = dataset.y
        e_data = dataset.e
        
        # Apply the appropriate offsets
        y_adjusted = np.copy(y_data)
        
        if dataset_name == "HD189567_HARPS" and hasattr(dataset, 'offset_flag'):
            offset_flags = dataset.offset_flag
            for i in range(len(y_data)):
                if offset_flags[i] == 0:
                    y_adjusted[i] -= HARPS_offset_0
                else:
                    y_adjusted[i] -= HARPS_offset_1
        elif dataset_name == "HD189567_ESPRESSO":
            y_adjusted -= ESPRESSO_offset
        
        # Add to the combined data arrays
        all_times.extend(x_data)
        all_rvs.extend(y_adjusted)
        all_rv_errs.extend(e_data)
    
    # Convert to numpy arrays
    all_times = np.array(all_times)
    all_rvs = np.array(all_rvs)
    all_rv_errs = np.array(all_rv_errs)
    
    # Calculate the GLS periodogram
    # First for the raw offset-corrected data
    # Use frequency grid from 1/10000 days to 1 day
    min_freq = 1/10000  # cycles per day
    max_freq = 1.0      # cycles per day
    frequency = np.linspace(min_freq, max_freq, 10000)
    
    # Calculate the periodogram
    ls = LombScargle(all_times, all_rvs, all_rv_errs)
    power = ls.power(frequency)
    
    # Convert to period for easier interpretation
    period = 1/frequency
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(period, power, 'k-')
    
    # Add vertical lines for the detected planet periods
    planet_periods = [14.28, 33.59, 662.77]  # From the model output
    planet_labels = ['Planet b', 'Planet c', 'Planet d']
    colors = ['r', 'g', 'b']
    
    for i, (p, label, color) in enumerate(zip(planet_periods, planet_labels, colors)):
        plt.axvline(x=p, color=color, linestyle='--', label=label)
    
    # Add some common periods of interest
    plt.axvline(x=365.25, color='gray', linestyle=':', alpha=0.7, label='1 year')
    
    # Set plot properties
    plt.xlabel('Period (days)')
    plt.ylabel('Power')
    plt.title('GLS Periodogram of Offset-Corrected RV Data')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig('plots/HD189567_GLS_periodogram_raw.png', dpi=300)
    print("GLS periodogram saved as plots/HD189567_GLS_periodogram_raw.png")
    
    # Now calculate periodograms after removing each planet's contribution
    # This will help visualize the remaining signals
    
    planet_signals = {
        'b': {'P': 14.28, 'K': 2.19, 'e': 0.072, 'omega': 61, 'mean_long': 163.8},
        'c': {'P': 33.59, 'K': 2.33, 'e': 0.368, 'omega': 79.8, 'mean_long': 101.7},
        'd': {'P': 662.77, 'K': 2.54, 'e': 0.811, 'omega': -33.2, 'mean_long': 357.2}
    }
    
    # Create a function to calculate Keplerian RV model
    def keplerian_rv(time, P, K, e, omega, mean_long, T_ref):
        # Convert to radians
        omega_rad = np.radians(omega)
        
        # Calculate mean anomaly
        M0 = np.radians(mean_long - omega)
        n = 2 * np.pi / P  # mean motion
        M = (M0 + n * (time - T_ref)) % (2 * np.pi)
        
        # Solve Kepler's equation for eccentric anomaly
        E = np.zeros_like(M)
        for i in range(len(M)):
            E_i = M[i]
            for _ in range(10):  # Iterative solution
                E_i = M[i] + e * np.sin(E_i)
            E[i] = E_i
        
        # Calculate true anomaly
        nu = 2 * np.arctan2(np.sqrt(1+e) * np.sin(E/2), np.sqrt(1-e) * np.cos(E/2))
        
        # Calculate radial velocity
        rv = K * (np.cos(nu + omega_rad) + e * np.cos(omega_rad))
        
        return rv
    
    # Reference time from the model
    T_ref = 2452937.54737118
    
    # Calculate the residuals after removing each planet
    residuals = {
        'after_b': np.copy(all_rvs),
        'after_bc': np.copy(all_rvs),
        'after_bcd': np.copy(all_rvs)
    }
    
    # Remove planet b
    rv_b = keplerian_rv(all_times, **planet_signals['b'], T_ref=T_ref)
    residuals['after_b'] -= rv_b
    
    # Remove planets b and c
    rv_c = keplerian_rv(all_times, **planet_signals['c'], T_ref=T_ref)
    residuals['after_bc'] = residuals['after_b'] - rv_c
    
    # Remove planets b, c, and d
    rv_d = keplerian_rv(all_times, **planet_signals['d'], T_ref=T_ref)
    residuals['after_bcd'] = residuals['after_bc'] - rv_d
    
    # Create periodograms for each residual set
    plt.figure(figsize=(12, 10))
    
    # Subplot 1: Raw data
    plt.subplot(4, 1, 1)
    ls_raw = LombScargle(all_times, all_rvs, all_rv_errs)
    power_raw = ls_raw.power(frequency)
    plt.semilogx(period, power_raw, 'k-')
    for i, (p, label, color) in enumerate(zip(planet_periods, planet_labels, colors)):
        plt.axvline(x=p, color=color, linestyle='--')
    plt.axvline(x=365.25, color='gray', linestyle=':', alpha=0.7)
    plt.ylabel('Power')
    plt.title('Raw Data')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: After removing planet b
    plt.subplot(4, 1, 2)
    ls_b = LombScargle(all_times, residuals['after_b'], all_rv_errs)
    power_b = ls_b.power(frequency)
    plt.semilogx(period, power_b, 'k-')
    for i, (p, label, color) in enumerate(zip(planet_periods[1:], planet_labels[1:], colors[1:])):
        plt.axvline(x=p, color=color, linestyle='--')
    plt.axvline(x=365.25, color='gray', linestyle=':', alpha=0.7)
    plt.ylabel('Power')
    plt.title('After Removing Planet b')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: After removing planets b and c
    plt.subplot(4, 1, 3)
    ls_bc = LombScargle(all_times, residuals['after_bc'], all_rv_errs)
    power_bc = ls_bc.power(frequency)
    plt.semilogx(period, power_bc, 'k-')
    plt.axvline(x=planet_periods[2], color=colors[2], linestyle='--')
    plt.axvline(x=365.25, color='gray', linestyle=':', alpha=0.7)
    plt.ylabel('Power')
    plt.title('After Removing Planets b and c')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: After removing all planets
    plt.subplot(4, 1, 4)
    ls_bcd = LombScargle(all_times, residuals['after_bcd'], all_rv_errs)
    power_bcd = ls_bcd.power(frequency)
    plt.semilogx(period, power_bcd, 'k-')
    plt.axvline(x=365.25, color='gray', linestyle=':', alpha=0.7, label='1 year')
    plt.xlabel('Period (days)')
    plt.ylabel('Power')
    plt.title('After Removing All Planets')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/HD189567_GLS_periodogram_residuals.png', dpi=300)
    print("GLS residual periodograms saved as plots/HD189567_GLS_periodogram_residuals.png")
    
except Exception as e:
    print(f"Error creating GLS periodogram: {str(e)}")
    traceback.print_exc()
