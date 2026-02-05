#!/usr/bin/env python3
"""
Phase-folded RV plots for PyORBIT runs (HD102365)

- Produces ONLY the phase-folded RV plots (with residuals) for all planets found in the run dictionaries
- Loops over the 18 configurations you listed (emcee + dynesty)
- Works as a .py script (no notebook magics)
- Saves outputs to:
  /work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/post_analysis/figures_pyorbit_HD102365/{emcee,dynesty}/<run_name>/

Minimal changes in plotting logic relative to your original "PLOT 1" section:
- Same columns, same error definition, same styling and layout
- Only generalized to loop over multiple runs and auto-detect datasets/planets
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import glob
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import collections
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator


# ============================================================================
# USER SETTINGS
# ============================================================================
OUTPUT_ROOT = "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/post_analysis/figures_pyorbit_HD102365"

# Plotting parameters (kept from original)
font_label = 12
dot_size = 18
figsize = (10, 7)

# If you want to force only b (+ optional c), set these.
# If False, the script plots all planets present in summary_percentiles_parameters (b,c,d,...)
FORCE_ONLY_B_AND_C = False
ENABLE_PLANET_C = True  # used only if FORCE_ONLY_B_AND_C=True

# Phase-folded x-limits (same as original)
DEFAULT_FOLDED_X_LIMS = (-0.25, 1.25)

# ----------------------------------------------------------------------------
# The 18 configurations (directories from your tables)
# IMPORTANT: these are the *outer* directories you listed; the script will
# resolve the actual run root that contains emcee_plot/ or dynesty_plot/.
# ----------------------------------------------------------------------------

EMCEE_RUN_DIRS = [
    "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365/all_instr/gp/2p/HD102365_all_instr_gp_2p_emcee",
    "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365/all_instr/no_gp/3p/HD102365_all_instr_no_gp_3p_emcee",
    "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365/espresso_only/gp/1p/HD102365_espresso_only_gp_1p_emcee",
    "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365/espresso_only/no_gp/2p/HD102365_espresso_only_no_gp_2p_emcee",
    "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365/no_espresso/gp/1p/HD102365_no_espresso_gp_1p_emcee",
    "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365/no_espresso/no_gp/3p/HD102365_no_espresso_no_gp_3p_emcee",
    "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_UCLES_HD102365/gp/1p/HD102365_UCLES_only_gp_1p_emcee",
    "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_UCLES_HD102365/no_gp/1p/HD102365_UCLES_only_no_gp_1p_emcee",
    "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365_0p_all/HD102365_all_instr_ESPRESSO_GP_0p_emcee",
]

DYNASTY_RUN_DIRS = [
    "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365_0p_all/HD102365_all_instr_gp_0p_dynesty",
    "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365/all_instr/no_gp/2p/HD102365_all_instr_no_gp_2p_dynesty",
    "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365/espresso_only/gp/2p/HD102365_espresso_only_gp_2p_dynesty",
    "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365/espresso_only/no_gp/3p/HD102365_espresso_only_no_gp_3p_dynesty",
    "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365/no_espresso/gp/1p/HD102365_no_espresso_gp_1p_dynesty",
    "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365/no_espresso/no_gp/2p/HD102365_no_espresso_no_gp_2p_dynesty",
    "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_UCLES_HD102365/gp/2p/HD102365_UCLES_only_gp_2p_dynesty",
    "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_UCLES_HD102365/no_gp/2p/HD102365_UCLES_only_no_gp_2p_dynesty",
    "/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365_0p_all/HD102365_all_instr_ESPRESSO_GP_0p_dynesty",
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def plots_in_grid():
    """Create a 2-panel grid layout for main plot and residuals."""
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.0, 1.0])
    gs.update(left=0.2, right=0.95, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)

    ax_0 = plt.subplot(gs[0])
    ax_1 = plt.subplot(gs[1])

    minorLocator = AutoMinorLocator()
    ax_0.xaxis.set_minor_locator(minorLocator)
    ax_1.xaxis.set_minor_locator(minorLocator)

    ax_0.ticklabel_format(useOffset=False, style="plain")
    ax_1.ticklabel_format(useOffset=False, style="plain")

    return ax_0, ax_1


def resolve_run_root(outer_dir, sampler):
    """
    Your runs are often nested like:
      .../HD102365_all_instr_gp_1p_dynesty/HD102365_all_instr_gp_1p_dynesty/dynesty_plot/...
    This resolves to the directory that actually contains {sampler}_plot/.
    """
    outer_dir = outer_dir.rstrip("/")
    candidates = [
        outer_dir,
        os.path.join(outer_dir, os.path.basename(outer_dir)),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, f"{sampler}_plot")):
            return c

    # Last resort: search one level down for a folder that contains {sampler}_plot
    for sub in glob.glob(os.path.join(outer_dir, "*")):
        if os.path.isdir(os.path.join(sub, f"{sampler}_plot")):
            return sub

    return None


def load_summary_dicts(run_root, sampler):
    dict_dir = os.path.join(run_root, f"{sampler}_plot", "dictionaries")
    params_path = os.path.join(dict_dir, "summary_percentiles_parameters.p")
    derived_path = os.path.join(dict_dir, "summary_percentiles_derived.p")

    if not os.path.isfile(params_path):
        raise FileNotFoundError(f"Missing: {params_path}")
    if not os.path.isfile(derived_path):
        raise FileNotFoundError(f"Missing: {derived_path}")

    summary_percentiles_parameters = pickle.load(open(params_path, "rb"))
    summary_percentiles_derived = pickle.load(open(derived_path, "rb"))
    return summary_percentiles_parameters, summary_percentiles_derived


def detect_planets(summary_percentiles_parameters):
    """
    Detect planet keys that look like planets (dict with 'P' and 'K').
    """
    planets = []
    for k, v in summary_percentiles_parameters.items():
        if isinstance(v, dict) and ("P" in v) and ("K" in v):
            planets.append(k)
    # keep stable-ish ordering: b,c,d... if present
    planets = sorted(planets)
    return planets


def detect_datasets(model_files_dir, planet_name):
    """
    Detect dataset prefixes from files:
      <dataset>_radial_velocities_<planet>.dat
    """
    pat = os.path.join(model_files_dir, f"*_radial_velocities_{planet_name}.dat")
    ds = []
    for p in glob.glob(pat):
        base = os.path.basename(p)
        suffix = f"_radial_velocities_{planet_name}.dat"
        if base.endswith(suffix):
            ds.append(base[: -len(suffix)])
    ds = sorted(set(ds))
    return ds


def default_label(dataset_name):
    return dataset_name


# ============================================================================
# MAIN PLOTTING (PHASE-FOLDED ONLY)
# ============================================================================
def make_phase_folded_for_run(outer_dir, sampler):
    sampler = sampler.lower()
    run_root = resolve_run_root(outer_dir, sampler)
    if run_root is None:
        print(f"[SKIP] Could not find {sampler}_plot under: {outer_dir}")
        return

    run_name = os.path.basename(run_root.rstrip("/"))
    model_files_dir = os.path.join(run_root, f"{sampler}_plot", "model_files")

    # Output directory: .../figures_pyorbit_HD102365/{sampler}/{run_name}/
    out_dir = os.path.join(OUTPUT_ROOT, sampler, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Styling (same as your original)
    plt.rcParams["font.family"] = "DeJavu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    matplotlib.rcParams.update({"font.size": font_label})

    # Load dictionaries
    summary_percentiles_parameters, summary_percentiles_derived = load_summary_dicts(run_root, sampler)

    # Detect planets
    planets = detect_planets(summary_percentiles_parameters)
    if FORCE_ONLY_B_AND_C:
        planets = [p for p in planets if p in (["b"] + (["c"] if ENABLE_PLANET_C else []))]

    if len(planets) == 0:
        print(f"[SKIP] No planets found in dictionaries for run: {run_root}")
        return

    # Detect datasets from the first planet (usually b). If empty, try the next planet.
    datasets_list = []
    for p in planets:
        datasets_list = detect_datasets(model_files_dir, p)
        if len(datasets_list) > 0:
            break
    if len(datasets_list) == 0:
        print(f"[SKIP] No datasets found in model_files for run: {run_root}")
        return

    datasets_labels = {d: default_label(d) for d in datasets_list}

    # Colors per dataset (same behavior as your original)
    dataset_colors = {dataset: f"C{i}" for i, dataset in enumerate(datasets_list)}

    # Build planet_dict with your original error-band logic
    planet_dict = collections.OrderedDict()
    for planet_name in planets:
        K_arr = summary_percentiles_parameters[planet_name]["K"]
        planet_dict[planet_name] = {
            "P": summary_percentiles_parameters[planet_name]["P"][3],
            "limits_folded_x": list(DEFAULT_FOLDED_X_LIMS),
            "transit_folded": False,
            "K_error_1sigma": (K_arr[4] - K_arr[2]) / 2,
            "K_error_2sigma": (K_arr[5] - K_arr[1]) / 2,
            "K_error_3sigma": (K_arr[6] - K_arr[0]) / 2,
        }

    # ------------------------------------------------------------------------
    # PLOT 1: FOLDED RV PLOTS FOR EACH PLANET (your original logic)
    # ------------------------------------------------------------------------
    for key_name, key_val in planet_dict.items():
        # These exist only for planet runs; 0p runs will have been skipped above.
        RV_kep_path = os.path.join(model_files_dir, f"RV_planet_{key_name}_kep.dat")
        RV_pha_path = os.path.join(model_files_dir, f"RV_planet_{key_name}_pha.dat")
        RV_tcf_path = os.path.join(model_files_dir, f"RV_planet_{key_name}_Tcf.dat")

        # Keep the original read pattern (even if RV_kep is not used later)
        RV_kep = np.genfromtxt(RV_kep_path, skip_header=1)

        if key_val.get("transit_folded", True):
            RV_pha = np.genfromtxt(RV_tcf_path, skip_header=1)
        else:
            RV_pha = np.genfromtxt(RV_pha_path, skip_header=1)

        fig = plt.figure(figsize=figsize)
        ax_0, ax_1 = plots_in_grid()

        # Error bands
        K_error_1sigma = key_val["K_error_1sigma"]
        K_error_2sigma = key_val["K_error_2sigma"]
        K_error_3sigma = key_val["K_error_3sigma"]
        K_rvs = np.amax(RV_pha[:, 1])
        RV_unitary = RV_pha[:, 1] / K_rvs

        ax_0.fill_between(
            RV_pha[:, 0],
            RV_unitary * (K_rvs - K_error_1sigma),
            y2=RV_unitary * (K_rvs + K_error_1sigma),
            alpha=0.10,
            color="black",
            zorder=0,
        )
        ax_0.fill_between(
            RV_pha[:, 0],
            RV_unitary * (K_rvs - K_error_2sigma),
            y2=RV_unitary * (K_rvs + K_error_2sigma),
            alpha=0.10,
            color="black",
            zorder=0,
        )
        ax_0.fill_between(
            RV_pha[:, 0],
            RV_unitary * (K_rvs - K_error_3sigma),
            y2=RV_unitary * (K_rvs + K_error_3sigma),
            alpha=0.10,
            color="black",
            zorder=0,
        )

        # Plot model
        ax_0.plot(RV_pha[:, 0] - 1, RV_pha[:, 1], color="k", linestyle="-", zorder=2, label="RV model")
        ax_0.plot(RV_pha[:, 0] + 1, RV_pha[:, 1], color="k", linestyle="-", zorder=2)

        # Plot data for each dataset
        for n_dataset, dataset in enumerate(datasets_list):
            color = dataset_colors[dataset]

            rv_file = os.path.join(model_files_dir, f"{dataset}_radial_velocities_{key_name}.dat")
            if not os.path.isfile(rv_file):
                # Some configurations may not include all datasets; skip missing.
                continue

            RV_mod = np.genfromtxt(rv_file, skip_header=1)
            error = np.sqrt(RV_mod[:, 9] ** 2 + RV_mod[:, 12] ** 2)

            if key_val.get("transit_folded", False):
                rv_phase = RV_mod[:, 1] / planet_dict[key_name]["P"]
            else:
                rv_phase = RV_mod[:, 2]

            # Main points
            ax_0.errorbar(rv_phase, RV_mod[:, 8], yerr=error, color="black", markersize=0, alpha=0.25, fmt="o", zorder=0)
            ax_0.scatter(rv_phase, RV_mod[:, 8], c=color, s=dot_size, zorder=20 - n_dataset, alpha=1.0, label=datasets_labels.get(dataset, dataset))

            # Points at phase-1
            ax_0.errorbar(rv_phase - 1, RV_mod[:, 8], yerr=error, color="black", markersize=0, alpha=0.25, fmt="o", zorder=0)
            ax_0.scatter(rv_phase - 1, RV_mod[:, 8], c="gray", s=dot_size, zorder=20, alpha=1.0)

            # Points at phase+1
            ax_0.errorbar(rv_phase + 1, RV_mod[:, 8], yerr=error, color="black", markersize=0, alpha=0.25, fmt="o", zorder=0)
            ax_0.scatter(rv_phase + 1, RV_mod[:, 8], c="gray", s=dot_size, zorder=20, alpha=1.0)

            # Residuals
            ax_1.errorbar(rv_phase, RV_mod[:, 10], yerr=error, color="black", markersize=0, alpha=0.25, fmt="o", zorder=1)
            ax_1.scatter(rv_phase, RV_mod[:, 10], c=color, s=dot_size, zorder=20 - n_dataset, alpha=1.0)

            # Residuals at phase-1
            ax_1.errorbar(rv_phase - 1, RV_mod[:, 10], yerr=error, color="black", markersize=0, alpha=0.25, fmt="o", zorder=1)
            ax_1.scatter(rv_phase - 1, RV_mod[:, 10], c="gray", s=dot_size, zorder=20, alpha=1.0)

            # Residuals at phase+1
            ax_1.errorbar(rv_phase + 1, RV_mod[:, 10], yerr=error, color="black", markersize=0, alpha=0.25, fmt="o", zorder=1)
            ax_1.scatter(rv_phase + 1, RV_mod[:, 10], c="gray", s=dot_size, zorder=20, alpha=1.0)

        # Residuals zero line and error bands
        ax_1.axhline(0.000, c="k", zorder=3)
        ax_1.fill_between(RV_pha[:, 0], -K_error_1sigma, K_error_1sigma, alpha=0.10, color="black", zorder=0)
        ax_1.fill_between(RV_pha[:, 0], -K_error_2sigma, K_error_2sigma, alpha=0.10, color="black", zorder=0)
        ax_1.fill_between(RV_pha[:, 0], -K_error_3sigma, K_error_3sigma, alpha=0.10, color="black", zorder=0)

        # Set limits
        if key_val.get("limits_folded_x", False):
            ax_0.set_xlim(key_val["limits_folded_x"][0], key_val["limits_folded_x"][1])
            ax_1.set_xlim(key_val["limits_folded_x"][0], key_val["limits_folded_x"][1])

        # Vertical lines
        if key_val.get("transit_folded", False):
            ax_0.axvline(-0.500, c="k", zorder=3, alpha=0.5, linestyle="--")
            ax_0.axvline(0.500, c="k", zorder=3, alpha=0.5, linestyle="--")
            ax_1.axvline(-0.500, c="k", zorder=3, alpha=0.5, linestyle="--")
            ax_1.axvline(0.500, c="k", zorder=3, alpha=0.5, linestyle="--")
        else:
            ax_0.axvline(0.00, c="k", zorder=3, alpha=0.5, linestyle="--")
            ax_0.axvline(1.00, c="k", zorder=3, alpha=0.5, linestyle="--")
            ax_1.axvline(0.00, c="k", zorder=3, alpha=0.5, linestyle="--")
            ax_1.axvline(1.00, c="k", zorder=3, alpha=0.5, linestyle="--")

        # Formatting
        ax_0.axes.get_xaxis().set_ticks([])
        ax_0.yaxis.set_major_locator(MultipleLocator(5))
        ax_0.yaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax_0.yaxis.set_minor_locator(MultipleLocator(1))
        ax_1.yaxis.set_major_locator(MultipleLocator(5))
        ax_1.yaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax_1.yaxis.set_minor_locator(MultipleLocator(1))

        ax_0.set_ylabel("RV [m/s]")
        ax_1.set_xlabel("Orbital Phase")
        ax_1.set_ylabel("Residuals [m/s]")

        handles, labels = ax_0.get_legend_handles_labels()
        from collections import OrderedDict
        unique = OrderedDict()
        for h, l in zip(handles, labels):
            if l not in unique and l is not None:
                unique[l] = h
        ax_0.legend(unique.values(), unique.keys(), framealpha=1.0, loc="lower left")

        plot_filename = f"{run_name}_{key_name}_folded.png"
        out_path = os.path.join(out_dir, plot_filename)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"[OK] {sampler}: {run_name} -> {out_dir}")


# ============================================================================
# ENTRY POINT
# ============================================================================
def main():
    # Ensure output root exists
    os.makedirs(os.path.join(OUTPUT_ROOT, "emcee"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "dynesty"), exist_ok=True)

    # EMCEE
    for d in EMCEE_RUN_DIRS:
        try:
            make_phase_folded_for_run(d, "emcee")
        except Exception as e:
            print(f"[FAIL] emcee run {d}: {e}")

    # DYNASTY
    for d in DYNASTY_RUN_DIRS:
        try:
            make_phase_folded_for_run(d, "dynesty")
        except Exception as e:
            print(f"[FAIL] dynesty run {d}: {e}")


if __name__ == "__main__":
    main()
