#!/usr/bin/env python3
"""
Unified ESSPIV analyzer for PyORBIT logs (emcee + dynesty) — BEST MODELS ONLY.

Selection logic:
  - emcee runs: best model = lowest "Median BIC (using likelihood)" within each (Dataset, Configuration)
  - dynesty runs: best model = highest "logz" within each (Dataset, Configuration)

Parameter parsing (median + 15–84p errors):
  - emcee: parse from the 3rd occurrence (1-indexed) of:
      "Statistics on the model parameters obtained from the posteriors samples"
  - dynesty: parse from the 1st occurrence (1-indexed) of the same header

Output:
  - ONLY one CSV:
      best_models_with_errors_<timestamp>.csv

Usage:
  python best_models_ESSPIV_unified.py /path/to/Results_submitted > terminal_output_best_models.txt 2>&1
"""

import re
import os
import sys
import glob
import pandas as pd
from datetime import datetime


# ----------------------------
# Small helpers
# ----------------------------

def planet_sort_key(p):
    m = re.match(r"^(\d+)p$", str(p))
    return int(m.group(1)) if m else 999


def parse_filename_fields(filepath):
    """
    Handles:
      configuration_file_emcee_run_DS1_1p_2_activity_indi.log
      configuration_file_run_DS1_1p_2_activity_indi.log
      configuration_file_dynesty_run_DS1_1p_2_activity_indi.log  (if exists)
    """
    basename = os.path.basename(filepath)
    cleaned = basename.replace(".log", "")

    # remove common prefixes
    for pref in [
        "configuration_file_emcee_run_",
        "configuration_file_run_",
        "configuration_file_dynesty_run_",
        "configuration_file_nested_run_",
    ]:
        if cleaned.startswith(pref):
            cleaned = cleaned[len(pref):]
            break

    m = re.match(r"(DS\d+)_([0-9]+p)_(.*)", cleaned)
    if m:
        return m.group(1), m.group(2), m.group(3)

    return "Unknown", "N/A", cleaned


def get_triplet(pdict):
    """Return (value, err_minus, err_plus) as strings; blanks if missing."""
    if not isinstance(pdict, dict):
        return ("", "", "")
    v = pdict.get("value", "")
    em = pdict.get("err_minus", "")
    ep = pdict.get("err_plus", "")
    return (
        "" if v is None else str(v),
        "" if em is None else str(em),
        "" if ep is None else str(ep),
    )


def select_stats_block_start(lines, prefer_nth=3, lookahead=800):
    """
    Choose which "Statistics on the model parameters..." block to parse.
    Prefer nth occurrence (1-indexed). If not available, fallback to last with "(15-84 p)", else last.
    """
    header = "Statistics on the model parameters obtained from the posteriors samples"
    idxs = [i for i, line in enumerate(lines) if header in line]
    if not idxs:
        return -1

    if len(idxs) >= prefer_nth:
        return idxs[prefer_nth - 1]

    for idx in reversed(idxs):
        chunk = "\n".join(lines[idx:min(len(lines), idx + lookahead)])
        if "(15-84 p)" in chunk:
            return idx

    return idxs[-1]


def detect_method_from_content(content):
    """
    Decide whether this is dynesty or emcee.
    Dynesty logs typically contain 'logz: ... +/- ...' or '*** Information criteria using the likelihood distribution' etc.
    We use 'logz:' as primary dynesty indicator.
    """
    if re.search(r"\blogz:\s*-?[\d\.]+\s*\+/-\s*[\d\.]+", content):
        return "dynesty"
    # emcee logs always have "Running emcee" and later "Median BIC ..."
    if "Running emcee" in content or re.search(r"Median BIC\s+\(using likelihood\)\s*=", content):
        return "emcee"
    # fallback
    return "unknown"


# ----------------------------
# Parsing (shared)
# ----------------------------

def parse_parameters_block(content, prefer_stats_block_n):
    """
    Parse orbital and activity parameters from the selected stats block.
    Returns (orbital_parameters, activity_parameters).
    """
    orbital_parameters = {}
    activity_parameters = {}

    lines = content.split("\n")
    start_idx = select_stats_block_start(lines, prefer_nth=prefer_stats_block_n, lookahead=800)
    if start_idx == -1:
        return orbital_parameters, activity_parameters

    current_planet = None
    in_activity_section = False

    num = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

    stop_markers = (
        "Statistics on the derived parameters obtained from the posteriors samples",
        "Statistics on the derived parameters",
        "Parameters corresponding to",
        "*** Information criteria",
        "Gelman-Rubin:",
    )

    for i in range(start_idx, len(lines)):
        line = lines[i]

        if any(m in line for m in stop_markers) and i > start_idx:
            break

        planet_match = re.search(r"----- common model:\s+([a-z])\s*$", line)
        if planet_match:
            current_planet = planet_match.group(1)
            in_activity_section = False
            orbital_parameters.setdefault(current_planet, {})
            continue

        if "----- common model:  activity" in line:
            in_activity_section = True
            current_planet = None
            continue

        s = line.strip()
        if not s:
            continue

        m4 = re.match(
            rf"^([A-Za-z_]+)\s+({num})\s+({num})\s+({num}).*\(15-84 p\)",
            s
        )
        m2 = re.match(rf"^([A-Za-z_]+)\s+({num})\s*$", s)

        if m4:
            pname = m4.group(1)
            median_val = float(m4.group(2))
            err_minus = abs(float(m4.group(3)))
            err_plus = abs(float(m4.group(4)))
        elif m2:
            pname = m2.group(1)
            median_val = float(m2.group(2))
            err_minus = None
            err_plus = None
        else:
            continue

        if in_activity_section:
            activity_parameters[pname] = {
                "value": median_val,
                "err_minus": err_minus,
                "err_plus": err_plus,
            }
        elif current_planet:
            orbital_parameters[current_planet][pname] = {
                "value": median_val,
                "err_minus": err_minus,
                "err_plus": err_plus,
            }

    return orbital_parameters, activity_parameters


def parse_log_file(filepath, emcee_stats_block_n=3, dynesty_stats_block_n=1):
    """
    Parse a log file and return a dict with:
      Dataset, Configuration, Planets, Method, logZ, MedianBIC, parameters, etc.
    Returns None if neither selection metric can be found.
    """
    try:
        with open(filepath, "r") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    method = detect_method_from_content(content)

    # Dynesty metrics
    logz = None
    logz_err = None
    m = re.search(r"\blogz:\s*(-?[\d\.]+)\s*\+/-\s*([\d\.]+)", content)
    if m:
        logz = float(m.group(1))
        logz_err = float(m.group(2))

    # Emcee metric (and also present in dynesty/emcee summaries sometimes)
    median_bic = None
    m = re.search(r"Median BIC\s+\(using likelihood\)\s*=\s*(-?[\d\.]+)", content)
    if m:
        median_bic = float(m.group(1))

    dataset, planets, config = parse_filename_fields(filepath)

    # Pick which stats block number to use
    if method == "emcee":
        prefer_stats_n = emcee_stats_block_n
    elif method == "dynesty":
        prefer_stats_n = dynesty_stats_block_n
    else:
        # try emcee-style first, then dynesty-style
        prefer_stats_n = emcee_stats_block_n

    orbital_params, activity_params = parse_parameters_block(content, prefer_stats_n)

    # Must have a metric to participate
    if method == "dynesty":
        if logz is None:
            # if mis-detected dynesty, but it has BIC, keep it as emcee-style fallback
            if median_bic is None:
                return None
            method = "emcee"
        # keep as dynesty even if BIC exists
    elif method == "emcee":
        if median_bic is None:
            # allow emcee logs with missing BIC? usually not useful
            return None
    else:
        # unknown: accept if at least one metric exists
        if logz is None and median_bic is None:
            return None

    return {
        "Dataset": dataset,
        "Configuration": config,
        "Planets": planets,
        "Method": method,
        "log(Z)": logz,
        "log(Z) error": logz_err,
        "Median BIC": median_bic,
        "Orbital Parameters": orbital_params,
        "Activity Parameters": activity_params,
        "File": os.path.basename(filepath),
        "Directory": os.path.dirname(os.path.abspath(filepath)),
    }


# ----------------------------
# Best-model selection + export
# ----------------------------

def analyze_and_export_best_only(log_files, output_dir, emcee_stats_block_n=3, dynesty_stats_block_n=1):
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    failed = []

    print(f"Processing {len(log_files)} log files...")
    for lf in log_files:
        d = parse_log_file(
            lf,
            emcee_stats_block_n=emcee_stats_block_n,
            dynesty_stats_block_n=dynesty_stats_block_n
        )
        if d is None:
            failed.append(lf)
        else:
            rows.append(d)

    if failed:
        print(f"\nWarning: {len(failed)} files skipped (missing usable metric).")
        for f in failed[:10]:
            print(f"  - {f}")
        if len(failed) > 10:
            print("  ... (truncated)")
        print()

    if not rows:
        print("No usable logs found.")
        return

    df = pd.DataFrame(rows)

    best_rows = []

    # Group per Dataset x Configuration x Method
    # (because you said dynesty needs different best-rule; keep them separate)
    for (dataset, config, method), g in df.groupby(["Dataset", "Configuration", "Method"]):
        g = g.copy()
        g["_psort"] = g["Planets"].apply(planet_sort_key)
        g = g.sort_values("_psort")

        criterion_used = None

        if method == "dynesty":
            # best by max logZ
            if g["log(Z)"].isna().all():
                # fallback to BIC if evidence missing
                best_idx = g["Median BIC"].idxmin()
                criterion_used = "Median BIC (fallback)"
            else:
                best_idx = g["log(Z)"].idxmax()
                criterion_used = "log(Z)"
        else:
            # emcee: best by min Median BIC
            if g["Median BIC"].isna().all():
                # fallback to logZ if somehow present
                best_idx = g["log(Z)"].idxmax()
                criterion_used = "log(Z) (fallback)"
            else:
                best_idx = g["Median BIC"].idxmin()
                criterion_used = "Median BIC"

        best = g.loc[best_idx]

        orb = best.get("Orbital Parameters") or {}
        planet_labels = sorted(orb.keys())

        P_v, P_em, P_ep = [], [], []
        K_v, K_em, K_ep = [], [], []
        e_v, e_em, e_ep = [], [], []

        for pl in planet_labels:
            pdata = orb.get(pl, {})
            a, b, c = get_triplet(pdata.get("P", {}))
            P_v.append(a); P_em.append(b); P_ep.append(c)
            a, b, c = get_triplet(pdata.get("K", {}))
            K_v.append(a); K_em.append(b); K_ep.append(c)
            a, b, c = get_triplet(pdata.get("e", {}))
            e_v.append(a); e_em.append(b); e_ep.append(c)

        best_rows.append({
            "Dataset": dataset,
            "Configuration": config,
            "Method": method,
            "CriterionUsed": criterion_used,
            "Best_Model_Planets": best["Planets"],

            "log(Z)": best.get("log(Z)", ""),
            "log(Z) error": best.get("log(Z) error", ""),
            "Median BIC": best.get("Median BIC", ""),

            "Planets": ",".join(planet_labels),

            "P_days": ",".join(P_v),
            "P_err_minus": ",".join(P_em),
            "P_err_plus": ",".join(P_ep),

            "K_mps": ",".join(K_v),
            "K_err_minus": ",".join(K_em),
            "K_err_plus": ",".join(K_ep),

            "e": ",".join(e_v),
            "e_err_minus": ",".join(e_em),
            "e_err_plus": ",".join(e_ep),

            "File": best["File"],
            "Directory": best["Directory"],
        })

    out_df = pd.DataFrame(best_rows).sort_values(["Dataset", "Configuration", "Method"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(output_dir, f"best_models_with_errors_{timestamp}.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"\nWrote: {out_csv}")
    print(f"Stats blocks: emcee uses #{emcee_stats_block_n}, dynesty uses #{dynesty_stats_block_n}\n")


# ----------------------------
# Main
# ----------------------------

def main():
    if len(sys.argv) > 1:
        search_directory = sys.argv[1]
    else:
        search_directory = "."

    if not os.path.isdir(search_directory):
        print(f"Error: Directory '{search_directory}' does not exist.")
        sys.exit(1)

    print(f"Searching for log files in: {os.path.abspath(search_directory)}\n")

    all_log_files = glob.glob(os.path.join(search_directory, "**", "*.log"), recursive=True)
    all_log_files = [
        f for f in all_log_files
        if os.path.basename(f).startswith("configuration_file_") and "_run_" in os.path.basename(f)
    ]

    # Deduplicate by basename: choose shallowest path
    file_groups = {}
    for lf in all_log_files:
        file_groups.setdefault(os.path.basename(lf), []).append(lf)

    log_files = []
    for bname, files in file_groups.items():
        if len(files) == 1:
            log_files.append(files[0])
        else:
            best = min(files, key=lambda p: p.count(os.sep))
            log_files.append(best)
            print(f"Note: Found {len(files)} copies of '{bname}', using: {best}")

    if not log_files:
        print("No matching .log files found.")
        sys.exit(1)

    print(f"\nFound {len(log_files)} unique log files to analyze.\n")

    # output folder name = basename of search_directory
    out_dir = os.path.basename(os.path.normpath(search_directory)) or "output_best_models"

    analyze_and_export_best_only(
        log_files,
        output_dir=out_dir,
        emcee_stats_block_n=3,
        dynesty_stats_block_n=1
    )


if __name__ == "__main__":
    main()
