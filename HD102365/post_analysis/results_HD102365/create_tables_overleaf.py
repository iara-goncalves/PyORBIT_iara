#!/usr/bin/env python3
"""
Two tables total (dynesty + emcee), ALL configurations included.

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


# -------------------- helpers --------------------

DASH = "—"

def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

def _safe_literal_dict(s: Any) -> Dict[str, Any]:
    import ast
    if s is None:
        return {}
    if isinstance(s, dict):
        return s
    txt = str(s).strip()
    if txt in {"", "{}", "nan", "NaN", "None"}:
        return {}
    try:
        obj = ast.literal_eval(txt)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _clean_planet_list(x: Any) -> List[str]:
    if x is None:
        return []
    s = str(x).strip().strip('"').strip("'").strip()
    if s == "" or s.lower() == "nan":
        return []
    return [p.strip() for p in s.split(",") if p.strip() != ""]

def _split_listlike(x: Any, n: int) -> List[str]:
    if x is None:
        return [""] * n
    s = str(x).strip().strip('"').strip("'").strip()
    if s == "" or s.lower() == "nan":
        return [""] * n
    vals = [v.strip() for v in s.split(",")] if "," in s else [s]
    if len(vals) < n:
        vals += [""] * (n - len(vals))
    return vals[:n]

def _first_present(row: pd.Series, cols: List[str], default: str = "") -> str:
    for c in cols:
        if c in row.index:
            v = row.get(c)
            if v is not None and str(v).strip() != "" and str(v).lower() != "nan":
                return str(v).strip()
    return default

def _format_dynesty_estimate(d: Dict[str, Any]) -> str:
    if not isinstance(d, dict) or not d:
        return ""
    v = d.get("value_str", d.get("value", ""))
    v = "" if v is None else str(v)

    lo = d.get("lower_error_str", d.get("lower_error", ""))
    hi = d.get("upper_error_str", d.get("upper_error", ""))

    lo = "" if lo is None else str(lo).strip()
    hi = "" if hi is None else str(hi).strip()

    def _abs_str(x: str) -> str:
        x = x.strip()
        return x[1:].strip() if x.startswith("-") else x

    if lo == "" and hi == "":
        return v

    hi_clean = hi.lstrip("+").strip()
    lo_abs = _abs_str(lo)

    parts = []
    if hi_clean != "":
        parts.append(f"+{hi_clean}")
    if lo_abs != "":
        parts.append(f"-{lo_abs}")

    return f"{v} ({'/'.join(parts)})" if parts else v

def _gp_value(row: pd.Series) -> str:
    gp = _first_present(row, ["GPType"], default="")
    if gp != "":
        return gp
    # fallback: infer from Configuration text
    cfg = _first_present(row, ["Configuration"], default="").lower()
    if "no_gp" in cfg:
        return "no_gp"
    if "_gp" in cfg or "gp" in cfg:
        return "gp"
    return ""

def _fmt_asym(value: str, err_minus: str, err_plus: str) -> str:
    value = (value or "").strip()
    em = (err_minus or "").strip()
    ep = (err_plus or "").strip()

    if value == "" or value.lower() == "nan":
        return DASH

    # if no errors provided, just return the value
    if (em == "" or em.lower() == "nan") and (ep == "" or ep.lower() == "nan"):
        return value

    # ensure minus is shown as positive magnitude in the -... part
    em_mag = em[1:] if em.startswith("-") else em
    ep_mag = ep[1:] if ep.startswith("-") else ep

    parts = []
    if ep_mag != "" and ep_mag.lower() != "nan":
        parts.append(f"+{ep_mag}")
    if em_mag != "" and em_mag.lower() != "nan":
        parts.append(f"-{em_mag}")

    return f"{value} ({'/'.join(parts)})" if parts else value


# -------------------- grouping / multirow --------------------

def _run_lengths(vals: List[str]) -> List[int]:
    """For each index, length of the run starting at i if i is start else 0."""
    n = len(vals)
    out = [0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and vals[j] == vals[i]:
            j += 1
        out[i] = j - i
        i = j
    return out

def _latex_escape(s: str) -> str:
    # minimal escaping for table cells
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = str(s)
    for k, v in repl.items():
        out = out.replace(k, v)
    return out

def df_to_latex_multirow_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    multirow_cols: List[str],
) -> str:
    """
    Create LaTeX table with \\multirow merges for consecutive identical values.

    Merging is hierarchical:
      - multirow_cols[0] merges over its full consecutive runs
      - multirow_cols[1] merges only within blocks of multirow_cols[0]
      - multirow_cols[2] merges only within blocks of (multirow_cols[0], multirow_cols[1])
      - etc.

    Requires in LaTeX preamble: \\usepackage{multirow}
    Assumes df already sorted in the desired order.
    """
    df = df.copy().reset_index(drop=True)
    cols = list(df.columns)

    # compute hierarchical run lengths for each multirow column
    n = len(df)
    runs_map: Dict[str, List[int]] = {}

    if n > 0 and multirow_cols:
        # first level: plain runs on the first multirow column
        first = multirow_cols[0]
        first_vals = df[first].astype(str).tolist()
        runs_map[first] = _run_lengths(first_vals)

        # deeper levels: runs reset when any previous grouping col changes
        for level in range(1, len(multirow_cols)):
            col = multirow_cols[level]
            runs = [0] * n
            i = 0
            while i < n:
                key = tuple(df.loc[i, multirow_cols[:level]].astype(str).tolist())
                j = i
                while j < n and tuple(df.loc[j, multirow_cols[:level]].astype(str).tolist()) == key:
                    j += 1

                block = df.loc[i:j-1, col].astype(str).tolist()
                block_runs = _run_lengths(block)
                for k in range(i, j):
                    runs[k] = block_runs[k - i]
                i = j

            runs_map[col] = runs

    # build LaTeX
    # (keep same column formatting idea: first 3 left-ish, rest centered)
    colfmt = "lll" + "c" * max(0, len(cols) - 3)

    lines: List[str] = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(f"\\begin{{tabular}}{{{colfmt}}}")
    lines.append("\\hline")
    lines.append(" & ".join([_latex_escape(c) for c in cols]) + r" \\")
    lines.append("\\hline")

    for i in range(n):
        row_cells: List[str] = []
        for c in cols:
            val = _latex_escape(str(df.at[i, c]))
            if c in runs_map:
                run = runs_map[c][i]
                if run > 0:
                    row_cells.append(rf"\multirow{{{run}}}{{*}}{{{val}}}")
                else:
                    row_cells.append("")  # covered by multirow above
            else:
                row_cells.append(val)

        lines.append(" & ".join(row_cells) + r" \\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    lines.append(f"\\label{{{_latex_escape(label)}}}")
    lines.append("\\end{table}")

    return "\n".join(lines) + "\n"


def df_for_render_blank_repeats(df: pd.DataFrame, cols_to_blank: List[str]) -> pd.DataFrame:
    """For PNG/PDF: blank repeated consecutive values to look 'connected'."""
    out = df.copy().reset_index(drop=True)
    for col in cols_to_blank:
        prev = None
        for i in range(len(out)):
            v = out.at[i, col]
            if i > 0 and v == prev:
                out.at[i, col] = ""
            prev = v
    return out


# -------------------- dynesty / emcee conversion --------------------

def dynesty_to_planet_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "Orbital Parameters" not in df.columns:
        raise SystemExit("[dynesty] Missing 'Orbital Parameters' column")

    rows = []
    for _, r in df.iterrows():
        cfg_group = _first_present(r, ["ConfigGroup", "Configuration"], default="")
        gp = _gp_value(r)
        model = _first_present(r, ["Planets", "Model"], default="")

        orb = _safe_literal_dict(r.get("Orbital Parameters", "{}"))
        if not orb:
            rows.append({"CONFIG": cfg_group, "MODEL": model, "PLANET": DASH, "P [days]": DASH, "K [m/s]": DASH, "e": DASH})
            continue

        any_planet = False
        for planet, pdict in orb.items():
            if not isinstance(pdict, dict):
                continue
            any_planet = True
            rows.append({
                "CONFIG": cfg_group,
                "MODEL": model,
                "GP": gp,
                "PLANET": str(planet),
                "P [days]": _format_dynesty_estimate(pdict.get("P", {})) or DASH,
                "K [m/s]": _format_dynesty_estimate(pdict.get("K", {})) or DASH,
                "e": _format_dynesty_estimate(pdict.get("e", {})) or DASH,
            })

        if not any_planet:
            rows.append({"CONFIG": cfg_group, "MODEL": model, "GP": gp, "PLANET": DASH, "P [days]": DASH, "K [m/s]": DASH, "e": DASH})

    out = pd.DataFrame(rows, columns=["CONFIG", "MODEL", "GP", "PLANET", "P [days]", "K [m/s]", "e"])
    out = out.sort_values(["CONFIG", "MODEL", "PLANET"], kind="stable").reset_index(drop=True)
    return out

def emcee_to_planet_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "Planets" not in df.columns:
        raise SystemExit("[emcee] Missing 'Planets' column")

    rows = []
    for _, r in df.iterrows():
        cfg_group = _first_present(r, ["ConfigGroup", "Configuration"], default="")
        gp = _gp_value(r)
        model = _first_present(r, ["Best_Model_Planets", "Model"], default="")

        planets = _clean_planet_list(r.get("Planets", ""))
        if not planets:
            rows.append({"CONFIG": cfg_group, "GP": gp, "MODEL": model, "PLANET": DASH,
                         "P [days]": DASH, "K [m/s]": DASH, "e": DASH})
            continue

        n = len(planets)

        P      = _split_listlike(r.get("P_days", ""), n)
        Pm     = _split_listlike(r.get("P_err_minus", ""), n)
        Pp     = _split_listlike(r.get("P_err_plus", ""), n)

        K      = _split_listlike(r.get("K_mps", ""), n)
        Km     = _split_listlike(r.get("K_err_minus", ""), n)
        Kp     = _split_listlike(r.get("K_err_plus", ""), n)

        E      = _split_listlike(r.get("e", ""), n)
        Em     = _split_listlike(r.get("e_err_minus", ""), n)
        Ep     = _split_listlike(r.get("e_err_plus", ""), n)

        for i, pl in enumerate(planets):
            rows.append({
                "CONFIG": cfg_group,
                "GP": gp,
                "MODEL": model,
                "PLANET": pl,
                "P [days]": _fmt_asym(P[i], Pm[i], Pp[i]),
                "K [m/s]": _fmt_asym(K[i], Km[i], Kp[i]),
                "e": _fmt_asym(E[i], Em[i], Ep[i]),
            })

    out = pd.DataFrame(rows, columns=["CONFIG", "GP", "MODEL", "PLANET", "P [days]", "K [m/s]", "e"])
    out = out.sort_values(["CONFIG", "GP", "MODEL", "PLANET"], kind="stable").reset_index(drop=True)
    return out


# -------------------- output --------------------

def render_table_figure(df: pd.DataFrame, outpath: Path, title: str, dpi: int = 300) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows, ncols = df.shape
    fig_w = min(16, max(9, 1.35 * ncols + 2))
    fig_h = min(22, max(3.0, 0.42 * (nrows + 2)))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=df.values,
        colLabels=list(df.columns),
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.0, 1.15)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_linewidth(1.0)

    ax.set_title(title, fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def write_outputs(df: pd.DataFrame, outdir: Path, name: str, formats: List[str], dpi: int,
                  caption: str, label: str) -> None:
    base = outdir / name

    # LaTeX with true merged cells for CONFIG and MODEL
    if "tex" in formats:
        tex = df_to_latex_multirow_table(
            df,
            caption=caption,
            label=label,
            multirow_cols=["CONFIG", "GP", "MODEL"],
        )
        base.with_suffix(".tex").write_text(tex, encoding="utf-8")

    # PNG/PDF: blank repeats to visually connect
    df_fig = df_for_render_blank_repeats(df, ["CONFIG", "GP", "MODEL"])

    if "png" in formats:
        render_table_figure(df_fig, base.with_suffix(".png"), caption, dpi=dpi)
    if "pdf" in formats:
        render_table_figure(df_fig, base.with_suffix(".pdf"), caption, dpi=dpi)


# -------------------- main --------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Create exactly two tables (dynesty + emcee), all configurations, grouped display.")
    p.add_argument("--dynesty", type=str, default=None)
    p.add_argument("--emcee", type=str, default=None)
    p.add_argument("--outdir", type=str, default="tables_overleaf")
    p.add_argument("--formats", nargs="+", default=["tex", "png"], choices=["tex", "png", "pdf"])
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    outdir = Path(args.outdir)
    _ensure_outdir(outdir)

    if not args.dynesty and not args.emcee:
        raise SystemExit("Provide --dynesty and/or --emcee")

    if args.dynesty:
        ddf = pd.read_csv(args.dynesty)
        dyn_tbl = dynesty_to_planet_rows(ddf)
        write_outputs(
            dyn_tbl, outdir,
            name="dynesty_planet_params",
            formats=args.formats, dpi=args.dpi,
            caption="Planet parameters from dynesty best models (all configurations).",
            label="tab:dynesty_planet_params",
        )

    if args.emcee:
        edf = pd.read_csv(args.emcee)
        emc_tbl = emcee_to_planet_rows(edf)
        write_outputs(
            emc_tbl, outdir,
            name="emcee_planet_params",
            formats=args.formats, dpi=args.dpi,
            caption="Planet parameters from emcee best models (all configurations).",
            label="tab:emcee_planet_params",
        )

if __name__ == "__main__":
    main()
