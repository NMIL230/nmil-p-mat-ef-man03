#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils_plot.py

Generic plotting utilities for comparing multiple "methods" across sample sizes
(e.g., number of trials) with flexible styling and axis control.

Key features
------------
1) Plots multiple methods over N (e.g., trials/tests) with mean lines and
   optional error visualization (shaded band or error bars).
2) Supports linear/log scales on both axes and **smart auto limits** that
   tightly fit your data with small margins (works for linear and log scales).
3) Manual axis override via `xlim`/`ylim` when needed.
4) Reusable, injectable styling: pass `method_properties` to control colors,
   markers, and human-friendly labels; or use the convenience function
   `build_method_properties(...)`.

Intended input format
---------------------
`df_plot`: a tidy DataFrame with **one row per (Method, N)** containing:
- Method : str   (method identifier, e.g. 'DLVM_DALE', 'IMLE_TB', ...)
- N      : int   (x-axis value, e.g. number of trials)
- Value  : float (y-axis central tendency, e.g. mean KLD or RMSE)
- Error  : float (dispersion to show; see `error_type`)

Important: The meaning of `Error` must match `error_type`.
- If `error_type='SD'`, `Error` should contain standard deviations.
- If `error_type='SE'`, `Error` should contain standard errors.

Dependencies
------------
- numpy, pandas, matplotlib, seaborn

Example
-------
>>> import pandas as pd
>>> from utils_plot import plot_generic_comparison, build_method_properties
>>>
>>> # Minimal synthetic example
>>> df_plot = pd.DataFrame({
...     "Method": ["DLVM_DALE"] * 5 + ["IMLE_TB"] * 5,
...     "N":      [1, 2, 5, 10, 20] * 2,
...     "Value":  [0.30, 0.24, 0.18, 0.15, 0.12, 0.35, 0.28, 0.22, 0.19, 0.16],
...     "Error":  [0.02, 0.02, 0.015, 0.012, 0.01, 0.025, 0.02, 0.016, 0.013, 0.011],
... })
>>>
>>> # Either build styles automatically from methods...
>>> styles = build_method_properties(sorted(df_plot["Method"].unique()))
>>>
>>> # ...or define manually (advanced users)
>>> styles_manual = {
...     "DLVM_DALE": {"color": "#1f77b4", "marker": "s", "label": "DLVM + DALE"},
...     "IMLE_TB":   {"color": "black",   "marker": "o", "label": "IMLE + TB"},
... }
>>>
>>> # Auto limits (tight margins) on a log y-axis, with shaded ±SD bands:
>>> plot_generic_comparison(
...     df_plot=df_plot,
...     output_path="example_plot.pdf",
...     x_scale="linear",
...     y_scale="log",
...     show_markers=True,
...     error_type="SD",
...     title="Example — KLD vs Trials",
...     xlabel="Number of trials",
...     ylabel="KLD",
...     method_properties=styles,          # or styles_manual
...     xlim=None, ylim=None,              # auto
...     x_margin=0.05, y_margin=0.08,     # tweak margins if desired
...     min_decades=0.15
... )

Notes on axis limits
--------------------
- Auto mode: limits are computed from data with small margins.
  * Linear: expands [min, max] by a fraction (x_margin/y_margin).
  * Log: expands in log10 space; requires positive data.
- Manual override:
  * `xlim=(1, 100)` sets x-axis to [1, 100].
  * `ylim=(1e-2, 1)` sets y-axis to [0.01, 1].
  * Set to None (default) for auto behavior.

Why "smart auto limits"?
------------------------
If your data live well above 1e-2 on a log y-axis, default matplotlib limits
can waste half the axis below your data. These utilities shrink the empty space
by fitting the limits around your actual values (with a small visual margin).
"""

from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker



# ---------------------------------------------------------------------------
# Axis limit helpers
# ---------------------------------------------------------------------------
def _safe_min_max(vals: np.ndarray) -> Tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (0.0, 1.0)
    return float(vals.min()), float(vals.max())


def compute_axis_limits(
    values: np.ndarray,
    *,
    scale: str = "linear",
    margin: float = 0.08,
    min_decades: float = 0.15,
) -> Tuple[float, float]:
    """
    Compute "tight" axis limits around data with a small margin.

    - Linear scale: expands [min, max] by `margin * span` on both sides.
    - Log scale: expands in log10 space by max(margin * span, min_decades).
      Non-positive values are ignored (log axes require > 0).

    Parameters
    ----------
    values : array-like
        Data values to fit the axis around.
    scale : {'linear', 'log'}, optional
        Axis scale. Default: 'linear'.
    margin : float, optional
        Relative margin. For linear, fraction of the span; for log, fraction
        of the log10 span. Default: 0.08.
    min_decades : float, optional
        Minimum margin (in log10 units) to keep on log scales. Default: 0.15.

    Returns
    -------
    (lo, hi) : tuple of float
        Suggested axis bounds.

    Examples
    --------
    Linear:
    >>> compute_axis_limits([10, 20, 30], scale='linear', margin=0.1)
    (7.0..., 33.0...)

    Log:
    >>> compute_axis_limits([0.1, 0.5, 1.0], scale='log', margin=0.05, min_decades=0.1)
    (approx 0.079..., approx 1.26...)
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (0.0, 1.0)

    if scale == "log":
        vpos = v[v > 0]
        if len(vpos) == 0:
            # Fallback to linear if nothing is positive
            lo, hi = _safe_min_max(v)
            if lo == hi:
                lo -= 0.5
                hi += 0.5
            return (lo, hi)

        vmin, vmax = float(vpos.min()), float(vpos.max())
        if vmin == vmax:
            d = max(min_decades, 0.2)
            return (10 ** (np.log10(vmin) - d), 10 ** (np.log10(vmax) + d))

        log_min, log_max = np.log10(vmin), np.log10(vmax)
        span = log_max - log_min
        d = max(span * margin, min_decades)
        return (10 ** (log_min - d), 10 ** (log_max + d))

    # Linear scale
    vmin, vmax = _safe_min_max(v)
    if vmin == vmax:
        pad = max(1e-12, abs(vmin) * 0.05)
        return (vmin - pad, vmax + pad)
    span = vmax - vmin
    lo = vmin - span * margin
    hi = vmax + span * margin
    # If data are strictly nonnegative, keep lower bound at 0 if the margin dips negative
    if vmin >= 0 and lo < 0:
        lo = 0.0
    return (lo, hi)


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------
def plot_generic_comparison(
    df_plot: pd.DataFrame,
    output_path: Optional[str],
    *,
    x_scale: str = 'linear',
    y_scale: str = 'linear',
    show_markers: bool = True,
    error_type: str = 'SD',  # 'SD' or 'SE'
    title: str = 'Method Comparison',
    xlabel: str = 'Number of trials',
    ylabel: str = 'Value',
    method_properties: Optional[Dict[str, Dict[str, str]]] = None,
    legend_order: Optional[List[str]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    x_margin: float = 0.05,
    y_margin: float = 0.08,
    min_decades: float = 0.15,
    # New: start-side padding ratios (extra blank space at axis start)
    x_start_pad_ratio: float = 0.0,
    y_start_pad_ratio: float = 0.0,
    # New: font sizes
    title_fontsize: float = 14.0,
    label_fontsize: float = 12.0,
    tick_fontsize: float = 8.0,
    legend_fontsize: float = 6.0,
    # New: x tick step for linear axis (default 20)
    x_tick_step: Optional[float] = 20.0,
    x_ticks: Optional[List[float]] = None,  # if not None, use these ticks instead of the default ones
    # New: figure size (inches)
    fig_size: Tuple[float, float] = (6.4, 4.8),
    # New: allow negative start pad (skip clamping at 0)
    allow_negative_x_pad: bool = False,
    allow_negative_y_pad: bool = False,
    # New: line width for curves
    line_width: float = 2.0,
    # New: optionally return the matplotlib Figure instead of closing
    return_fig: bool = False,
    show_grid: bool = False,
    legend_ncol: int = 2,
) -> Optional[plt.Figure]:
    """
    Plot a generic "method vs. N" comparison with flexible styling and axis control.

    Parameters
    ----------
    df_plot : pandas.DataFrame
        Must contain the following columns:
          - 'Method' (str): method identifier (e.g., 'DLVM_DALE', 'IMLE_TB', ...)
          - 'N'      (int): x-axis position (e.g., number of trials)
          - 'Value'  (float): central tendency (mean) to plot on y-axis
          - 'Error'  (float): dispersion to visualize; **must match `error_type`**
            If `error_type='SD'`, supply standard deviations.
            If `error_type='SE'`, supply standard errors.

        Example:
        >>> df_plot = pd.DataFrame({
        ...     "Method": ["DLVM_DALE"] * 5 + ["IMLE_TB"] * 5,
        ...     "N":      [1, 2, 5, 10, 20] * 2,
        ...     "Value":  [0.30, 0.24, 0.18, 0.15, 0.12, 0.35, 0.28, 0.22, 0.19, 0.16],
        ...     "Error":  [0.02, 0.02, 0.015, 0.012, 0.01, 0.025, 0.02, 0.016, 0.013, 0.011],
        ... })

    output_path : str
        Path to save the figure (e.g., 'plots/kld_vs_trials.pdf').
        Parent directories will be created if needed.

    x_scale, y_scale : {'linear', 'log'}, optional
        Axis scale types. Note that for log scales, data must be positive on that axis.

    show_markers : bool, optional
        If True, scatters markers on the lines; otherwise lines only.

    error_type : {'SD', 'SE'}, optional
        Controls **how** the error is **presented** (shaded band vs. error bars)
        and **how it is labeled** in the legend. The values in `df_plot['Error']`
        must **already** be SD if 'SD', or SE if 'SE'.
        - 'SD': Draw a solid line and a shaded ±SD band (preferred for dense lines).
        - 'SE': Draw a line + error bars with caps and a legend suffix "± SE".

        This function does **not** convert SD <-> SE. If you need SE from SD,
        compute it before calling (e.g., `SE = SD / sqrt(n)`), and pass `error_type='SE'`.

    title, xlabel, ylabel : str, optional
        Title and axis labels.

    method_properties : dict[str, dict[str, str]] or None, optional
        A mapping from method name to visual properties. If None, a default
        mapping is created via `build_method_properties(...)`.

        Each entry should contain:
          - "color": matplotlib-compatible color (e.g., '#1f77b4' or 'black')
          - "marker": matplotlib marker string (e.g., 'o', 's', 'x')
          - "label": label to show in the legend

        Example:
        >>> method_properties = {
        ...     "DLVM_DALE": {"color": "#1f77b4", "marker": "s", "label": "DLVM + DALE"},
        ...     "IMLE_TB":   {"color": "black",   "marker": "o", "label": "IMLE + TB"},
        ...     "DLVM_TB":   {"color": "#ff7f0e", "marker": "s", "label": "DLVM + TB"},
        ... }

    xlim, ylim : (float, float) or None, optional
        Manual axis limits. If None, smart auto limits are applied:
        >>> xlim = (1, 100)     # x in [1, 100]
        >>> ylim = (1e-2, 1.0)  # y in [0.01, 1.0]
        Use None for auto behavior.

    x_margin, y_margin : float, optional
        Fractional margins used by auto limit computation:
        - Linear: fraction of data span to pad on each side.
        - Log: fraction of log10 span. Default (x_margin=0.05, y_margin=0.08).

    min_decades : float, optional
        Minimum margin in log10 units to keep on log scales (prevents overly
        tight bounds in log space). Default: 0.15.

    Returns
    -------
    None
        The figure is saved to `output_path`.

    Examples
    --------
    1) Auto limits on log y-axis with shaded ±SD:
    >>> styles = build_method_properties(sorted(df_plot["Method"].unique()))
    >>> plot_generic_comparison(
    ...     df_plot=df_plot,
    ...     output_path="kld_vs_trials.pdf",
    ...     x_scale="linear",
    ...     y_scale="log",
    ...     show_markers=True,
    ...     error_type="SD",
    ...     title="KLD vs Trials",
    ...     xlabel="Number of trials",
    ...     ylabel="KLD",
    ...     method_properties=styles,
    ...     xlim=None, ylim=None,
    ... )

    2) Manual y limits on linear y-axis; lines only; error bars as SE:
    >>> plot_generic_comparison(
    ...     df_plot=df_plot.assign(Error=lambda d: d["Error"] / np.sqrt(10)),  # convert to SE externally
    ...     output_path="rmse_vs_trials.pdf",
    ...     x_scale="linear",
    ...     y_scale="linear",
    ...     show_markers=False,
    ...     error_type="SE",
    ...     title="RMSE vs Trials",
    ...     xlabel="Number of trials",
    ...     ylabel="RMSE",
    ...     method_properties=styles,
    ...     ylim=(0.05, 0.40),  # manual override
    ... )
    """
    if df_plot.empty:
        raise ValueError("df_plot is empty; nothing to plot.")

    required_cols = {"Method", "N", "Value", "Error"}
    missing = required_cols.difference(df_plot.columns)
    if missing:
        raise ValueError(f"df_plot is missing required columns: {sorted(missing)}")

    all_methods = sorted(df_plot['Method'].unique())
    # If a legend order is provided, use it to order plotting (and thus legend)
    if legend_order:
        # Keep only valid methods; append any not specified to the end in original order
        specified = [m for m in legend_order if m in all_methods]
        remaining = [m for m in all_methods if m not in specified]
        all_methods = specified + remaining
    all_n_values = np.sort(df_plot['N'].unique())

    if method_properties is None:
        method_properties = build_method_properties(all_methods)

    fig, ax = plt.subplots(figsize=fig_size)
    if show_grid:
        sns.set_theme(style="whitegrid")
    else:
        sns.set_theme(style="white")

    # Ensure axis box (spines) is solid black for consistency across backends
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_edgecolor('black')
        ax.spines[side].set_linewidth(1.0)
    # Keep tick marks/labels black
    ax.tick_params(colors='black')

    # Apply axis scales (matplotlib manages ticks appropriately)
    if x_scale == 'log':
        ax.set_xscale('log')
    if y_scale == 'log':
        ax.set_yscale('log')

    # Draw each method
    for method in all_methods:
        md = df_plot[df_plot['Method'] == method].sort_values('N')
        if md.empty:
            continue

        x = md['N'].values
        y = md['Value'].values
        err = md['Error'].values  # Must already be SD or SE, matching error_type

        props = method_properties.get(method, {"color": "gray", "marker": "x", "label": method})
        linestyle = props.get("linestyle", "-")

        if show_markers:
            if error_type.upper() == 'SD':
                ax.plot(x, y, '-', color=props['color'], label=props['label'], linewidth=line_width, linestyle=linestyle)
                ax.scatter(x, y, marker=props['marker'], s=40,
                           color=props['color'], edgecolors='black', linewidths=1.0)
                ax.fill_between(x, y - err, y + err, color=props['color'], alpha=props.get('sd_fill_alpha', 0.20))
            else:  # 'SE'
                ax.errorbar(x, y, yerr=err, fmt='-', linestyle=linestyle,
                            color=props['color'], ecolor='lightgray', elinewidth=line_width,
                            capsize=5, capthick=line_width, linewidth=line_width,
                            label=f"{props['label']} ± SE")
                ax.scatter(x, y, marker=props['marker'], s=40,
                           color=props['color'], edgecolors='black', linewidths=1.0)
        else:
            if error_type.upper() == 'SD':
                ax.plot(x, y, '-', color=props['color'], label=props['label'], linewidth=line_width, linestyle=linestyle)
                ax.fill_between(x, y - err, y + err, color=props['color'], alpha=props.get('sd_fill_alpha', 0.20))
            else:  # 'SE'
                ax.errorbar(x, y, yerr=err, fmt='-', linestyle=linestyle,
                            color=props['color'], ecolor='lightgray', elinewidth=line_width,
                            capsize=5, capthick=line_width, linewidth=line_width,
                            label=f"{props['label']} ± SE")

    ax.set_title(title, fontsize=title_fontsize, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    # Auto or manual axis limits
    # Compute base x limits (auto or manual)
    if xlim is None:
        base_xlo, base_xhi = compute_axis_limits(
            df_plot['N'].values, scale=x_scale, margin=x_margin, min_decades=min_decades
        )
    else:
        base_xlo, base_xhi = xlim

    # Apply start-side padding on x
    if x_scale == 'log':
        # Expand lower bound in log10 space by a fraction of current log-span
        if base_xlo > 0 and base_xhi > 0 and x_start_pad_ratio > 0:
            log_lo, log_hi = np.log10(base_xlo), np.log10(base_xhi)
            span = max(log_hi - log_lo, 1e-12)
            d = span * x_start_pad_ratio
            base_xlo = 10 ** (log_lo - d)
    else:
        if x_start_pad_ratio > 0:
            span = max(base_xhi - base_xlo, 1e-12)
            base_xlo = base_xlo - span * x_start_pad_ratio
            # If data are strictly nonnegative, keep lower bound at 0 unless explicitly allowed
            if not allow_negative_x_pad:
                if (np.nanmin(df_plot['N'].values) >= 0) and (base_xlo < 0):
                    base_xlo = 0.0
    ax.set_xlim(base_xlo, base_xhi)

    # Compute base y limits (auto or manual)
    if ylim is None:
        base_ylo, base_yhi = compute_axis_limits(
            df_plot['Value'].values, scale=y_scale, margin=y_margin, min_decades=min_decades
        )
    else:
        base_ylo, base_yhi = ylim

    # Apply start-side padding on y
    if y_scale == 'log':
        if base_ylo > 0 and base_yhi > 0 and y_start_pad_ratio > 0:
            log_lo, log_hi = np.log10(base_ylo), np.log10(base_yhi)
            span = max(log_hi - log_lo, 1e-12)
            d = span * y_start_pad_ratio
            base_ylo = 10 ** (log_lo - d)
    else:
        if y_start_pad_ratio > 0:
            span = max(base_yhi - base_ylo, 1e-12)
            base_ylo = base_ylo - span * y_start_pad_ratio
            if not allow_negative_y_pad:
                if (np.nanmin(df_plot['Value'].values) >= 0) and (base_ylo < 0):
                    base_ylo = 0.0
    ax.set_ylim(base_ylo, base_yhi)

    # Ticks
    if x_scale == 'log':
        if x_ticks is None:
            # Log x: show ticks only at 1 and multiples of 10; no minor ticks
            tick_vals = [int(v) for v in all_n_values if (int(v) == 1 or int(v) % 10 == 0)]
        else:
            tick_vals = x_ticks
        if tick_vals:
            ax.xaxis.set_major_locator(mticker.FixedLocator(tick_vals))
            ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xticks([], minor=True)
    else:
        # Linear x: if step specified, show ticks every `x_tick_step`
        if x_tick_step is not None and x_tick_step > 0:
            lo, hi = ax.get_xlim()
            # compute start at the first multiple of step >= lo
            import math
            start = math.ceil(lo / x_tick_step) * x_tick_step
            # Guarantee inclusion of 0 when appropriate
            if lo <= 0 <= hi:
                start = min(0.0, start)
            if x_ticks is None:
                ticks = np.arange(start, hi + 1e-9, x_tick_step)
            else:
                ticks = x_ticks # override the default ticks
            ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
        else:
            if x_ticks is None:
                tick_vals = [int(v) for v in all_n_values if (int(v) == 1 or int(v) % 10 == 0)]
            else:
                tick_vals = x_ticks # override the default ticks if x_ticks are provided
            if tick_vals:
                ax.xaxis.set_major_locator(mticker.FixedLocator(tick_vals))
                ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        # Disable minor ticks on linear x
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xticks([], minor=True)

    # Y-axis tick formatting
    if y_scale == 'log':
        # Major ticks at powers of 10; labels shown as plain numbers (0.01, 0.1, 1, 10, ...)
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.set_yticks([], minor=True)

    # Ensure ticks (major + minor) are visible consistently across backends
    # Show only major ticks (no minor ticks) on both axes
    ax.minorticks_off()
    # Defensive: zero-length minor ticks even if a backend re-enables them
    ax.tick_params(axis='x', which='minor', length=0)
    ax.tick_params(axis='y', which='minor', length=0)
    ax.tick_params(axis='x', which='major', labelsize=tick_fontsize,
                   length=7, width=1.0, direction='out', bottom=True, top=False)
    ax.tick_params(axis='y', which='major', labelsize=tick_fontsize,
                   length=7, width=1.0, direction='out', left=True, right=False)
    # Disable minor tick locators explicitly (important for log y)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)
    # Explicitly set tick positions to avoid backend quirks
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if show_grid:
        ax.grid(True, linestyle='--', linewidth=0.5, which='major')
    ax.legend(fontsize=legend_fontsize, loc='upper right', ncol=legend_ncol)
    fig.tight_layout()

    # Save and/or return
    if output_path is not None:
        import os
        from pathlib import Path as _Path
        _Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
        # Use tight bbox to ensure ticks/labels are not clipped; consistent for PNG/PDF
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='black')
        print(f"Plot saved to: {output_path}")

    if return_fig:
        return fig
    else:
        plt.close(fig)
        return None
