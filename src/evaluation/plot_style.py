"""Shared, publication-quality plotting style for the evaluation figures.

Single source of truth so the rollup figures (logged by ``mlflow_logging``) and
the inspection notebook (``06_evaluation``) share one consistent, thesis-ready
look. The palette encodes the two ablation axes at a glance:

* **hue** = model  base is blue, fine-tuned is orange;
* **shade** = retrieval  RAG configs are deep, no-RAG configs are light.

So A/B (no-RAG) read as the light pair and C/D (RAG) as the deep pair, while the
blue→orange contrast tracks the fine-tuning effect. The luminance gap between
light/deep and blue/orange also keeps the figures legible in grayscale print.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Canonical config order; A/B (no-RAG) then C/D (RAG) for the visual pairing.
CONFIG_ORDER = ("A_base", "B_finetuned", "C_base_rag", "D_finetuned_rag")

# Short, thesis-friendly axis labels.
CONFIG_LABELS = {
    "A_base": "A · Base",
    "B_finetuned": "B · FT",
    "C_base_rag": "C · Base+RAG",
    "D_finetuned_rag": "D · FT+RAG",
}

# hue = model (base=blue, FT=orange); shade = RAG (deep) vs no-RAG (light).
CONFIG_COLORS = {
    "A_base": "#9CB8D6",          # base, no-RAG   light blue
    "B_finetuned": "#F1C39C",     # FT,   no-RAG   light orange
    "C_base_rag": "#34699A",      # base, RAG      deep blue
    "D_finetuned_rag": "#D2752B",  # FT,   RAG      deep orange
}

# Human-readable titles; falls back to a prettified key.
METRIC_TITLES = {
    "hallucinated_pmids_rate": "Hallucinated PMID rate",
    "format_adherence_among_answered": "Format adherence (answered)",
    "abstention_rate": "Abstention rate",
    "error_rate": "Error rate",
    "pmid_citation_rate": "PMID citation rate",
    "bracket_citation_rate": "Bracket-reference rate",
    "placeholder_citation_rate": "Placeholder citation rate",
    "no_citation_rate": "No-citation rate",
    "citation_grounding_precision": "Citation grounding precision",
    "pmid_citations_per_answered_mean": "PMID citations per answer (mean)",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
}

_INK = "#2B2B2B"
_MUTED = "#8A8A8A"
_GRID = "#D5D9DE"

# Neutral accents for non-categorical plots (histograms, scatter, single series).
ACCENT = "#34699A"
ACCENT_WARM = "#D2752B"


def apply_style() -> None:
    """Apply the shared rcParams. Idempotent; safe to call once per session."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12.5,
        "axes.titleweight": "semibold",
        "axes.titlepad": 10,
        "axes.labelsize": 10.5,
        "axes.labelcolor": _INK,
        "axes.edgecolor": "#AAB0B6",
        "axes.linewidth": 0.8,
        "axes.axisbelow": True,
        "grid.color": _GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": _INK,
        "ytick.color": _INK,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.color": _INK,
        "legend.frameon": False,
        "legend.fontsize": 10,
    })


def metric_title(key: str) -> str:
    """Human-readable title for a metric key."""
    return METRIC_TITLES.get(key, key.replace("_", " ").capitalize())


def colors_for(configs: Iterable[str]) -> list[str]:
    return [CONFIG_COLORS.get(c, _MUTED) for c in configs]


def labels_for(configs: Iterable[str]) -> list[str]:
    return [CONFIG_LABELS.get(c, c) for c in configs]


def style_axes(ax) -> None:
    """Despine top/right and keep a light horizontal-only grid behind the data."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.7)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)


def _is_nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def annotate_bars(
    ax,
    bars,
    values: Sequence[float],
    *,
    tops: Sequence[float] | None = None,
    fmt: str = "{:.2f}",
    fontsize: float = 9,
    dy: float = 0.02,
) -> None:
    """Print value labels above each bar; NaN bars get an italic ``n/a``."""
    for i, (bar, v) in enumerate(zip(bars, values, strict=True)):
        x = bar.get_x() + bar.get_width() / 2
        if _is_nan(v):
            ax.text(x, dy, "n/a", ha="center", va="bottom",
                    fontsize=fontsize, color=_MUTED, style="italic")
            continue
        top = tops[i] if (tops is not None and not _is_nan(tops[i])) else v
        ax.text(x, top + dy, fmt.format(v), ha="center", va="bottom",
                fontsize=fontsize, color=_INK)


def grouped_bar(
    ax,
    configs: Sequence[str],
    values: Sequence[float],
    *,
    ylabel: str = "",
    title: str = "",
    ylim: tuple[float, float] | None = (0, 1.08),
    ci_low: Sequence[float] | None = None,
    ci_high: Sequence[float] | None = None,
    annotate: bool = True,
    fmt: str = "{:.2f}",
):
    """The canonical per-config bar chart shared by every evaluation figure.

    NaN values render as a zero-height bar plus an ``n/a`` label so missing
    configs (e.g. RAGAS faithfulness for the no-RAG A/B) stay visible.
    """
    xs = range(len(configs))
    yerr = None
    tops = list(values)
    if ci_low is not None and ci_high is not None:
        lower = [m - lo if not (_is_nan(lo) or _is_nan(m)) else 0.0
                 for m, lo in zip(values, ci_low, strict=True)]
        upper = [hi - m if not (_is_nan(hi) or _is_nan(m)) else 0.0
                 for m, hi in zip(values, ci_high, strict=True)]
        yerr = [lower, upper]
        tops = [hi if not _is_nan(hi) else v for hi, v in zip(ci_high, values, strict=True)]

    plot_vals = [0.0 if _is_nan(v) else v for v in values]
    bars = ax.bar(
        list(xs), plot_vals, width=0.68,
        color=colors_for(configs), edgecolor="white", linewidth=1.3,
        yerr=yerr, capsize=4, ecolor="#5A5A5A", error_kw={"elinewidth": 1.2},
    )
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels_for(configs))
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    style_axes(ax)
    if annotate:
        annotate_bars(ax, bars, values, tops=tops, fmt=fmt)
    return bars


def config_legend(fig, configs: Sequence[str] = CONFIG_ORDER, *, ncol: int = 4,
                  y: float = 1.02) -> None:
    """Figure-level legend mapping each config color to its label."""
    handles = [
        Patch(facecolor=CONFIG_COLORS[c], edgecolor="white", label=CONFIG_LABELS[c])
        for c in configs
    ]
    fig.legend(handles=handles, loc="upper center", ncol=ncol,
               bbox_to_anchor=(0.5, y))
