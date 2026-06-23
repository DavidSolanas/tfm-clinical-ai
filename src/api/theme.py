"""Custom Gradio theme + CSS for the clinical evidence-synthesis console (plan §5).

Direction (deliberately off the three AI-default looks):
a calm **clinical-paper / lab-document** surface — warm-neutral paper, deep ink
text, a single restrained diagnostic teal for structure/interaction, and an
oxblood/amber reserved *exclusively* for hallucination + abstention alerts so a
"warning" colour reads as a trust signal. Monospace is used for the data layer
(PMIDs, scores, metadata) to set it apart from prose; a characterful serif is
used sparingly for headings (journal-masthead feel).

Ledger CSS classes are namespaced ``evl-*`` to keep specificity disciplined and
avoid cancelling Gradio's own rules (the frontend skill's warning).
"""

from __future__ import annotations

import gradio as gr
from gradio.themes.utils import colors, fonts, sizes

# Restrained diagnostic teal as the single interactive/structural accent.
_TEAL = colors.Color(
    name="clinical_teal",
    c50="#e9f4f3",
    c100="#cce6e4",
    c200="#a6d3d0",
    c300="#73b8b4",
    c400="#3f9591",
    c500="#0f6e6e",
    c600="#0c5c5c",
    c700="#0a4b4b",
    c800="#073838",
    c900="#052828",
    c950="#031c1c",
)


def build_theme() -> gr.themes.Base:
    """Return the clinical-paper Gradio theme."""
    theme = gr.themes.Base(
        primary_hue=_TEAL,
        secondary_hue=colors.stone,
        neutral_hue=colors.stone,
        text_size=sizes.text_md,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        font=(
            fonts.GoogleFont("Spline Sans"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ),
        font_mono=(
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    )
    theme.set(
        body_background_fill="#f5f2ec",
        body_background_fill_dark="#16191a",
        body_text_color="#1f2421",
        background_fill_primary="#fbf9f4",
        background_fill_secondary="#f0ece3",
        block_background_fill="#fbf9f4",
        block_border_color="#ddd6c8",
        block_border_width="1px",
        block_radius="10px",
        block_label_text_weight="600",
        button_primary_background_fill="#0f6e6e",
        button_primary_background_fill_hover="#0c5c5c",
        button_primary_text_color="#ffffff",
        button_large_radius="8px",
        input_background_fill="#ffffff",
        input_border_color="#ccc4b2",
    )
    return theme


# Heading serif (journal-masthead feel) loaded via @import so it is available
# without adding it to the theme's body font stack.
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');

:root {
  --evl-ink: #1f2421;
  --evl-muted: #5e6660;
  --evl-paper: #fbf9f4;
  --evl-line: #ddd6c8;
  --evl-teal: #0f6e6e;
  --evl-teal-soft: #e9f4f3;
  --evl-oxblood: #8a2c20;
  --evl-oxblood-soft: #f6e7e3;
  --evl-amber: #9a6b00;
  --evl-amber-soft: #f6edd6;
  --evl-serif: 'Fraunces', Georgia, 'Times New Roman', serif;
  --evl-mono: 'IBM Plex Mono', ui-monospace, monospace;
}

/* ----- Masthead ------------------------------------------------------- */
.evl-masthead { padding: 4px 2px 10px; border-bottom: 2px solid var(--evl-ink); margin-bottom: 6px; }
.evl-masthead h1 {
  font-family: var(--evl-serif); font-weight: 600; font-size: 1.9rem;
  letter-spacing: -0.01em; color: var(--evl-ink); margin: 0;
}
.evl-masthead p { color: var(--evl-muted); font-size: 0.92rem; margin: 4px 0 0; max-width: 64ch; }

/* ----- Service health banner ----------------------------------------- */
.evl-health { font-size: 0.86rem; padding: 8px 12px; border-radius: 8px; border: 1px solid transparent; }
.evl-health--ok { background: var(--evl-teal-soft); border-color: #bfe0dd; color: var(--evl-teal); }
.evl-health--down { background: var(--evl-oxblood-soft); border-color: #e6c5bf; color: var(--evl-oxblood); }
.evl-health code { font-family: var(--evl-mono); font-size: 0.82rem; }

/* ----- Answer panel: 3-part contract --------------------------------- */
.evl-answer { font-size: 0.97rem; line-height: 1.62; color: var(--evl-ink); white-space: pre-wrap; }
.evl-answer .evl-cite {
  font-family: var(--evl-mono); font-size: 0.84em; color: var(--evl-teal);
  text-decoration: none; border-bottom: 1px solid currentColor; padding-bottom: 1px;
}
.evl-answer .evl-cite:hover { background: var(--evl-teal-soft); }

/* ----- Citation badge ------------------------------------------------- */
.evl-badge {
  display: flex; align-items: center; gap: 8px; font-size: 0.88rem;
  padding: 8px 12px; border-radius: 8px; margin-bottom: 10px; font-weight: 500;
}
.evl-badge__icon { font-weight: 700; }
.evl-badge--ok { background: var(--evl-teal-soft); color: var(--evl-teal); border: 1px solid #bfe0dd; }
.evl-badge--alert { background: var(--evl-oxblood-soft); color: var(--evl-oxblood); border: 1px solid #e6c5bf; }
.evl-badge .evl-pmid { color: inherit; }

/* ----- Evidence Ledger ------------------------------------------------ */
.evl-ledger { display: flex; flex-direction: column; gap: 10px; }
.evl-card {
  background: var(--evl-paper); border: 1px solid var(--evl-line);
  border-left: 3px solid var(--evl-line); border-radius: 8px; padding: 10px 12px;
}
.evl-card--cited { border-left-color: var(--evl-teal); }
.evl-card--hallucinated { border-left-color: var(--evl-oxblood); }
.evl-card__head { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.evl-rank {
  font-family: var(--evl-mono); font-size: 0.74rem; font-weight: 600; color: #fff;
  background: var(--evl-ink); border-radius: 4px; padding: 1px 6px; min-width: 1.2em; text-align: center;
}
.evl-pmid { font-family: var(--evl-mono); font-size: 0.82rem; color: var(--evl-teal); text-decoration: none; }
.evl-pmid:hover { text-decoration: underline; }
.evl-chip {
  margin-left: auto; font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.04em; padding: 2px 8px; border-radius: 999px;
}
.evl-chip--cited { background: var(--evl-teal); color: #fff; }
.evl-chip--available { background: #ece7db; color: var(--evl-muted); }
.evl-chip--hallucinated { background: var(--evl-oxblood); color: #fff; }
.evl-card__title { font-size: 0.9rem; font-weight: 500; line-height: 1.4; margin: 6px 0; color: var(--evl-ink); }
.evl-card__foot { display: flex; align-items: center; gap: 10px; }
.evl-year { font-family: var(--evl-mono); font-size: 0.8rem; color: var(--evl-muted); }
.evl-meter {
  flex: 1; height: 5px; background: #e6e0d3; border-radius: 999px; overflow: hidden; max-width: 160px;
}
.evl-meter > span { display: block; height: 100%; background: var(--evl-teal); }
.evl-score { font-family: var(--evl-mono); font-size: 0.78rem; color: var(--evl-muted); }
.evl-empty { color: var(--evl-muted); font-size: 0.9rem; padding: 16px; text-align: center; font-style: italic; }

/* ----- Abstention (non-error, deliberate) ----------------------------- */
.evl-abstain {
  background: var(--evl-amber-soft); border: 1px solid #e3d6ac; border-left: 3px solid var(--evl-amber);
  border-radius: 8px; padding: 14px 16px;
}
.evl-abstain__title { font-family: var(--evl-serif); font-weight: 600; font-size: 1.05rem; color: var(--evl-amber); }
.evl-abstain__body { color: var(--evl-ink); font-size: 0.92rem; margin: 6px 0; }
.evl-abstain__note { color: var(--evl-muted); font-size: 0.84rem; margin: 0; }

/* ----- Section labels ------------------------------------------------- */
.evl-section-label {
  font-family: var(--evl-mono); font-size: 0.74rem; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--evl-muted); margin: 0 0 6px;
}

/* ----- Loading / thinking state -------------------------------------- */
.evl-loading { display: flex; align-items: center; gap: 10px; color: var(--evl-muted); font-size: 0.9rem; }
.evl-loading__dot {
  width: 9px; height: 9px; border-radius: 50%; background: var(--evl-teal); animation: evl-pulse 1.2s ease-in-out infinite;
}
@keyframes evl-pulse { 0%,100% { opacity: 0.3; } 50% { opacity: 1; } }

@media (prefers-reduced-motion: reduce) {
  .evl-loading__dot { animation: none; opacity: 0.7; }
}

/* Keyboard focus visibility (quality floor). */
.evl-cite:focus-visible, .evl-pmid:focus-visible { outline: 2px solid var(--evl-teal); outline-offset: 2px; border-radius: 2px; }
"""
