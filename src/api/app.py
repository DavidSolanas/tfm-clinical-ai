"""Gradio Blocks UI for the Hybrid Clinical Assistance System (plan §4–§5).

Layout only - all business logic lives in :class:`~src.api.service.ClinicalAssistantService`
and all rendering in :mod:`src.api.formatting`. The page is the clinician
"evidence synthesis console": a patient note in, a structured 3-part answer with
linkified PMID citations out, and the Evidence Ledger beside it as the signature
element. The app always runs the flagship configuration (fine-tuned + RAG).

Run with ``scripts/run_app.sh`` (sets the CPU/GPU env split), or directly::

    CUDA_VISIBLE_DEVICES="" uv run python -m src.api.app
"""

from __future__ import annotations

import gradio as gr

from src.api.examples import EXAMPLE_NOTES
from src.api.formatting import (
    render_abstention,
    render_answer,
    render_citation_badge,
    render_evidence_ledger,
)
from src.api.service import ClinicalAssistantService
from src.api.theme import CSS, build_theme
from src.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# The app always runs the flagship configuration (fine-tuned + RAG).
_CONFIG = "D"

_LEDGER_PLACEHOLDER = (
    '<div class="evl-empty">Run an analysis to populate the evidence ledger.</div>'
)


def _health_banner_html(service: ClinicalAssistantService) -> str:
    health = service.health()
    if health.ok:
        fam = {"base": "base", "ft": "fine-tuned"}.get(health.model_family, "-")
        return (
            '<div class="evl-health evl-health--ok">'
            f"Services online · model loaded: <strong>{fam}</strong> · "
            "Qdrant connected</div>"
        )
    return f'<div class="evl-health evl-health--down">{health.detail}</div>'


def _run_analysis(note: str, service: ClinicalAssistantService):
    """Event handler: run the flagship config and return updates for every output slot."""
    result = service.answer(note, _CONFIG)
    cfg_tag = f"{result.config} · {result.config_label}"

    # Error → surface in the answer slot, clear the ledger.
    if result.error:
        answer_html = f'<div class="evl-health evl-health--down">{result.error}</div>'
        return (
            answer_html,
            "",
            _LEDGER_PLACEHOLDER,
            _health_banner_html(service),
            f"Config {cfg_tag} · failed in {result.wall_time_s}s",
        )

    # Abstention → deliberate non-error state; show top candidates in the ledger.
    if result.abstained:
        answer_html = render_abstention(result.reason)
        ledger_html = (
            render_evidence_ledger(result.docs, None) if result.docs else _LEDGER_PLACEHOLDER
        )
        return (
            answer_html,
            "",
            ledger_html,
            _health_banner_html(service),
            f"Config {cfg_tag} · abstained · {result.wall_time_s}s",
        )

    answer_html = render_answer(result.response)
    badge_html = render_citation_badge(result.citation_check)
    ledger_html = render_evidence_ledger(result.docs, result.citation_check)
    avg = result.avg_final_score
    footer = (
        f"Config {cfg_tag} · {len(result.docs)} sources · "
        f"avg score {avg:.2f} · {result.wall_time_s}s"
        if avg is not None
        else f"Config {cfg_tag} · {result.wall_time_s}s"
    )

    return answer_html, badge_html, ledger_html, _health_banner_html(service), footer


def build_app(service: ClinicalAssistantService | None = None) -> gr.Blocks:
    """Construct the Gradio Blocks app (does not launch it).

    Args:
        service: A ready service. If ``None``, one is constructed (but not
            pre-warmed - the caller should ``prewarm()`` before launch).

    Returns:
        A ``gr.Blocks`` instance.
    """
    service = service or ClinicalAssistantService()

    # In Gradio 6, theme/css moved from the Blocks constructor to launch().
    with gr.Blocks(title="Clinical Evidence Console") as demo:
        gr.HTML(
            '<div class="evl-masthead">'
            "<h1>Clinical Evidence Console</h1>"
            "<p>Turn a patient note into a traceable, evidence-based recommendation. "
            "Every claim links back to a real PubMed source, and the system declines "
            "when the evidence isn&rsquo;t there.</p>"
            "</div>"
        )

        health_banner = gr.HTML(_health_banner_html(service))

        with gr.Row(equal_height=False):
            # --- Left column: input + answer ---
            with gr.Column(scale=7):
                gr.HTML('<p class="evl-section-label">Patient note</p>')
                note_input = gr.Textbox(
                    label="",
                    placeholder="Paste a de-identified clinical note or transcription…",
                    lines=8,
                    autofocus=True,
                )
                gr.Examples(
                    examples=EXAMPLE_NOTES,
                    inputs=note_input,
                    label="Example notes",
                )
                analyze_btn = gr.Button("Analyze evidence", variant="primary", size="lg")

                gr.HTML('<p class="evl-section-label">Recommendation</p>')
                answer_panel = gr.HTML(
                    '<div class="evl-empty">Enter a note, '
                    "then press <strong>Analyze evidence</strong>.</div>"
                )
                status_footer = gr.Markdown("", elem_classes=["evl-section-label"])

            # --- Right column: the Evidence Ledger ---
            with gr.Column(scale=5):
                gr.HTML('<p class="evl-section-label">Citation check</p>')
                citation_badge = gr.HTML("")
                gr.HTML('<p class="evl-section-label">Evidence ledger</p>')
                evidence_ledger = gr.HTML(_LEDGER_PLACEHOLDER)

        outputs = [answer_panel, citation_badge, evidence_ledger, health_banner, status_footer]

        def _handler(note):
            return _run_analysis(note, service)

        analyze_btn.click(
            fn=lambda: (
                '<div class="evl-loading"><span class="evl-loading__dot"></span>'
                "Retrieving evidence and synthesizing&hellip; this can take 30&ndash;60s.</div>",
                "",
                _LEDGER_PLACEHOLDER,
            ),
            inputs=None,
            outputs=[answer_panel, citation_badge, evidence_ledger],
            queue=False,
        ).then(
            fn=_handler,
            inputs=[note_input],
            outputs=outputs,
        )

    return demo


def main() -> None:
    """Entry point: build the service, pre-warm, and launch the app."""
    setup_logging()
    service = ClinicalAssistantService()
    try:
        service.prewarm()
    except Exception:  # noqa: BLE001 (launch UI anyway; banner shows the failure)
        logger.exception("Pre-warm failed - launching UI in degraded state")
    demo = build_app(service)
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            theme=build_theme(),
            css=CSS,
        )
    finally:
        service.shutdown()


if __name__ == "__main__":
    main()
