"""Short de-identified clinical vignettes for instant demo (plan §6.6).

Five concise notes spanning specialties. The last one is deliberately off-corpus
(a non-clinical query) to demonstrate the abstention safety state in a RAG
config: it reliably trips the retrieval gate (< 2 docs >= 0.35 final_score) and
renders the "No sufficient evidence" panel. Verified live 2026-06-19 — note that
medical-sounding-but-empty text (e.g. an administrative note) does NOT abstain,
because its tokens still retrieve clinical abstracts; a clearly non-medical query
is the honest way to demo the gate. These are synthetic; no real patient data.
"""

from __future__ import annotations

EXAMPLE_NOTES: list[list[str]] = [
    [
        "A 58-year-old patient with a 6-year history of type 2 diabetes mellitus presents "
        "for routine follow-up. HbA1c is 8.4% despite metformin 1000 mg twice daily. BMI 31. "
        "Blood pressure 138/86. No history of cardiovascular disease. The patient reports "
        "occasional polyuria and fatigue. What additional glucose-lowering therapy is "
        "appropriate given the cardiovascular and weight considerations?"
    ],
    [
        "A 67-year-old presents with acute onset of left-sided weakness and slurred speech "
        "that began 90 minutes ago. NIH Stroke Scale is 11. CT head shows no hemorrhage. "
        "Blood pressure 172/94, glucose 132 mg/dL. No recent surgery or anticoagulant use. "
        "What is the recommended acute reperfusion management for this presentation?"
    ],
    [
        "A 45-year-old with newly diagnosed major depressive disorder, moderate severity, "
        "with no psychotic features and no prior psychiatric history. The patient is "
        "treatment-naive and prefers an oral medication with a favorable side-effect profile. "
        "What first-line pharmacologic treatment is supported by current evidence?"
    ],
    [
        "A 3-year-old child presents with a 2-day history of fever to 39.1°C, ear pain, and "
        "irritability. Otoscopy shows a bulging, erythematous tympanic membrane with reduced "
        "mobility, consistent with acute otitis media. No drug allergies. What is the "
        "evidence-based first-line antibiotic management?"
    ],
    [
        # Off-corpus on purpose: trips the retrieval abstention gate so the demo
        # shows the deliberate "No sufficient evidence" safety state (verified live).
        "My car makes a grinding noise when I brake and the steering wheel vibrates at "
        "highway speed. How do I replace the brake rotors and pads myself, and what "
        "torque should I use on the caliper bolts?"
    ],
]
