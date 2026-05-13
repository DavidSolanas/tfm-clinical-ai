"""PubMed bulk ingestion for the clinical RAG corpus.

Drives ingestion off the extended MeSH taxonomy (C/D/E/F/G/A/B/H/N trees)
using ``[majr]`` (MeSH Major Topic) queries with a recency filter and a
publication-type blacklist. Applies weighted per-stratum sampling so that
disease/procedure categories dominate the corpus while keeping evidence
diversity (mixes reviews, trials, case reports, observational studies).

Usage::

    uv run python scripts/pubmed_ingestion.py [options]

Options:
    --output PATH       Output JSON path          (default: data/raw/pubmed_bulk_corpus.json)
    --target INT        Total corpus size target   (default: 50000)
    --min-year INT      Recency lower bound        (default: 2015, covers post-paradigm-shift era)
    --checkpoint PATH   Checkpoint file for resume (default: <output>.ckpt.json)
    --resume            Resume from checkpoint instead of starting fresh
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError

from Bio import Entrez, Medline
from dotenv import load_dotenv

from src.logging_config import get_logger

logger = get_logger(__name__)

# ── MeSH category groups ──────────────────────────────────────────────────────

MTSAMPLES_DISEASE_CORE: list[str] = [
    # C tree: broad human clinical disease coverage
    "Cardiovascular Diseases",
    "Congenital, Hereditary, and Neonatal Diseases and Abnormalities",
    "Digestive System Diseases",
    "Endocrine System Diseases",
    "Eye Diseases",
    "Hemic and Lymphatic Diseases",
    "Immune System Diseases",
    "Infections",
    "Musculoskeletal Diseases",
    "Neoplasms",
    "Nervous System Diseases",
    "Nutritional and Metabolic Diseases",
    "Otorhinolaryngologic Diseases",
    "Pathological Conditions, Signs and Symptoms",
    "Respiratory Tract Diseases",
    "Skin and Connective Tissue Diseases",
    "Stomatognathic Diseases",
    "Urogenital Diseases",
    "Wounds and Injuries",
    # Important C sub-branches for clinical notes
    "Bacterial Infections and Mycoses",
    "Virus Diseases",
    "Parasitic Diseases",
    "Respiratory Tract Infections",
    "Urinary Tract Infections",
    "Sexually Transmitted Diseases",
    "Sepsis",
    "Soft Tissue Infections",
    "Wound Infection",
    "Zoonoses",
    # Useful but often missing from MTSamples-focused lists
    "Chemically-Induced Disorders",
    "Drug-Related Side Effects and Adverse Reactions",
    "Poisoning",
    "Substance-Related Disorders",
    "Disorders of Environmental Origin",
    "Occupational Diseases",
    "Pregnancy Complications",
    "Infant, Newborn, Diseases",
]

MTSAMPLES_PROCEDURE_DIAGNOSTIC_CORE: list[str] = [
    # E tree: diagnostics
    "Diagnosis",
    "Diagnostic Techniques and Procedures",
    "Clinical Decision-Making",
    "Diagnosis, Differential",
    "Diagnostic Errors",
    "Prognosis",
    "Diagnostic Imaging",
    "Radiography",
    "Tomography, X-Ray Computed",
    "Magnetic Resonance Imaging",
    "Ultrasonography",
    "Endoscopy",
    "Electrocardiography",
    "Electroencephalography",
    "Clinical Laboratory Techniques",
    "Cytological Techniques",
    "Biopsy",
    "Autopsy",
    "Neuroimaging",
    # E tree: surgery
    "Surgical Procedures, Operative",
    "Ambulatory Surgical Procedures",
    "Cardiovascular Surgical Procedures",
    "Digestive System Surgical Procedures",
    "Endocrine Surgical Procedures",
    "Minimally Invasive Surgical Procedures",
    "Neurosurgical Procedures",
    "Obstetric Surgical Procedures",
    "Ophthalmologic Surgical Procedures",
    "Oral Surgical Procedures",
    "Orthopedic Procedures",
    "Otorhinolaryngologic Surgical Procedures",
    "Plastic Surgery Procedures",
    "Thoracic Surgical Procedures",
    "Transplantation",
    "Urogenital Surgical Procedures",
    "Wound Closure Techniques",
    "Perioperative Care",
    # E tree: therapeutics
    "Therapeutics",
    "Drug Therapy",
    "Emergency Treatment",
    "Patient Care",
    "Pain Management",
    "Anesthesia and Analgesia",
    "Airway Management",
    "Respiratory Therapy",
    "Radiotherapy",
    "Rehabilitation",
    "Physical Therapy Modalities",
    "Renal Replacement Therapy",
    "Nutrition Therapy",
    "Psychotherapy",
    "Psychiatric Somatic Therapies",
    "Biological Therapy",
    "Precision Medicine",
    "Genomic Medicine",
]

ANATOMY_AND_SYSTEMS: list[str] = [
    # A tree: useful for mapping notes to body systems
    "Body Regions",
    "Cardiovascular System",
    "Digestive System",
    "Endocrine System",
    "Hemic and Immune Systems",
    "Integumentary System",
    "Musculoskeletal System",
    "Nervous System",
    "Respiratory System",
    "Sense Organs",
    "Stomatognathic System",
    "Urogenital System",
    "Tissues",
    "Cells",  # broad; may return basic cell-biology papers
    "Embryonic Structures",
    "Fluids and Secretions",
]

ORGANISMS_AND_MICROBIOLOGY: list[str] = [
    # B tree + infection-relevant organism strata
    "Bacteria",
    "Viruses",
    "Fungi",
    "Eukaryota",  # broad; may include non-clinical basic-biology papers
    "Animals",  # broad; mostly animal-model studies
    "Archaea",
    "Blood-Borne Pathogens",
    "Organisms, Genetically Modified",
]

DRUGS_CHEMICALS_BIOMARKERS: list[str] = [
    # D tree
    "Pharmaceutical Preparations",
    "Chemical Actions and Uses",
    "Pharmacologic Actions",
    "Toxic Actions",
    "Biomarkers",
    "Biological Factors",
    "Hormones, Hormone Substitutes, and Hormone Antagonists",
    "Enzymes and Coenzymes",
    "Amino Acids, Peptides, and Proteins",  # may pull basic biochemistry
    "Nucleic Acids, Nucleotides, and Nucleosides",
    "Carbohydrates",
    "Lipids",
    "Organic Chemicals",
    "Inorganic Chemicals",
    "Biomedical and Dental Materials",
    "Complex Mixtures",
    "Biological Products",
    "Dosage Forms",
    "Drug Combinations",
    "Prescription Drugs",
    "Nonprescription Drugs",
    "Controlled Substances",
]

PHYSIOLOGY_PATHOPHYSIOLOGY_BASIC_SCIENCE: list[str] = [
    # G tree
    "Physiological Phenomena",
    "Circulatory and Respiratory Physiological Phenomena",
    "Digestive System and Oral Physiological Phenomena",
    "Musculoskeletal and Neural Physiological Phenomena",
    "Reproductive and Urinary Physiological Phenomena",
    "Ocular Physiological Phenomena",
    "Immune System Phenomena",
    "Metabolism",
    "Pharmacological and Toxicological Phenomena",
    "Genetic Phenomena",
    "Microbiological Phenomena",
    "Microbiota",
    "Host-Pathogen Interactions",
    "Drug Resistance, Microbial",
    "Cell Physiological Phenomena",
    "Biological Phenomena",
    "Chemical Phenomena",  # may return basic-chemistry papers
    "Biochemical Phenomena",
    "Mathematical Concepts",  # mostly biostatistics / methods papers
    "Algorithms",
    "Sensitivity and Specificity",
]

PSYCHIATRY_PSYCHOLOGY: list[str] = [
    # F tree
    "Mental Disorders",
    "Anxiety Disorders",
    "Mood Disorders",
    "Neurocognitive Disorders",
    "Neurodevelopmental Disorders",
    "Schizophrenia Spectrum and Other Psychotic Disorders",
    "Substance-Related Disorders",
    "Sleep Wake Disorders",
    "Trauma and Stressor Related Disorders",
    "Behavior and Behavior Mechanisms",
    "Psychological Phenomena",
    "Behavioral Disciplines and Activities",
    "Mental Health Services",
    "Psychotherapy",
    "Psychological Tests",
]

HEALTHCARE_PUBLIC_HEALTH_SYSTEMS: list[str] = [
    # N tree
    "Public Health",
    "Environment and Public Health",
    "Delivery of Health Care",
    "Health Services",
    "Health Facilities",
    "Health Personnel",
    "Health Workforce",
    "Quality of Health Care",
    "Quality Assurance, Health Care",
    "Health Services Research",
    "Health Care Quality, Access, and Evaluation",
    "Health Care Economics and Organizations",
    "Health Services Administration",
    "Health Planning",
    "Population Characteristics",
    "Socioeconomic Factors",
    "Demography",
    "Ethics",
    "Universal Health Care",
    "Public Health Systems Research",
]

DISCIPLINES_OCCUPATIONS: list[str] = [
    # H tree
    "Medicine",
    "Clinical Medicine",
    "Surgery",
    "Nursing",
    "Dentistry",
    "Pharmacology",
    "Pharmacy",
    "Podiatry",
    "Allied Health Occupations",
    "Evidence-Based Practice",
    "Biomedical Engineering",
    "Toxicology",
    "Nutritional Sciences",
    "Psychology, Medical",
    "Natural Science Disciplines",  # broad meta-discipline
    "Biological Science Disciplines",
    "Neurosciences",
    "Genetics",
    "Molecular Biology",
    "Epidemiology",
]

# Sampling weights for each stratum group.
# PUBLICATION_TYPES is used as a query filter (blacklist), not as a query stratum.
_STRATUM_CONFIG: list[tuple[str, list[str], float]] = [
    ("disease_core", MTSAMPLES_DISEASE_CORE, 0.35),
    ("procedures", MTSAMPLES_PROCEDURE_DIAGNOSTIC_CORE, 0.25),
    ("drugs_chemicals", DRUGS_CHEMICALS_BIOMARKERS, 0.12),
    ("anatomy_physiology", ANATOMY_AND_SYSTEMS + PHYSIOLOGY_PATHOPHYSIOLOGY_BASIC_SCIENCE, 0.10),
    ("psychiatry", PSYCHIATRY_PSYCHOLOGY, 0.06),
    ("healthcare", HEALTHCARE_PUBLIC_HEALTH_SYSTEMS, 0.06),
    ("organisms", ORGANISMS_AND_MICROBIOLOGY, 0.04),
    ("disciplines", DISCIPLINES_OCCUPATIONS, 0.02),
]

# ── PubMed query construction ─────────────────────────────────────────────────

_PT_BLACKLIST = (
    "Editorial[PT] OR Letter[PT] OR Comment[PT] OR News[PT] OR "
    'Biography[PT] OR "Personal Narrative"[PT] OR '
    '"Published Erratum"[PT] OR "Retracted Publication"[PT]'
)


def _build_query(descriptor: str, min_year: int) -> str:
    return f'"{descriptor}"[majr] AND ({min_year}:3000[dp]) NOT ({_PT_BLACKLIST})'


# ── Entrez fetching ───────────────────────────────────────────────────────────

_EFETCH_BATCH = 200
_MAX_RETRIES = 4
_BACKOFF_BASE = 2.0


def _efetch_page(webenv: str, query_key: str, retstart: int, retmax: int) -> list:
    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            handle = Entrez.efetch(
                db="pubmed",
                rettype="medline",
                retmode="text",
                webenv=webenv,
                query_key=query_key,
                retstart=retstart,
                retmax=retmax,
            )
            records = list(Medline.parse(handle))
            handle.close()
            return records
        except (IncompleteRead, HTTPError, URLError, ConnectionError) as exc:
            last_err = exc
            wait = _BACKOFF_BASE * (2**attempt)
            logger.warning(
                "efetch retstart=%d failed (%s); retry %d/%d in %.1fs",
                retstart,
                type(exc).__name__,
                attempt + 1,
                _MAX_RETRIES,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"efetch failed after {_MAX_RETRIES} retries: {last_err}")


def _record_to_dict(r: dict, mesh_category: str, mesh_group: str) -> dict | None:
    abstract = r.get("AB", "")
    if not abstract:
        return None
    dp = r.get("DP", "")
    return {
        "pmid": r.get("PMID", ""),
        "title": r.get("TI", ""),
        "abstract": abstract,
        "mesh": r.get("MH", []),
        "authors": "; ".join(r.get("AU", [])),
        "author_keywords": r.get("OT", []),
        "journal": r.get("JT", ""),
        "year": dp[:4] if dp else "",
        "doi": r.get("LID", "").split()[0] if r.get("LID") else "",
        "publication_types": r.get("PT", []),
        "mesh_category": mesh_category,
        "mesh_group": mesh_group,
    }


def fetch_category(
    descriptor: str,
    mesh_group: str,
    max_results: int,
    min_year: int,
) -> list[dict]:
    query = _build_query(descriptor, min_year)
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance",
        usehistory="y",
    )
    record = Entrez.read(handle)
    handle.close()

    count = min(int(record["Count"]), max_results)
    logger.info("  [%s] %d results for %r", mesh_group, count, descriptor)
    if count == 0:
        return []

    webenv = record["WebEnv"]
    query_key = record["QueryKey"]

    raw: list = []
    for retstart in range(0, count, _EFETCH_BATCH):
        batch = _efetch_page(
            webenv=webenv,
            query_key=query_key,
            retstart=retstart,
            retmax=min(_EFETCH_BATCH, count - retstart),
        )
        raw.extend(batch)
        time.sleep(0.15)

    papers = []
    for r in raw:
        doc = _record_to_dict(r, mesh_category=descriptor, mesh_group=mesh_group)
        if doc:
            papers.append(doc)

    logger.debug("  [%s] %r → %d with abstracts", mesh_group, descriptor, len(papers))
    return papers


# ── Main ingestion orchestration ──────────────────────────────────────────────


def _save_checkpoint(checkpoint_path: Path, seen_pmids: set[str], completed: set[str]) -> None:
    tmp = checkpoint_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"seen_pmids": list(seen_pmids), "completed_categories": list(completed)}, f)
    tmp.replace(checkpoint_path)


def ingest_corpus(
    output_path: Path,
    target: int,
    min_year: int,
    checkpoint_path: Path,
    resume: bool,
) -> list[dict]:
    seen_pmids: set[str] = set()
    completed: set[str] = set()

    # On resume, reload progress state and rebuild seen_pmids from the
    # partially-written output file (one JSON object per line).
    if resume:
        if checkpoint_path.exists():
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            seen_pmids = set(ckpt["seen_pmids"])
            completed = set(ckpt["completed_categories"])
            logger.info(
                "Resuming — %d seen PMIDs, %d categories already done",
                len(seen_pmids),
                len(completed),
            )
        else:
            logger.warning("--resume given but no checkpoint found; starting fresh.")

    append_mode = resume and output_path.exists()
    out_fh = open(output_path, "a" if append_mode else "w", encoding="utf-8")
    docs_written = sum(1 for _ in open(output_path)) if append_mode else 0
    logger.info("Output file: %s (%d docs already written)", output_path, docs_written)

    try:
        for group_name, categories, weight in _STRATUM_CONFIG:
            per_cat = max(50, math.ceil(target * weight / len(categories)))
            logger.info(
                "Stratum %r — %d categories, weight=%.2f, budget=%d/cat",
                group_name,
                len(categories),
                weight,
                per_cat,
            )
            for descriptor in categories:
                cat_key = f"{group_name}::{descriptor}"
                if cat_key in completed:
                    logger.debug("  Skipping (already done): %r", descriptor)
                    continue

                try:
                    papers = fetch_category(descriptor, group_name, per_cat, min_year)
                except Exception as exc:
                    logger.error("  fetch failed for %r: %s", descriptor, exc)
                    completed.add(cat_key)
                    _save_checkpoint(checkpoint_path, seen_pmids, completed)
                    continue

                new = 0
                for p in papers:
                    pmid = p.get("pmid", "")
                    if pmid and pmid not in seen_pmids:
                        seen_pmids.add(pmid)
                        out_fh.write(json.dumps(p, ensure_ascii=False) + "\n")
                        new += 1

                docs_written += new
                logger.info(
                    "  %r → fetched=%d new=%d total_written=%d",
                    descriptor,
                    len(papers),
                    new,
                    docs_written,
                )
                completed.add(cat_key)
                _save_checkpoint(checkpoint_path, seen_pmids, completed)
                time.sleep(0.5)
    finally:
        out_fh.close()

    logger.info("Ingestion complete — %d unique documents written", docs_written)

    # Read back the NDJSON and return as a list for the caller to re-serialise
    # as a standard JSON array (the format expected by downstream tools).
    corpus: list[dict] = []
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(json.loads(line))
    return corpus


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build the PubMed bulk corpus for clinical RAG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output", type=Path, default=Path("data/raw/pubmed_bulk_corpus.json"))
    p.add_argument("--target", type=int, default=200_000, help="Target total corpus size")
    p.add_argument("--min-year", type=int, default=2015, help="Recency lower bound (inclusive)")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint file path (default: output + .ckpt.json)",
    )
    p.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    return p


def main() -> None:
    load_dotenv(override=True)

    Entrez.email = os.getenv("NCBI_EMAIL", "")
    Entrez.api_key = os.getenv("NCBI_API_KEY") or None

    if not Entrez.email:
        raise EnvironmentError("NCBI_EMAIL is not set. Add it to your .env file.")

    args = _build_parser().parse_args()
    output_path: Path = args.output
    checkpoint_path: Path = args.checkpoint or output_path.with_suffix(".ckpt.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting PubMed ingestion — target=%d, min_year=%d, output=%s",
        args.target,
        args.min_year,
        output_path,
    )

    # Use a staging path for the NDJSON stream; replace with a JSON array at the end.
    ndjson_path = output_path.with_suffix(".ndjson")

    corpus = ingest_corpus(
        output_path=ndjson_path,
        target=args.target,
        min_year=args.min_year,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
    )

    logger.info("Writing final JSON array to %s …", output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    ndjson_path.unlink(missing_ok=True)
    logger.info("Saved %d documents to %s", len(corpus), output_path)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint removed.")


if __name__ == "__main__":
    main()
