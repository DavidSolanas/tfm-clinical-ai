# Bibliography Reference Guide — TFM: Hybrid Clinical Assistance System (LLM + RAG)

> **Purpose:** Operational reference for navigating the thesis bibliography. Each entry includes a summary, the thesis section where it applies, and the specific argument it supports. Organized by thematic block aligned with the thesis structure.
>
> **Last updated:** 2026-04-14 (rev. 3 — added Pingua et al. 2025 and Collaco et al. 2026)  
> **Total references:** 30

---

## Table of Contents

1. [Foundational Architecture (Transformers & BERT)](#1-foundational-architecture)
2. [Base Large Language Models](#2-base-large-language-models)
3. [Efficient Fine-Tuning (LoRA / QLoRA)](#3-efficient-fine-tuning)
4. [Retrieval-Augmented Generation — Foundational](#4-rag-foundational)
5. [RAG vs. Fine-Tuning — Ablation Study Justification](#5-rag-vs-fine-tuning--ablation-study-justification)
6. [Biomedical Embeddings](#6-biomedical-embeddings)
7. [Vector Databases & Similarity Search](#7-vector-databases--similarity-search)
8. [Hallucinations in LLMs](#8-hallucinations-in-llms)
9. [RAG & LLMs in Clinical/Healthcare Settings](#9-rag--llms-in-clinicalhealthcare-settings)
10. [Evaluation Frameworks](#10-evaluation-frameworks)
11. [Clinical Datasets & De-identification](#11-clinical-datasets--de-identification)
12. [Data Scarcity & Synthetic Data Augmentation](#12-data-scarcity--synthetic-data-augmentation)
13. [Clinical Decision-Making & Information Overload](#13-clinical-decision-making--information-overload)

---

## 1. Foundational Architecture

| BibTeX Key | Authors & Year | Title | Summary | When / Where to Use | Why It Matters |
|---|---|---|---|---|---|
| `vaswani_attention_2023` | Vaswani et al. (2017/2023) | *Attention Is All You Need* | Introduces the Transformer architecture — the foundation of all modern LLMs. Proposes the self-attention mechanism as a replacement for recurrence. | **Chapter 2 (State of the Art):** When introducing the architecture underlying LLaMA-3 and BERT-family encoders. | Every model in the thesis (LLaMA-3, BioBERT, PubMedBERT) is built on this architecture. Without this citation, the theoretical foundation is undocumented. |
| `devlin_bert_2019` | Devlin et al. (2019) | *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* | Introduces BERT, the encoder-only Transformer that established the pretraining + fine-tuning paradigm. Sets the baseline that BioBERT and ClinicalBERT extend. | **Chapter 2 (State of the Art):** When contextualizing how domain-specific models (BioBERT) evolved from general-purpose pretrained encoders. | Necessary antecedent for all biomedical BERT variants used as embedding models in the RAG pipeline. |


---

## 2. Base Large Language Models

| BibTeX Key | Authors & Year | Title | Summary | When / Where to Use | Why It Matters |
|---|---|---|---|---|---|
| `grattafiori_llama_2024` | Grattafiori et al. / Meta AI (2024) | *The Llama 3 Herd of Models* | Official technical report for the LLaMA 3 model family. Details architecture, training data, and benchmarks for models ranging from 8B to 405B parameters. | **Chapter 2 (State of the Art):** When introducing LLaMA-3-8B as the backbone model. **Chapter 3 (Methodology):** When justifying the choice of the 8B variant. | This is the primary citation for the model being fine-tuned. Must be cited whenever LLaMA-3 is mentioned by name in the thesis. |
| `openai_gpt-4_2024` | OpenAI (2024) | *GPT-4 Technical Report* | Technical description of GPT-4's capabilities and training process. Represents the proprietary model family against which open-source models are compared. | **Chapter 2 (State of the Art):** As a reference point for proprietary LLM capabilities. **Chapter 3 (Methodology):** To support the DECISIONS.md argument for choosing open-source over closed APIs. | Supports the contrast between proprietary and open-source approaches. Indirectly justifies LLaMA-3 selection on privacy/reproducibility grounds. |

---

## 3. Efficient Fine-Tuning

| BibTeX Key | Authors & Year | Title | Summary | When / Where to Use | Why It Matters |
|---|---|---|---|---|---|
| `hu_lora_2021` | Hu et al. (2021) | *LoRA: Low-Rank Adaptation of Large Language Models* | Introduces LoRA, which freezes pretrained weights and injects trainable low-rank matrices into attention layers. Reduces trainable parameters dramatically while preserving performance. | **Chapter 2 (State of the Art):** Efficient fine-tuning block. **Chapter 3 (Methodology):** When describing the mathematical basis of the fine-tuning approach. | LoRA is the algorithmic foundation of QLoRA. This citation is mandatory any time you explain *how* fine-tuning is implemented in your system. |
| `dettmers_qlora_2023` | Dettmers et al. (2023) | *QLoRA: Efficient Finetuning of Quantized LLMs* | Extends LoRA by adding 4-bit NormalFloat quantization (NF4) of the base model, enabling fine-tuning of 65B models on a single GPU. The direct method used in the thesis. | **Chapter 3 (Methodology):** As the primary citation for the fine-tuning technique. **Chapter 2 (State of the Art):** In the efficient fine-tuning subsection. | This is the specific method implemented via Unsloth. Must appear in the methodology chapter whenever the fine-tuning pipeline is described. |
| `parthasarathy_ultimate_2024` | Parthasarathy et al. (2024) | *The Ultimate Guide to Fine-Tuning LLMs from Basics to Breakthroughs* | Comprehensive survey covering fine-tuning methodologies (supervised, unsupervised, instruction-based), hyperparameter tuning, LoRA, MoE, RLHF, and deployment pipelines. | **Chapter 2 (State of the Art):** As a broad reference for fine-tuning landscape and best practices. **Chapter 3 (Methodology):** For citing hyperparameter decisions or the seven-stage pipeline framework. | Useful to provide the broader context of fine-tuning methodology without requiring many individual citations for standard practices. |

---

## 4. RAG — Foundational

| BibTeX Key | Authors & Year | Title | Summary | When / Where to Use | Why It Matters |
|---|---|---|---|---|---|
| `lewis_retrieval-augmented_2021` | Lewis et al. (2021) | *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | Seminal paper defining RAG: combining a parametric LLM with a non-parametric dense retrieval index. Introduces the RAG-Token and RAG-Sequence formulations. Sets the theoretical standard for all subsequent RAG work. | **Chapter 2 (State of the Art):** As the foundational citation for RAG. **Chapter 3 (Methodology):** Whenever the RAG pipeline is introduced conceptually. | This is the paper that defines RAG. It is non-negotiable — cite it every time RAG is introduced as a technique. |

---

## 5. RAG vs. Fine-Tuning — Ablation Study Justification

| BibTeX Key | Authors & Year | Title | Summary | When / Where to Use | Why It Matters |
|---|---|---|---|---|---|
| `balaguer_rag_2024` | Balaguer et al. (2024) | *RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture* | Direct empirical comparison of four configurations: base model, fine-tuned model, RAG-augmented model, and combined FT+RAG. Shows accuracy gains of ~6 p.p. from fine-tuning and ~5 p.p. additional from RAG — demonstrating cumulative, complementary benefits. | **Chapter 3 (Methodology):** To justify the four-configuration experimental design. **Chapter 4 (Results):** To frame the interpretation of ablation results. **Chapter 2 (State of the Art):** In the subsection comparing RAG and fine-tuning. | This is the strongest precedent for your exact experimental design. Cite it to argue that comparing Base / FT / RAG / FT+RAG is a validated research methodology, not an arbitrary choice. |
| `ovadia_fine-tuning_2024` | Ovadia et al. (2024) | *Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs* | Compares unsupervised fine-tuning vs. RAG across knowledge-intensive tasks. Finds RAG consistently outperforms unsupervised fine-tuning for injecting new factual knowledge, but fine-tuning improves stylistic/domain adaptation. | **Chapter 2 (State of the Art):** To frame the complementarity debate between the two strategies. **Chapter 5 (Discussion):** When explaining why each configuration addresses a different aspect of the problem. | Supports the thesis narrative that fine-tuning and RAG solve *different* problems: fine-tuning shapes how the model speaks (style, format), RAG shapes what it knows (current evidence). Essential for the "Distinct Contributions" section. |
| `pingua_medical_FT_vs_RAG_2025` | Pingua et al. (2025) | *Medical LLMs: Fine-Tuning vs. Retrieval-Augmented Generation* | Empirical comparison of RAG alone, FT alone, and FT+RAG on healthcare data across multiple models. Finds RAG and FT+RAG outperforming FT alone, with LLaMA showing superior overall performance. | **Chapter 3 (Methodology):** To justify the 4-configuration ablation study in a medical context. **Chapter 4 (Results):** To contextualize our LLaMA-based FT+RAG results against similar medical benchmarks. | Directly validates the thesis experimental design specifically for medical domains and strongly supports the choice of LLaMA as the base model. |
| `collaco_integrating_2026` | Collaco et al. (2026) | *Integrating Fine-Tuning and Retrieval-Augmented Generation for Healthcare AI Systems: A Scoping Review* | Scoping review of FT + RAG hybrid systems in healthcare. Concludes that FT+RAG systems consistently outperform FT-only or RAG-only approaches, improving accuracy and reducing hallucinations. | **Chapter 2 (State of the Art):** When reviewing the current evidence for hybrid architectures in clinical AI. **Chapter 5 (Discussion):** To support the conclusion that the hybrid approach is superior to either method alone. | Provides macro-level systemic evidence that the thesis's core hypothesis (FT+RAG hybrid superiority) is aligned with the emerging consensus in healthcare AI. |

---

## 6. Biomedical Embeddings

| BibTeX Key | Authors & Year | Title | Summary | When / Where to Use | Why It Matters |
|---|---|---|---|---|---|
| `lee_biobert_2020` | Lee et al. (2020) | *BioBERT: a pre-trained biomedical language representation model for biomedical text mining* | Introduces BioBERT, a BERT model pretrained on PubMed abstracts and PMC full-text articles. Outperforms general-domain BERT on NER, relation extraction, and QA tasks in biomedical text. | **Chapter 3 (Methodology):** When introducing the embedding model for the RAG pipeline. **Chapter 2 (State of the Art):** In the biomedical NLP embeddings subsection. | Primary citation for BioBERT. If this is your chosen embedding model, this citation is mandatory in the methodology. |
| `alsentzer_publicly_2019` | Alsentzer et al. (2019) | *Publicly Available Clinical BERT Embeddings* | Introduces ClinicalBERT, pretrained on MIMIC-III clinical notes. Focuses on discharge summaries and general clinical text. Complements BioBERT for intra-hospital documentation rather than research literature. | **Chapter 3 (Methodology):** As an alternative embedding model considered during benchmarking. **Chapter 2 (State of the Art):** When surveying domain-specific BERT variants. | Supports the embedding model benchmarking rationale. Cite alongside BioBERT to show awareness of alternatives. Use in the "why BioBERT over ClinicalBERT" argument (PubMed corpus alignment). |
| `fang_bioformer_2023` | Fang et al. (2023) | *Bioformer: an efficient transformer language model for biomedical text mining* | Presents Bioformer, a 60%-smaller BERT-family model pretrained on PubMed and PMC. Achieves performance within 0.1–0.9% of PubMedBERT while being 2–3× faster. Benchmarks across 15 NLP tasks including NER, RE, and QA. | **Chapter 3 (Methodology):** In the embedding model selection and benchmarking subsection, specifically to justify trade-offs between accuracy and computational efficiency. **Chapter 5 (Discussion) / Limitations:** To contextualize embedding model choice. | Provides the benchmark tables needed to justify your embedding model selection empirically. Use to show you evaluated the accuracy/efficiency trade-off, not just picked BioBERT by default. |
| `gu_pubmedbert_2021` | Gu et al. (2021) | *Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing* | Introduces PubMedBERT, pretrained from scratch exclusively on PubMed text (as opposed to fine-tuned from general BERT). Achieves state-of-the-art on the BLURB benchmark across 8 biomedical NLP tasks. Demonstrates that from-scratch domain pretraining consistently outperforms domain adaptation of general-purpose models. | **Chapter 2 (State of the Art):** As the primary citation for PubMedBERT in the biomedical embeddings subsection. **Chapter 3 (Methodology):** When justifying PubMedBERT as a candidate embedding model alongside BioBERT. | PubMedBERT is the other main candidate embedding model in the thesis (BioBERT/PubMedBERT are listed as the RAG embedding pair). This citation is mandatory any time PubMedBERT is mentioned. Also directly relevant to the empirical benchmarking argument: the BLURB results provide the quantitative basis for choosing between BioBERT and PubMedBERT. |

---

## 7. Vector Databases & Similarity Search

| BibTeX Key | Authors & Year | Title | Summary | When / Where to Use | Why It Matters |
|---|---|---|---|---|---|
| `malkov_efficient_2020` | Malkov & Yashunin (2020) | *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs* | Introduces HNSW (Hierarchical Navigable Small World), the graph-based ANN algorithm that is the backbone of Qdrant's indexing strategy. Achieves logarithmic search complexity with high recall. | **Chapter 3 (Methodology):** When describing how Qdrant performs semantic retrieval. Specifically when explaining ANN-based cosine similarity search. | Qdrant does not need its own academic citation — what needs citing is the algorithm it implements. This paper provides the mathematical foundation for the retrieval speed claims in your system. |
| `johnson_billion-scale_2021` | Johnson, Douze & Jégou (2021) | *Billion-Scale Similarity Search with GPUs (FAISS)* | Presents FAISS, Facebook AI's GPU-accelerated library for dense vector similarity search at massive scale. Introduces IVF-based approximate search and GPU indexing optimizations. | **Chapter 2 (State of the Art):** When surveying vector database and similarity search technologies as context for choosing Qdrant. **Chapter 3 (Methodology):** As a reference point for the broader ecosystem of ANN tools. | Positions Qdrant within the landscape of similarity search solutions. Cite alongside Malkov to give a complete picture of the ANN field, then explain why Qdrant's HNSW + filtering capabilities were preferred for this use case. |

---

## 8. Hallucinations in LLMs

| BibTeX Key | Authors & Year | Title | Summary | When / Where to Use | Why It Matters |
|---|---|---|---|---|---|
| `ji_survey_2023` | Ji et al. (2023) | *Survey of Hallucination in Natural Language Generation* | Comprehensive survey of hallucination types, metrics, and mitigation strategies in NLG. Covers summarization, dialogue, QA, and machine translation. The canonical reference for defining hallucination as a problem. | **Chapter 1 (Introduction):** When motivating the thesis problem. **Chapter 2 (State of the Art):** Hallucination subsection. **Chapter 4/5 (Results/Discussion):** When interpreting hallucination reduction metrics. | Essential to define and frame the hallucination problem formally. Required to justify the 35% hallucination reduction hypothesis and to discuss RAGAS Faithfulness metric. |
| `pal_med-halt_2023` | Pal, Umapathi & Sankarasubbu (2023) | *Med-HALT: Medical Domain Hallucination Test for Large Language Models* | Introduces Med-HALT, a benchmark specifically designed to evaluate hallucinations in medical LLMs. Tests reasoning-based and memory-based hallucinations across multiple LLMs including LLaMA-2 and GPT-3.5. | **Chapter 2 (State of the Art):** In the hallucination and medical AI safety subsection. **Chapter 3 (Methodology):** If citing domain-specific hallucination risks as motivation for RAG grounding. | Strengthens the domain-specific argument: general hallucination metrics are insufficient for medicine. Provides precedent that medical hallucination benchmarking is an active research area. |

---

## 9. RAG & LLMs in Clinical / Healthcare Settings

| BibTeX Key | Authors & Year | Title | Summary | When / Where to Use | Why It Matters |
|---|---|---|---|---|---|
| `amugongo_retrieval_2025` | Amugongo et al. (2025) | *Retrieval augmented generation for large language models in healthcare: A systematic review* | Systematic review of RAG methodologies applied in healthcare. Identifies Naive RAG, Advanced RAG, and Modular RAG variants in clinical use. Notes dominance of proprietary models (GPT-3.5/4) and absence of standardized evaluation frameworks. | **Chapter 2 (State of the Art):** As the primary systematic review framing the RAG-in-healthcare landscape. **Chapter 1 (Introduction):** When contextualizing the research gap (open-source + standardized evaluation). | Key paper for positioning the thesis contribution. The gap identified (proprietary model dominance, no standardized evaluation) is exactly what the thesis addresses with open-source LLaMA-3 and RAGAS. |
| `rau_context-based_2023` | Rau et al. (2023) | *A Context-based Chatbot Surpasses Radiologists and Generic ChatGPT in Following the ACR Appropriateness Guidelines* | Demonstrates a RAG-based chatbot (built with LlamaIndex + GPT-3.5-turbo) that outperforms radiologists on imaging recommendations. Reports 5-minute decision time vs. 50 minutes for human radiologists. | **Chapter 2 (State of the Art):** As a concrete example of RAG applied to clinical decision support, and notably using LlamaIndex — the same orchestration framework as the thesis. | Provides clinical validation for the RAG approach and directly demonstrates LlamaIndex in a medical setting. Cite when positioning the thesis within existing clinical RAG applications. |
| `yu_zero-shot_2023` | Yu, Guo & Sano (2023) | *Zero-Shot ECG Diagnosis with Large Language Models and Retrieval-Augmented Generation* | Applies zero-shot RAG to cardiac arrhythmia and sleep apnea diagnosis from ECG signals. Converts domain knowledge into retrieval databases; outperforms few-shot LLM methods and is competitive with supervised approaches. | **Chapter 2 (State of the Art):** When surveying RAG applications in clinical diagnosis beyond text. | Demonstrates RAG can substitute for fine-tuning in zero-shot clinical scenarios. Useful as a contrast case: shows when RAG alone may suffice vs. when fine-tuning adds value. |
| `thompson_large_2023` | Thompson et al. (2023) | *Large Language Models with Retrieval-Augmented Generation for Zero-Shot Disease Phenotyping* | Uses RAG + MapReduce for phenotyping pulmonary hypertension (a rare disease) from EHRs. Outperforms rule-based physician logic (F1: 0.75 vs. 0.62). | **Chapter 2 (State of the Art):** When demonstrating RAG's value for rare disease scenarios with limited labeled data. **Chapter 5 (Discussion):** To contextualize results on MTSamples (a similarly constrained dataset). | Provides precedent for RAG compensating for data scarcity, directly relevant to the MTSamples limitation argument. |
| `pressman_clinical_2024` | Pressman et al. (2024) | *Clinical and Surgical Applications of Large Language Models: A Systematic Review* | Systematic review of 34 studies on LLM applications in clinical and surgical settings. Identifies uses in diagnosis, triage, surgical planning, and administrative tasks. Covers concerns about accuracy and bias. | **Chapter 2 (State of the Art):** Broad review of LLM use in clinical contexts; supports motivation for the thesis. **Chapter 1 (Introduction):** When stating the potential and risks of LLMs in healthcare. | Provides the macro-level evidence that LLMs in clinical settings are an established research area. Supports the thesis relevance argument without requiring narrow domain citations. |

---

## 10. Evaluation Frameworks

| BibTeX Key | Authors & Year | Title | Summary | When / Where to Use | Why It Matters |
|---|---|---|---|---|---|
| `es_ragas_2025` | Es et al. (2025) | *Ragas: Automated Evaluation of Retrieval Augmented Generation* | Introduces RAGAS, a framework for automated evaluation of RAG pipelines. Key metrics: Context Precision, Context Recall, Faithfulness (>0.90 threshold), and Answer Relevancy. Enables reference-free evaluation. | **Chapter 3 (Methodology):** As the primary evaluation framework citation. **Chapter 4 (Results):** When reporting Context Precision, Faithfulness, and other RAGAS metrics. | RAGAS is the evaluation framework for the entire quantitative evaluation. This citation must appear every time RAGAS metrics are mentioned. Directly tied to the thesis's quantitative thresholds (Context Precision >80%, Faithfulness >0.90). |

---

## 11. Clinical Datasets & De-identification

| BibTeX Key | Authors & Year | Title | Summary | When / Where to Use | Why It Matters |
|---|---|---|---|---|---|
| `boyle_medical_2017` | Boyle, Tara (2017) | *Medical Transcriptions* [Kaggle dataset] | The canonical dataset used for fine-tuning. Contains ~4,999 de-identified medical transcription reports across 40 specialties, scraped from mtsamples.com and published on Kaggle under CC0 (Public Domain) license. There is no peer-reviewed paper for this dataset — it is a community-curated resource with a named creator, a stable URL, and an explicit open license. | **Chapter 3 (Methodology):** As the primary citation for the fine-tuning training corpus. Every mention of "MTSamples" in the thesis must trace back to this reference. | This is the only citable source for the dataset. The CC0 license and public availability are also part of the privacy/reproducibility argument for using it over real patient data. Note: when citing a Kaggle dataset in a thesis, use the format: Author, Title, Platform, Year, URL. |

---

## 12. Data Scarcity & Synthetic Data Augmentation

| BibTeX Key | Authors & Year | Title | Summary | When / Where to Use | Why It Matters |
|---|---|---|---|---|---|
| `montenegro_what_2025` | Montenegro, Gomes & Machado (2025) | *What We Know About the Role of Large Language Models for Medical Synthetic Dataset Generation* | PRISMA systematic review of 153 studies on LLM-generated synthetic medical text. Covers RAG-enhanced generation, structured fine-tuning, and domain adaptation. Identifies hallucinations as a persistent problem even in generated text. Discusses SOAP/Calgary-Cambridge consultation models. | **Chapter 5 (Discussion) / Limitations:** When acknowledging MTSamples' small size as a limitation and proposing synthetic data augmentation as future work. **Chapter 6 (Future Work):** As the primary citation for the data augmentation direction. | Validates the future work direction. Cite when stating: "A limitation of this work is the size of MTSamples; future iterations could apply LLM-based synthetic data augmentation to expand the training corpus, as reviewed in [Montenegro et al., 2025]." |

---

## 13. Clinical Decision-Making & Information Overload

| BibTeX Key | Authors & Year | Title | Summary | When / Where to Use | Why It Matters |
|---|---|---|---|---|---|
| `laker_quality_2018` | Laker et al. (2018) | *Quality and Efficiency of the Clinical Decision-Making Process: Information Overload and Emphasis Framing* | Controlled experiment with ED physicians showing that information overload degrades decision quality and speed. Introduces "emphasis framing" as a mitigation technique. Measures quality and timeliness of clinical decisions. | **Chapter 1 (Introduction):** When motivating the problem of information overload in clinical settings as the root cause the thesis addresses. | Provides empirical evidence for the problem statement. Quantifies the cost of information overload (decision quality, speed) — directly supports the 65% time reduction hypothesis framing. |
| `rebitzer_influence_2008` | Rebitzer, Rege & Shepard (2008) | *Influence, information overload, and information technology in health care* | Analyzes how IT systems in healthcare can induce information overload rather than mitigate it. Examines the role of peer influence and technology in clinical decision-making. | **Chapter 1 (Introduction):** As secondary support for the information overload problem in healthcare IT. | Provides broader economic/organizational context for why clinical AI assistance systems are needed. Supports the motivation chapter alongside Laker et al. |

---

## Quick Reference: Citation by Thesis Section

| Thesis Section | Priority References |
|---|---|
| Chapter 1 — Introduction / Problem Statement | `ji_survey_2023`, `laker_quality_2018`, `rebitzer_influence_2008`, `amugongo_retrieval_2025`, `pressman_clinical_2024` |
| Chapter 2 — State of the Art: Transformers & LLMs | `vaswani_attention_2023`, `devlin_bert_2019`, `grattafiori_llama_2024`, `openai_gpt-4_2024` |
| Chapter 2 — State of the Art: Efficient Fine-Tuning | `hu_lora_2021`, `dettmers_qlora_2023`, `parthasarathy_ultimate_2024` |
| Chapter 2 — State of the Art: RAG | `lewis_retrieval-augmented_2021`, `balaguer_rag_2024`, `ovadia_fine-tuning_2024`, `collaco_integrating_2026` |
| Chapter 2 — State of the Art: Biomedical NLP | `lee_biobert_2020`, `alsentzer_publicly_2019`, `gu_pubmedbert_2021`, `fang_bioformer_2023` |
| Chapter 2 — State of the Art: RAG in Healthcare | `amugongo_retrieval_2025`, `rau_context-based_2023`, `yu_zero-shot_2023`, `thompson_large_2023`, `collaco_integrating_2026` |
| Chapter 2 — State of the Art: Hallucinations | `ji_survey_2023`, `pal_med-halt_2023` |
| Chapter 3 — Methodology: Base Model | `grattafiori_llama_2024` |
| Chapter 3 — Methodology: Fine-Tuning Pipeline | `hu_lora_2021`, `dettmers_qlora_2023` |
| Chapter 3 — Methodology: RAG Pipeline | `lewis_retrieval-augmented_2021`, `lee_biobert_2020` |
| Chapter 3 — Methodology: Vector Store | `malkov_efficient_2020`, `johnson_billion-scale_2021` |
| Chapter 3 — Methodology: Embedding Selection | `lee_biobert_2020`, `gu_pubmedbert_2021`, `alsentzer_publicly_2019`, `fang_bioformer_2023` |
| Chapter 3 — Methodology: Dataset | `boyle_medical_2017` |
| Chapter 3 — Methodology: Evaluation | `es_ragas_2025` |
| Chapter 3 — Methodology: 4-Config Experimental Design | `balaguer_rag_2024`, `ovadia_fine-tuning_2024`, `pingua_medical_FT_vs_RAG_2025` |
| Chapter 4 — Results | `es_ragas_2025`, `ji_survey_2023`, `balaguer_rag_2024`, `pingua_medical_FT_vs_RAG_2025` |
| Chapter 5 — Discussion | `ovadia_fine-tuning_2024`, `thompson_large_2023`, `amugongo_retrieval_2025`, `collaco_integrating_2026` |
| Chapter 5 — Limitations | `montenegro_what_2025`, `fang_bioformer_2023` |
| Chapter 6 — Future Work | `montenegro_what_2025` |
