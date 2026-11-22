# RAG Pipeline for sample Legal Data

> A hands-on project to understand how a Retrieval-Augmented Generation (RAG) application is translated from a product problem into a working technical solution.

## Why I Built This

As a Product Manager working with data-driven platforms and exploring emerging AI capabilities, I wanted to go beyond understanding Generative AI at a conceptual, product-strategy or architecture level. I wanted to get hands-on with how a GenAI application is put together: how its components work, how they interact, and how technical choices influence the experience and quality delivered to users.

I chose **Legal Document Intelligence** as a sample use case and used it to explore the end-to-end building blocks of a RAG application — from document ingestion and chunking to embeddings, hybrid retrieval, reranking, LLM-based answer generation, guardrails and evaluation.

I used coding LLMs as development assistants during implementation. My objective was not simply to generate code, but to understand the underlying concepts, validate the approach, investigate implementation issues and learn the trade-offs involved in assembling a working solution.

Through this exercise, I wanted to:

- **Connect product intent to technology:** understand how a user problem and product requirements translate into system capabilities and technical components.
- **Understand the RAG lifecycle:** learn how ingestion, chunking, embedding, retrieval, reranking and generation work together.
- **Explore technical trade-offs:** understand how choices such as chunk size, embedding models, retrieval strategies and reranking affect relevance and answer quality.
- **Evaluate the solution:** go beyond whether an LLM returns an answer and examine retrieval quality, groundedness and response relevance.
- **Build production awareness:** identify the additional considerations needed to evolve a working prototype into a reliable, secure, scalable and cost-conscious product.

This repository captures the working implementation and the learning journey. It is a sample learning and demonstration project, **not a production-ready legal advice system**.

## Generic RAG System Design (HLD Production Grade):

- [System Design](RAG_SYSTEMDESIGN_README.md) — detailed implemented architecture, component interactions, data flows, technical decisions and separately identified enterprise-level recommendations.

---

## 1. Product Overview

Legal Document Intelligence is a sample application that allows a user to ask natural-language questions about a collection of legal documents and receive responses based on relevant retrieved passages.

The application follows a Retrieval-Augmented Generation approach:

1. Legal documents are loaded and prepared for retrieval.
2. Documents are divided into chunks and represented as embeddings.
3. A user query is processed through retrieval components.
4. Relevant passages are retrieved using dense and sparse retrieval.
5. Retrieval results are combined and reranked.
6. The selected context is passed to a language model to generate a response.
7. Input and output guardrails validate the interaction.

The intention is to make responses more closely grounded in the supplied document collection than a response generated from the language model's general knowledge alone.

## 2. Product Problem and Intended Capabilities

**Problem statement**

Users need a practical way to find relevant information across a collection of lengthy legal documents without manually searching through each document. The application explores how users can ask questions in natural language and receive answers grounded in relevant document passages.

The intended capabilities are:

| Product need | Capability explored |
|---|---|
| Prepare a document collection for search | Document loading, preprocessing and chunking |
| Find conceptually relevant passages | Dense / semantic retrieval |
| Find passages containing important legal terminology | Sparse / keyword retrieval |
| Combine complementary retrieval results | Hybrid retrieval using Reciprocal Rank Fusion (RRF) |
| Improve the ordering of candidate passages | Cross-encoder reranking |
| Generate an answer using retrieved evidence | Context-grounded LLM generation |
| Validate user queries and generated responses | Input and output guardrails |
| Measure the quality of the solution | Retrieval evaluation and RAGAS evaluation |


## 3. Solution Overview

The implementation has two main workflows.

### 3.1 Offline document preparation and indexing

The offline workflow prepares the source documents and builds the data structures used by retrieval.

```text
Source legal documents
        |
        v
Document loading and preprocessing
        |
        v
Chunking and chunk persistence
        |
        v
Embedding generation
        |
        +------------------------+
        |                        |
        v                        v
   ChromaDB                 BM25 index
   (dense search)           (sparse search)
```

The project keeps document-processing and index-building artifacts locally. They are generated from the source corpus and are not intended to be committed to this repository.

### 3.2 Online question-answering workflow

```text
User question
      |
      v
Input guardrail
      |
      v
Dense retrieval + BM25 retrieval
      |
      v
Reciprocal Rank Fusion
      |
      v
Reranking and context selection
      |
      v
LLM answer generation
      |
      v
Output guardrail
      |
      v
Response
```

The online workflow is coordinated by the RAG orchestrator. The implementation also returns retrieved context and source metadata for inspection.

> These diagrams are a high-level view of the current project. The detailed component design, interfaces, data flow, failure handling and proposed enterprise-level architecture will be documented separately in [`docs/system-design.md`](docs/system-design.md). Implemented capabilities will be distinguished from future architectural recommendations.

## 4. RAG Components Explored

| Component | Role in the solution |
|---|---|
| Document loader and preprocessing | Load and prepare source documents |
| Chunking | Split documents into retrievable passages while retaining source and location metadata |
| Embedding model | Convert text into vector representations for semantic retrieval |
| ChromaDB | Persist and search document vectors |
| BM25 | Retrieve passages using lexical / keyword relevance |
| Reciprocal Rank Fusion | Combine ranked results from dense and sparse retrieval |
| Reranker | Re-score candidate passages for query-specific relevance |
| Gemini | Generate an answer using the retrieved context |
| Input and output guardrails | Apply validation before and after generation |
| Evaluation modules | Assess retrieval, generation and RAG quality |

## 5. Technology Stack

The following reflects the configuration currently represented in the project. Exact dependency versions and runtime prerequisites will be maintained in `requirements.txt`.

| Area | Technology / configuration |
|---|---|
| Language | Python |
| Embeddings | `BAAI/bge-m3` |
| Vector store | ChromaDB |
| Sparse retrieval | `bm25s` |
| Retrieval fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | `BAAI/bge-reranker-v2-m3` |
| Answer generation | Google Gemini API — `gemini-3.5-flash-lite` (configured model identifier) |
| Guardrail configuration | YAML |
| RAG evaluation | RAGAS |
| Configuration | Python settings and YAML files |


### 5.1 Component choices and practical considerations

The stack is a set of replaceable components rather than a claim that one tool is universally best. The choices below reflect the current implementation and the concepts this project is intended to demonstrate.

| Component | Why it is used here | Practical considerations |
|---|---|---|
| **BGE-M3 embeddings** | Creates vector representations used by dense retrieval. | Embedding quality depends on the domain, language, query style and document content. Evaluate candidate models on representative queries rather than choosing solely from benchmark reputation. The query instruction and document/query encoding approach should be consistent with the model's intended usage. |
| **ChromaDB** | Persists embeddings and associated chunk metadata and supports similarity search. | The vector store is responsible for storage and retrieval, not for deciding whether a passage is legally authoritative. Preserve source identifiers and location metadata so retrieved passages can be traced back to their documents. |
| **BM25 (`bm25s`)** | Provides lexical retrieval alongside semantic search. | Particularly useful for exact legal terminology, section numbers, party names and phrases that may be diluted in semantic similarity. Tokenization and corpus preparation influence its results. |
| **Reciprocal Rank Fusion** | Combines ranked dense and sparse results without requiring their raw scores to be directly comparable. | The fusion constant and candidate depths affect the combined ranking. Tune using a fixed benchmark and compare the fused output with each individual retriever. |
| **BGE reranker** | Re-scores query–passage candidates to improve query-specific ordering. | A cross-encoder is generally more computationally expensive than retrieving vectors, so it is commonly applied to a bounded candidate set rather than the entire corpus. |
| **Gemini** | Generates natural-language answers using the selected retrieved context. | Model selection involves answer quality, context-window capacity, latency, cost, availability and governance requirements. Model identifiers and provider behavior can change; verify the configured model against the provider's current documentation. |
| **YAML guardrail configuration** | Keeps selected validation rules and thresholds outside the Python implementation. | Separating configuration from code makes rules easier to inspect and tune. Configuration-based checks are not a substitute for comprehensive security testing or authorization. |
| **RAGAS** | Provides evaluation metrics for aspects of retrieval context and generated answers. | Treat automated scores as evaluation signals. Validate benchmark quality and periodically review a sample of results manually. |

### 5.2 Chunking strategies for unstructured documents

Chunking is one of the most influential design choices in a RAG pipeline. The best strategy depends on document structure, the type of questions users ask, the need to preserve context and the retrieval/evaluation results. A single chunk size or strategy should not be assumed to work equally well for every corpus.

This project has explored multiple chunking configurations and chunk sizes during experimentation. The values currently retained in `settings.py` are the configuration for the present sample workflow; they are **not a universal recommendation or a claim that one size is optimal**.

Common strategies include:

| Strategy | Description | When it can help |
|---|---|---|
| **Fixed-size chunking** | Splits text into chunks using a target character or token length, often with overlap between adjacent chunks. | Simple baseline; provides predictable chunk sizes, but may split a paragraph, clause or logical unit. |
| **Paragraph-based chunking** | Uses paragraph boundaries as natural split points, optionally combining short paragraphs or splitting very long ones. | Preserves local meaning when paragraphs are well structured; paragraph lengths can vary considerably. |
| **Sentence-based chunking** | Splits on sentence boundaries and groups sentences to reach a target size. | Avoids cutting sentences in half, but a sentence alone may not contain enough context to answer a question. |
| **Recursive / structure-aware chunking** | Attempts larger structural boundaries first (for example, section, heading or paragraph) and falls back to smaller boundaries when a unit exceeds the target size. | Useful for documents with headings, numbered clauses and nested sections because it can preserve document hierarchy. |
| **Semantic chunking** | Uses semantic similarity or topic shifts between adjacent text units to identify likely boundaries. | Can keep conceptually related content together, but adds processing complexity and requires evaluation to justify it. |
| **Page- or section-based chunking** | Uses document layout, page boundaries, headings or legal section/clause identifiers as boundaries. | Helps preserve citations and navigation in structured legal material; page boundaries alone may split a logical provision. |
| **Overlap-based chunking** | Repeats a small amount of text between neighboring chunks so context near a boundary is not lost. | Can improve boundary continuity, but excessive overlap creates duplicate content, larger indexes and redundant retrieved context. |

#### Document cleanup before chunking

Chunking quality also depends on preprocessing. Typical cleanup can include removing repeated headers and footers, page numbers, broken line wraps, OCR artifacts, duplicate whitespace and irrelevant boilerplate while preserving meaningful numbering, headings, tables, citations and clause references. Over-aggressive cleanup can remove legal meaning, so transformations should be validated against the original document.

For legal documents, a useful approach to investigate is structure-aware chunking that preserves section and clause hierarchy, with controlled splitting for unusually long provisions. The right approach should be selected empirically using the same benchmark questions and retrieval metrics.

### 5.3 Prompt management

Prompts are part of the solution design and should be version-controlled and reviewed like other application artifacts. Rather than embedding long prompt text inside Python functions, keep reusable prompts in a dedicated Markdown file, for example:

```text
src/
  generation/
    generator.py
    prompts/
      legal_answer_prompt.md
```

The generation code can load the prompt template and supply the user question and retrieved context at runtime. This makes instructions easier to review, compare and version without changing generation logic.

A `SKILL.md` file is appropriate when defining a reusable agent skill or tool workflow with explicit instructions and capabilities. For this application’s ordinary RAG answer-generation prompt, a focused Markdown prompt file is simpler and clearer. The exact prompt location should follow the implementation once we inspect and refactor `generator.py`.

Prompt changes should be evaluated against a stable benchmark. Review whether the prompt asks the model to answer from supplied evidence, identify when evidence is insufficient, avoid inventing citations and distinguish source-supported facts from uncertainty.



### 5.4 Context-window and token-budget control

Token control is not the responsibility of one component. It is managed across the retrieval and generation path:

1. **Chunking** controls the approximate size of each retrievable unit. Token-aware chunking can make chunk lengths more predictable than character-only limits.
2. **Retrieval depth (`top_k`)** controls how many candidates are fetched from dense and sparse retrieval. Larger candidate pools can improve the chance of finding evidence, but increase reranking work.
3. **Reranking and final context selection (`top_n`)** help select a smaller, more relevant subset from the candidate pool before generation. Reranking does not itself enforce a token budget; it helps prioritize which passages should consume the available context.
4. **Context assembly** should enforce an explicit token budget for the combined system instructions, question, source passages and any conversation history. It should avoid duplicate or low-value passages and preserve source metadata.
5. **Generation settings** such as maximum output tokens limit the response length, not the size of the input context. The model's context-window limit must accommodate both input and output tokens.

For this project, retrieval candidate counts, the final context count and the configured maximum output tokens are available in settings. A production-oriented implementation should additionally measure the token count of the assembled prompt before sending it to the model, reserve capacity for the expected answer, and define behavior when the context exceeds the budget (for example, remove lower-ranked passages or reduce context size). This is important for predictable latency and cost as well as avoiding context-window errors.

## 6. Evaluation Approach — Measuring RAG Quality

A RAG system should not be evaluated only on whether the LLM returns a fluent answer. Retrieval and generation are separate stages, and a weak retrieval result can limit the quality of the final response even when the language model is capable.

This project therefore evaluates retrieval at multiple stages and evaluates generated answers separately. The purpose is to identify **where quality changes, which component contributes to that change, and what should be tuned next**.

### 6.1 Retrieval stages compared

The retrieval evaluator compares the following outputs:

| Stage | What it represents | Why compare it |
|---|---|---|
| **Dense retrieval** | Candidate passages retrieved using embedding similarity. | Establishes the semantic-search baseline and shows whether relevant passages can be found even when the query and document use different wording. |
| **BM25 retrieval** | Candidate passages retrieved using lexical term matching. | Shows how well exact terms, names, clause references and domain-specific vocabulary are matched. |
| **Fused retrieval** | Dense and BM25 result lists combined using Reciprocal Rank Fusion (RRF). | Shows whether combining semantic and lexical signals improves coverage and ranking over either retriever alone. |
| **Final reranked results** | Fused candidates reordered by the configured cross-encoder reranker, with the configured final context limit. | Shows whether query-aware relevance scoring improves the quality of the passages selected for generation. |

Comparing these stages helps diagnose the retrieval pipeline. For example, if BM25 finds a clause that dense retrieval misses, lexical retrieval may be contributing useful coverage. If fusion improves recall but reranking reduces it, the candidate pool, reranker or final selection limit may need investigation.

### 6.2 Retrieval metrics

The evaluator uses benchmark questions and their annotated gold evidence to assess whether the retrieved passages contain the expected evidence. In this project, evidence matching uses source-document identity and character-span overlap, so the results depend on the quality and completeness of the benchmark annotations.

| Metric | What it measures | Why it matters / how it informs decisions |
|---|---|---|
| **Hit Rate** | The proportion of evaluated questions for which at least one retrieved result matches the expected gold evidence. | Answers: “Did retrieval find something relevant at all?” A low score can indicate query, chunking, embedding or candidate-pool problems. |
| **Passage Recall** | How much of the expected relevant passage evidence is covered by the retrieved passages. | Indicates whether retrieval is finding enough of the supporting material, not just one matching fragment. Useful when answers require multiple clauses or details. |
| **Precision** | The proportion of retrieved passages that match the expected evidence. | Indicates how much of the retrieved set is relevant. Low precision can mean too much irrelevant context is being sent downstream, increasing noise and token use. |
| **Evidence Recall** | The proportion of the annotated gold evidence that is recovered by the retrieved results. | Helps identify missing supporting evidence. This is particularly important when a response needs all material parts of a legal provision, rather than one broadly related passage. |
| **Mean Reciprocal Rank (MRR)** | The average reciprocal rank of the first relevant result across queries. A relevant result at rank 1 contributes 1.0; at rank 2 contributes 0.5; at rank 3 contributes about 0.33. | Measures how early the first useful passage appears. Higher MRR means users and downstream generation are more likely to encounter useful evidence near the top of the ranked list. |

**How to use the metrics together:** Hit Rate and recall-oriented measures show whether relevant evidence is being found; Precision indicates how much irrelevant material is included; MRR shows how highly the first relevant passage is ranked. No single metric is sufficient. For example, increasing retrieval depth may improve recall while reducing precision and increasing the context size. The appropriate balance depends on the use case and the cost of missing evidence versus including extra passages.

The metric names above reflect the evaluator's implemented outputs. In particular, Passage Recall and Evidence Recall are based on this project's evidence-span matching logic; their exact calculation should be interpreted using the implementation and benchmark annotation format rather than assumed to be interchangeable with similarly named metrics in other evaluation libraries.

### 6.3 Generation evaluation

The generation evaluator runs benchmark questions through the RAG workflow and records generated answers, reference answers, retrieved contexts and source information for analysis.

This makes it possible to inspect answer quality alongside the evidence supplied to the model. A poor answer can then be investigated as a retrieval issue, a context-construction issue or a generation/prompt issue, rather than treating every failure as an LLM failure.

### 6.4 RAG quality evaluation with RAGAS

The RAGAS evaluator assesses selected generated responses using the following configured metrics:

| Metric | What it measures | Why it matters |
|---|---|---|
| **Faithfulness** | Whether claims in the generated answer are supported by the retrieved context. | Helps detect unsupported statements and hallucination risk. |
| **Answer Relevancy** | Whether the generated answer addresses the user's question. | Helps identify answers that are off-topic, incomplete or overly generic. |
| **Context Precision** | Whether the retrieved context contains relevant information, particularly near the top of the context ranking. | Helps assess retrieval noise and whether useful passages are prioritized. |
| **Context Recall** | Whether the retrieved context contains the information needed to answer the reference question. | Helps identify missing evidence that may prevent a complete answer. |

RAGAS scores are model- and configuration-dependent evaluation signals, not absolute guarantees of correctness. They should be interpreted alongside benchmark quality, human review and retrieval metrics.

### 6.5 Using evaluation to make decisions

The metrics are intended to support an iterative tuning process:

- **Low Hit Rate / Recall:** inspect document parsing, chunk boundaries, query formulation, embedding suitability and retrieval depth.
- **Low Precision:** inspect noisy chunks, overly broad retrieval, duplicate passages and the candidate ranking.
- **Low MRR:** inspect ranking quality, fusion settings and reranker performance.
- **Low Faithfulness:** inspect whether the required evidence was retrieved, whether context was constructed correctly, and whether the prompt sufficiently constrains the answer to the evidence.
- **Low Answer Relevancy:** inspect query interpretation, prompt instructions and whether the context actually addresses the question.
- **Low Context Precision or Recall:** revisit retrieval configuration, chunking strategy and the number of passages selected.

Evaluation should be run against a stable benchmark set when comparing configurations. Record the model, chunking configuration, retrieval parameters, reranker, context limit and evaluation sample size with each run so that changes can be attributed meaningfully.

**Evaluation results will be reported here after the final evaluation run.** No performance figures are stated in this draft; results should be added from the actual generated reports, together with sample size, configuration and relevant limitations.

## 7. Repository Structure

The repository is organized by responsibility. The structure below describes the main source packages; individual files and test locations should be kept aligned with the actual repository.

```text
legal-document-intelligence-rag/
├── README.md
├── .env.example
├── .gitignore
├── requirements.txt
├── docs/
│   ├── product-requirements.md
│   └── system-design.md
├── evaluation/
│   └── reports/                 # Local generated reports; not committed
└── src/
    ├── config/                  # Application and guardrail configuration
    ├── embedding/               # Embedding generation and validation
    ├── evaluation/              # Benchmark loading and evaluators
    ├── generation/              # LLM answer generation
    ├── guardrails/              # Input and output validation
    ├── ingestion/               # Document processing, chunking and BM25 index build
    ├── pipeline/                # Online RAG and offline evaluation orchestration
    ├── retrieval/               # Dense, sparse, fusion and reranking components
    ├── utility/                 # Operational and index inspection/rebuild scripts
    └── utils/                   # Shared utilities such as logging
```

Generated artifacts such as processed documents, chunks, embeddings, ChromaDB files, BM25 indexes and evaluation reports are local outputs. They are excluded from version control. The source corpus is also kept outside the repository unless its licence explicitly permits redistribution.

## 8. Setup and Configuration

### Prerequisites

- Python version compatible with the dependencies in `requirements.txt`
- Git
- A Google Gemini API key for answer generation
- Access to the legal-document corpus and benchmark files used for the project

### Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd legal-document-intelligence-rag
```

### Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Configure environment variables

Create a local `.env` file using `.env.example` as a template. Add your own API key; never commit `.env` or secret values.

The source corpus and benchmark paths are currently configured in `src/config/settings.py`. The local absolute data path will be made configurable before the repository is presented as portable for other users.

## 9. Running the Application

Run Python modules from the `src` directory with the project virtual environment activated.

The online RAG pipeline can currently be exercised with:

```bash
python -m pipeline.test_online_pipeline
```

The guardrail tests can be run with:

```bash
python -m guardrails.test_input_guardrail
python -m guardrails.test_output_guardrail
```

The evaluation modules and orchestrator are part of the project. Their final execution instructions will be added here after the ingestion/index rebuild workflow and offline evaluation sequence have been revalidated.

## 10. Data, Reproducibility and Repository Scope

The repository is intended to contain source code and documentation, not private or redistributable copies of legal documents or generated indexes.

The following are kept local and excluded from version control:

- Original legal source documents and benchmark data unless redistribution is explicitly permitted.
- Processed documents and generated chunks.
- Embedding outputs, ChromaDB storage and BM25 index artifacts.
- Generated evaluation JSON and Excel reports.
- API keys, `.env` files and local virtual environments.

The ingestion and index-building workflow will provide the reproducible path for creating the derived artifacts from an appropriately licensed local corpus.

## 11. Limitations and Future Work

This is a learning and demonstration implementation. It should not be used to provide legal advice or make legal decisions.

The current project focuses on the core RAG workflow and its evaluation. Further work will document and assess the additional capabilities needed for an enterprise deployment, such as:

- Authentication, authorization and document-level access controls.
- Data privacy, retention and audit requirements.
- Resilience, availability and recovery objectives.
- API, deployment and infrastructure design.
- Latency, throughput and cost budgets.
- Monitoring, tracing and operational quality measurement.
- More comprehensive security testing and evaluation.

These are areas for future design and validation, not claims about capabilities already implemented.

---

**Project status:** Hands-on RAG learning and implementation project. Documentation and reproducibility steps are being finalized alongside the code.
