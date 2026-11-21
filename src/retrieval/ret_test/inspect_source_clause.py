from pathlib import Path
from config.settings import PROCESSED_DOCUMENTS_PATH

# Locate the preprocessed source document.
target_file = "CopAcc_NDA-and-ToP-Mentors_2.0_2017.txt"

with PROCESSED_DOCUMENTS_PATH.open("r", encoding="utf-8") as file:
    import json
    documents = [json.loads(line) for line in file if line.strip()]

document = next(
    (
        item for item in documents
        if item["source_file"] == target_file
    ),
    None
)

if document is None:
    raise ValueError(f"Document not found: {target_file}")

# Inspect the source text around the benchmark evidence span.
text = document["processed_text"]
gold_start = 11461
gold_end = 11963

# Display surrounding context to identify paragraph boundaries.
context_start = max(0, gold_start - 1000)
context_end = min(len(text), gold_end + 1000)

print(f"Document: {target_file}")
print(f"Document length: {len(text)} characters")
print(f"Gold evidence span: {gold_start}-{gold_end}")

print("\nSOURCE TEXT WITH SURROUNDING CONTEXT\n")
print(text[context_start:context_end])

print("\nPARAGRAPH BOUNDARIES")

for position in range(context_start, context_end):
    if text[position:position + 2] == "\n\n":
        print(f"Paragraph boundary at character: {position}")