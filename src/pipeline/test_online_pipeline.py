from pipeline.rag_orchestrator import RAGOrchestrator

rag_pipeline = RAGOrchestrator()

result = rag_pipeline.ask(
    "What are the termination conditions in this agreement?"
)

print("\nStatus:", result["status"])
print("\nAnswer:", result["answer"])
print("\nSources:")

for source in result["sources"]:
    print(source)