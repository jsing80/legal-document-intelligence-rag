
import chromadb

from config.settings import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME

print("Configured ChromaDB directory:", CHROMA_DB_DIR.resolve())
print("Expected collection:", CHROMA_COLLECTION_NAME)

client = chromadb.PersistentClient(
    path=str(CHROMA_DB_DIR)
)

collections = client.list_collections()

print("\nCollections available in configured directory:")

if not collections:
    print("No collections found.")
else:
    for collection in collections:
        print(
            f"Name: {collection.name}, "
            f"Count: {collection.count()}"
        )

