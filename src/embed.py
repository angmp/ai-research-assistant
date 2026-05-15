import os
import chromadb
from sentence_transformers import SentenceTransformer

CHUNK_PATH = "data/chunks"
DB_PATH = "vectordb/chroma"

# embedding model (ringan tapi bagus)
model = SentenceTransformer("all-MiniLM-L6-v2")

# init chroma db
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="rag_collection")


def load_chunks():
    chunks = []
    ids = []

    for file in os.listdir(CHUNK_PATH):
        if file.endswith(".txt"):
            file_path = os.path.join(CHUNK_PATH, file)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks.append(text)
            ids.append(file)

    return chunks, ids


def embed_and_store():
    chunks, ids = load_chunks()

    if len(chunks) == 0:
        print("No chunks found!")
        return

    print("Total chunks:", len(chunks))

    embeddings = model.encode(chunks).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    print("Embedding selesai dan masuk ke vector DB")


if __name__ == "__main__":
    embed_and_store()