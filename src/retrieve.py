import chromadb
from sentence_transformers import SentenceTransformer

# HARUS sama dengan yang di ingest
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="vectordb/chroma")
collection = client.get_collection(name="research")


def get_embedding(text: str):
    return model.encode(text).tolist()


def retrieve(query, k=5, source=None):
    query_embedding = get_embedding(query)

    where_filter = {"source": source} if source else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where_filter
    )

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # ambil hasil valid (tanpa threshold aneh)
    paired = list(zip(documents, distances))

    # sort by similarity (lebih kecil = lebih relevan)
    paired.sort(key=lambda x: x[1])

    return [doc for doc, _ in paired]


if __name__ == "__main__":
    query = "overview of battery energy storage system research"

    print("\n=== ALL ARTICLES ===")
    hasil = retrieve(query)

    for i, doc in enumerate(hasil):
        print(f"\n--- CHUNK {i+1} ---\n")
        print(doc)

    print("\n=== ONLY artikel3 ===")
    hasil2 = retrieve(query, source="artikel3")

    for i, doc in enumerate(hasil2):
        print(f"\n--- CHUNK {i+1} ---\n")
        print(doc)