import os
import chromadb
from sentence_transformers import SentenceTransformer

CHUNK_PATH = "data/chunks"
DB_PATH = "vectordb/chroma"

# =========================
# EMBEDDING MODEL
# =========================
model = SentenceTransformer(
    "ibm-granite/granite-embedding-97m-multilingual-r2"
)

# =========================
# INIT CHROMA DB
# =========================
client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name="research"
)

# =========================
# EMBEDDING FUNCTION
# =========================
def get_embedding(text: str):
    return model.encode(text).tolist()


# =========================
# LOAD CHUNKS + METADATA
# =========================
def load_chunks():

    chunks = []
    ids = []
    metadatas = []

    for file in os.listdir(CHUNK_PATH):

        if file.endswith(".txt"):

            file_path = os.path.join(CHUNK_PATH, file)

            # ambil nama source asli
            # contoh:
            # artikel3_chunk_0.txt
            # -> artikel3
            source = file.split("_chunk_")[0]

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks.append(text)

            ids.append(file)

            metadatas.append({
                "source": source,
                "chunk_file": file
            })

    return chunks, ids, metadatas


# =========================
# INGEST KE VECTOR DB
# =========================
def embed_and_store():

    chunks, ids, metadatas = load_chunks()

    if len(chunks) == 0:
        print("No chunks found!")
        return

    print("Total chunks:", len(chunks))

    # bikin embedding
    embeddings = model.encode(
        chunks,
        show_progress_bar=True
    ).tolist()

    # simpan ke chromadb
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    print("Embedding selesai + masuk vector DB")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    embed_and_store()