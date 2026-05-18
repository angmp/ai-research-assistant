import os
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

INPUT_PATH = "data/raw"
DB_PATH = "vectordb/chroma"

# embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# chroma db
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="research")


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # biar ada overlap context

    return chunks


def ingest_all_pdfs():
    files = [f for f in os.listdir(INPUT_PATH) if f.endswith(".pdf")]

    if not files:
        print("Tidak ada PDF")
        return

    all_docs = []
    all_embeddings = []
    all_ids = []
    all_metadatas = []

    for file in files:
        pdf_path = os.path.join(INPUT_PATH, file)
        source = file.replace(".pdf", "")

        print("Processing:", file)

        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)

        embeddings = model.encode(chunks).tolist()

        for i, chunk in enumerate(chunks):
            all_docs.append(chunk)
            all_embeddings.append(embeddings[i])
            all_ids.append(f"{source}_chunk_{i}")
            all_metadatas.append({
                "source": source
            })

    # simpan ke vector DB
    collection.add(
        documents=all_docs,
        embeddings=all_embeddings,
        ids=all_ids,
        metadatas=all_metadatas
    )

    print(" Ingest selesai (RAG ready)")


if __name__ == "__main__":
    ingest_all_pdfs()