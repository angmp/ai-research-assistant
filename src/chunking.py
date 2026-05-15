import os

INPUT_PATH = "data/processed"
OUTPUT_PATH = "data/chunks"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return split_text(text)


def save_chunks(filename, chunks):
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    base_name = filename.replace(".txt", "")

    for i, chunk in enumerate(chunks):
        output_file = os.path.join(
            OUTPUT_PATH,
            f"{base_name}_chunk_{i}.txt"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(chunk)


def chunk_all_files():
    if not os.path.exists(INPUT_PATH):
        print("Folder tidak ditemukan:", INPUT_PATH)
        return

    files = [f for f in os.listdir(INPUT_PATH) if f.endswith(".txt")]

    if len(files) == 0:
        print("Tidak ada file .txt di:", INPUT_PATH)
        return

    for file in files:
        file_path = os.path.join(INPUT_PATH, file)

        print("Processing:", file)

        chunks = process_file(file_path)

        save_chunks(file, chunks)

        print("Done chunks:", len(chunks))


if __name__ == "__main__":
    chunk_all_files()