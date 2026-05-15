import os
from pypdf import PdfReader

INPUT_PATH = "data/raw"
OUTPUT_PATH = "data/processed"


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def save_text(filename, text):
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    output_file = os.path.join(
        OUTPUT_PATH,
        filename.replace(".pdf", ".txt")
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    return output_file


def ingest_all_pdfs():
    if not os.path.exists(INPUT_PATH):
        print("Input folder tidak ditemukan:", INPUT_PATH)
        return

    files = [f for f in os.listdir(INPUT_PATH) if f.endswith(".pdf")]

    if len(files) == 0:
        print("Tidak ada file PDF di:", INPUT_PATH)
        return

    for file in files:
        pdf_path = os.path.join(INPUT_PATH, file)

        print("Processing:", file)

        text = extract_text_from_pdf(pdf_path)

        print("Text length:", len(text))

        output_file = save_text(file, text)

        print("Saved:", output_file)


if __name__ == "__main__":
    ingest_all_pdfs()