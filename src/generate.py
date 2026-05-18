from retrieve import retrieve
from transformers import pipeline

# load local text generation model
generator = pipeline(
    "text-generation",
    model="google/flan-t5-base"
)

def generate_answer(query):

    # ambil chunk relevan
    results = retrieve(query)

    context = "\n".join(results)

    prompt = f"""
Answer the question based on the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    response = generator(
        prompt,
        max_new_tokens=200
    )

    return response[0]["generated_text"]


if __name__ == "__main__":

    question = input("Question: ")

    answer = generate_answer(question)

    print("\nANSWER:\n")
    print(answer)