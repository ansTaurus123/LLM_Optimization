from transformers import pipeline

# Load Zephyr-7B model
generator = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta")

# Generate hallucinated responses
hallucinated_responses = []
for q in df["question"][:500]:
    data = generator(q, max_length=100, do_sample=True)
    hallucinated_responses.append(data[0]["generated_text"])

df["hallucinated_response"] = hallucinated_responses
df.to_csv("generated_hallucinations.csv", index=False)
