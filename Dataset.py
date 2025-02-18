from datasets import load_dataset
import pandas as pd

# Load dataset
truthfulqa = load_dataset("truthful_qa", "generation", split="validation")

# Convert to Pandas DataFrame
df = pd.DataFrame({"question": truthfulqa["question"], "correct_answer": truthfulqa["best_answer"]})

# Save table for paper inclusion
df.head(10).to_csv("truthfulqa_sample.csv", index=False)
