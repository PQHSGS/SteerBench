"""Prepare CAST refusal data for ml-lineas JsonSubsetsDataset.

Produces a JSON file with "source" (compliant) and "target" (refusal) subsets,
each containing question + response-prefix strings. Pairs are aligned by index.
Model: google/gemma-2-2b (base, not -it), no chat template applied.
"""
import json
import itertools
import os

BASE_DIR = "/home/caotue/SAESteeringBench"
RESPONSE_FILE = f"{BASE_DIR}/TrainDataset/behaviour/refusal/CAST/behaviour_refusal.json"
QUESTION_FILE = f"{BASE_DIR}/TrainDataset/behaviour/refusal/CAST/alpaca.json"
OUTPUT_DIR = "/home/caotue/SAESteeringBench/Code/ml-lineas/data/refusal"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(RESPONSE_FILE) as f:
    responses = json.load(f)
with open(QUESTION_FILE) as f:
    questions = json.load(f)

compliant = responses["compliant_responses"]
non_compliant = responses["non_compliant_responses"]
response_pairs = list(zip(compliant, non_compliant))

source_texts = []
target_texts = []

# Cycle through 100 response-pair types to cover all ~2100 questions
for q_entry, (agree, refuse) in zip(questions, itertools.cycle(response_pairs)):
    q = q_entry["question"]
    source_texts.append(f"{q}\n{agree}")
    target_texts.append(f"{q}\n{refuse}")

data = {"source": source_texts, "target": target_texts}
output_path = f"{OUTPUT_DIR}/refusal_data.json"
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"Created {len(source_texts)} paired examples ({len(response_pairs)} response types, {len(questions)} questions)")
print(f"Output: {output_path}")
print(f"Example source: {source_texts[0]!r}")
print(f"Example target: {target_texts[0]!r}")
