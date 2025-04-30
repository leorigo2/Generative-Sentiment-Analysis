import copy
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import csv
import urllib.request
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/test_text.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
html = html[:-1]
test_text = [row for row in html]

mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/test_labels.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
html = html[:-1]
test_labels = ["Negative" if row == "0" else "Neutral" if row == "1" else "Positive" for row in html]

mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/val_text.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
html = html[:-1]
val_text = [row for row in html]

mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/val_labels.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
html = html[:-1]
val_labels = [" Negative" if row == "0" else " Neutral" if row == "1" else " Positive" for row in html]


# Define allowed sentiment labels
allowed_labels = [" Positive", " Negative", " Neutral"]  # note the leading space

# Map them to token IDs
allowed_token_ids = [tokenizer(label, add_special_tokens=False)["input_ids"][0] for label in allowed_labels]

# Custom logits processor to only allow certain tokens on the first output token
class RestrictFirstTokenProcessor(torch.nn.Module):
    def __init__(self, allowed_ids):
        super().__init__()
        self.allowed_ids = allowed_ids
        self.called = False

    def __call__(self, input_ids, scores):
        if not self.called:
            mask = torch.full_like(scores, float('-inf'))
            for idx in self.allowed_ids:
                mask[:, idx] = scores[:, idx]
            scores = mask
            self.called = True
        return scores

def analyze_sentiment(text: str, max_new_tokens: int = 5) -> str:
    prompt = f"""Analyze the sentiment of the following text and complete the sentence.
    Text: {text}
    This text expresses sentiment:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Restrict only the first generated token to be one of the three
    logits_processor = LogitsProcessorList()
    logits_processor.append(RestrictFirstTokenProcessor(allowed_token_ids))

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            logits_processor=logits_processor,
            pad_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = full_output[len(prompt):].strip()
    return f"{generated}"

# Example
if __name__ == "__main__":
    correct = 0 
    posCorrect = posNumber = 0
    neuCorrect = neuNumber = 0
    negCorrect = negNumber = 0

    for i, text in enumerate(test_text):
        result = analyze_sentiment(text)  # First, generate the model's output

        # Clean whitespace if needed
        result = result.strip()
        true_label = test_labels[i].strip()

        # Global accuracy
        if result == true_label:
            correct += 1

        # Per-class tracking
        if true_label == "Positive":
            if result == "Positive":
                posCorrect += 1
            posNumber += 1

        elif true_label == "Neutral":
            if result == "Neutral":
                neuCorrect += 1
            neuNumber += 1

        elif true_label == "Negative":
            if result == "Negative":
                negCorrect += 1
            negNumber += 1

    # Print results
    print(f"Overall Accuracy: {correct / len(test_labels):.4f}")
    print(f"Positive Accuracy: {posCorrect} / {posNumber} ({posCorrect / posNumber:.4f})")
    print(f"Neutral Accuracy: {neuCorrect} / {neuNumber} ({neuCorrect / neuNumber:.4f})")
    print(f"Negative Accuracy: {negCorrect} / {negNumber} ({negCorrect / negNumber:.4f})")