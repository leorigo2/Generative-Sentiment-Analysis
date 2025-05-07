import copy
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import csv
import urllib.request
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time
import kagglehub
from kagglehub import KaggleDatasetAdapter

max_new_tokens = 1
dataset  = "huggingface"
model = "gpt"

models = {
    "gpt" : "gpt2",
    "qwen": "Qwen/Qwen3-1.7B",
    "llama": "meta-llama/Llama-3.2-1B-Instruct"
}


tokenizer = AutoTokenizer.from_pretrained(models[model])
model = AutoModelForCausalLM.from_pretrained(models[model])

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Working on:", device)

device = torch.device(device)
model.to(device)


hf_dataset = kagglehub.load_dataset(
    KaggleDatasetAdapter.HUGGING_FACE,
    "mdismielhossenabir/sentiment-analysis",
    path="sentiment_analysis.csv",
)

hf_test_data = hf_dataset
hf_test_text = [item["text"] for item in hf_test_data]
hf_test_labels = [item["sentiment"].strip().capitalize() for item in hf_test_data]

mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/test_text.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
html = html[:-1]
gh_test_text = [row for row in html]

mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/test_labels.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
html = html[:-1]
gh_test_labels = ["Negative" if row == "0" else "Neutral" if row == "1" else "Positive" for row in html]

test_text = {
    "huggingface": hf_test_text,
    "github": gh_test_text 
}

test_labels = {
    "huggingface": hf_test_labels,
    "github": gh_test_labels
}


i = 0
pos_text = []
neg_text = []
neu_text = []

for i, text in enumerate(test_text[dataset]): 
    if(test_labels[dataset][i] == "Positive"):
        pos_text.append(text)
    if(test_labels[dataset][i] == "Neutral"):
        neu_text.append(text)
    if(test_labels[dataset][i] == "Negative"):
        neg_text.append(text)

sorted_pos = sorted(pos_text, key=len)
sorted_neu = sorted(neu_text, key=len)
sorted_neg = sorted(neg_text, key=len)


allowed_labels = [" Positive", " Negative", " Neutral"] 

allowed_token_ids = [tokenizer(label, add_special_tokens=False)["input_ids"][0] for label in allowed_labels]

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

prompts = [
    """Analyze the sentiment of the following texts. Choose one of: Positive, Negative, Neutral. 

    Text: "{}"
    Sentiment: Positive

    Text: "{}"
    Sentiment: Positive

    Text: "{}"
    Sentiment: Positive

    Text: "{}"
    Sentiment: Negative

    Text: "{}"
    Sentiment: Negative

    Text: "{}"
    Sentiment: Negative

    Text: "{}"
    Sentiment: Neutral

    Text: "{}"
    Sentimet: Neutral

    Text: "{}"
    Sentiment: Neutral

    Text: "{}"
    Sentiment:"""

    ]



def analyze_sentiment(text: str, prompt: str) -> str:

    prompt = prompt.format(sorted_pos[-1], sorted_pos[-2], sorted_pos[-3], sorted_neg[-1], sorted_neg[-2], sorted_neg[-3], sorted_neu[-1], sorted_neu[-2], sorted_neu[-3], text)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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

if __name__ == "__main__":

    for prompt in prompts:
        start_time = time.time()

        correct = 0
        posCorrect = posNumber = 0
        neuCorrect = neuNumber = 0
        negCorrect = negNumber = 0

        for i, text in enumerate(test_text[dataset]):
            result = analyze_sentiment(text, prompt)

            result = result.strip()
            true_label = test_labels[dataset][i].strip()

            if result == true_label:
                correct += 1

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

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Prompt: ", prompt)
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print(f"Overall Accuracy: {correct / len(test_labels[dataset]):.4f}")
        print(f"Positive Accuracy: {posCorrect} / {posNumber} ({posCorrect / posNumber:.4f})")
        print(f"Neutral Accuracy: {neuCorrect} / {neuNumber} ({neuCorrect / neuNumber:.4f})")
        print(f"Negative Accuracy: {negCorrect} / {negNumber} ({negCorrect / negNumber:.4f})")
