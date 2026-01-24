import copy
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import csv
import urllib.request
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
import numpy as np
import time
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

models = {
    "llama": "meta-llama/Llama-3.2-1B-Instruct"
}

max_new_tokens = 10 
dataset  = "github" 
model = "llama"
prompting = 1 # 0 corresponds to few shot, 1 to zero shot

# Generation parameters
temperature = 0.7 # To invalidate set to 1.0 
top_k = 5 # To invalidate set to 0.0
top_p = 0.9 # To invalidate set to 1.0

tokenizer = AutoTokenizer.from_pretrained(models[model])
model = AutoModelForCausalLM.from_pretrained(models[model])


# Select device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Working on:", device)
device = torch.device(device)
model.to(device)

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
    "github": gh_test_text 
}

test_labels = {
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

allowed_labels = ["Positive", "Negative", "Neutral"]

prompts = [
    [
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
        Sentiment: Neutral

        Text: "{}"
        Sentiment: Neutral

        Text: "{}"
        Sentiment:"""
    ],
    [
        """Analyze the sentiment of the following text and complete the sentence.
        Text: {}
        This text expresses sentiment:""",
        """Analyze the sentiment of the following text and complete the sentence, choosing only between [Positive, Negative, Neutral]
        Text: {}
        This text expresses sentiment:""",
        """Is the following text expressing "Positive", "Negative" or "Neutral" sentiment?
        Text: {}
        Answer:"""
    ]
]

def extract_label_from_output(output_text, allowed_labels):
    output_text = output_text.lower()
    for label in allowed_labels:
        if label.lower() in output_text:
            return label
    return "Unknown"

def analyze_sentiment(text, prompt):
    if prompting == 0:
      prompt = prompt.format(sorted_pos[-1], sorted_pos[-2], sorted_pos[-3], sorted_neg[-1], sorted_neg[-2], sorted_neg[-3], sorted_neu[-1], sorted_neu[-2], sorted_neu[-3], text)
    else:
      prompt = prompt.format(text)    
      inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = full_output[len(prompt):].strip()
    predicted_label = extract_label_from_output(generated_text, allowed_labels)

    return predicted_label

# Main
if __name__ == "__main__":
    for prompting in range(2):
        prompt_name = "zeroshot" if prompting == 1 else "fewshot"
        results = {}

        for prompt in prompts[prompting]:
            start_time = time.time()

            correct = 0
            posCorrect = posNumber = 0
            neuCorrect = neuNumber = 0
            negCorrect = negNumber = 0

            pred = []
            true = []

            for i, text in enumerate(test_text[dataset]):
                result = analyze_sentiment(text, prompt)
                true_label = test_labels[dataset][i]
                
                pred.append(result)
                true.append(true_label)

                if result == true_label:
                    correct += 1

                if true_label == "Positive":
                    posNumber += 1
                    if result == "Positive":
                        posCorrect += 1

                elif true_label == "Neutral":
                    neuNumber += 1
                    if result == "Neutral":
                        neuCorrect += 1

                elif true_label == "Negative":
                    negNumber += 1
                    if result == "Negative":
                        negCorrect += 1

            end_time = time.time()
            elapsed_time = end_time - start_time
            total_samples = len(test_text[dataset])
            overall_accuracy = correct / total_samples
            results[prompt] = {
                "overall_accuracy": overall_accuracy,
                "posCorrect": posCorrect,
                "posNumber": posNumber,
                "neuCorrect": neuCorrect,
                "neuNumber": neuNumber,
                "negCorrect": negCorrect,
                "negNumber": negNumber,
                "elapsed_time": elapsed_time,
                "total_samples": total_samples,
                "pred": pred,
                "true": true,
                "correct": correct,
            }

        # Select the best prompt based on overall accuracy
        best_prompt = max(results, key=lambda x: results[x]["overall_accuracy"])
        best_result = results[best_prompt]

        print("--- Results ---")
        print(f"Prompting: {prompt_name}")
        print(f"Best Prompt: {best_prompt[:60]}...")
        print(f"Time: {best_result['elapsed_time']:.2f} s")
        print(f"Overall Accuracy: {best_result['correct']}/{best_result['total_samples']} ({best_result['overall_accuracy']:.4f})")
        print(f"Positive Accuracy: {best_result['posCorrect']}/{best_result['posNumber']} ({best_result['posCorrect'] / best_result['posNumber']:.4f})")
        print(f"Neutral Accuracy: {best_result['neuCorrect']}/{best_result['neuNumber']} ({best_result['neuCorrect'] / best_result['neuNumber']:.4f})")
        print(f"Negative Accuracy: {best_result['negCorrect']}/{best_result['negNumber']} ({best_result['negCorrect'] / best_result['negNumber']:.4f})")

        print("\nConfusion Matrix:")
        cm = confusion_matrix(best_result['true'], best_result['pred'], labels=allowed_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=allowed_labels)
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{model}_{prompt_name}.jpg")
        print(f"\nSaved confusion matrix as: confusion_matrix_{model}_{prompt_name}.jpg\n")