import copy
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import csv
import urllib.request
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time
from kagglehub import KaggleDatasetAdapter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

models = {
    "qwen": "Qwen/Qwen1.5-1.8B-Chat",
}

max_new_tokens = 20
dataset  = "github" # "huggingface" or "github"
model_name = "qwen"

# Generation parameters
temperature = 0.7 # To invalidate set to 1.0 
top_k = 5 # To invalidate set to 0.0
top_p = 0.9 # To invalidate set to 1.0

tokenizer = AutoTokenizer.from_pretrained(models[model_name])
model = AutoModelForCausalLM.from_pretrained(models[model_name])


# Seleziona il dispositivo
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
gh_test_text = [row for row in html[:1000]]

mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/test_labels.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
html = html[:-1]
gh_test_labels = ["Negative" if row == "0" else "Neutral" if row == "1" else "Positive" for row in html[:1000]]

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

allowed_labels = ["Positive", "Negative", "Neutral"]

system_prompt_zeroshot = "You are a helpful assistant that recognizes the sentiment of a text and chooses between Positive, Negative and Neutral."

system_prompt_fewshot = f"""You are a helpful assistant that recognizes the sentiment of a text and chooses between Positive, Negative and Neutral. Following this examples:
        Text: "{sorted_pos[-1]}"
        Sentiment: Positive

        Text: "{sorted_pos[-2]}"
        Sentiment: Positive

        Text: "{sorted_pos[-3]}"
        Sentiment: Positive

        Text: "{sorted_neg[-1]}"
        Sentiment: Negative

        Text: "{sorted_neg[-2]}"
        Sentiment: Negative

        Text: "{sorted_neg[-3]}"
        Sentiment: Negative

        Text: "{sorted_neu[-1]}"
        Sentiment: Neutral

        Text: "{sorted_neu[-2]}"
        Sentiment: Neutral

        Text: "{sorted_neu[-3]}"
        Sentiment: Neutral
    """

prompts = [
    """Text: {}"""
]

def extract_label_from_output(output_text, allowed_labels):
    output_text = output_text.lower()
    for label in allowed_labels:
        if label.lower() in output_text:
            return label
    return "Unknown"

def analyze_sentiment(text, prompt, prompting):
      
    prompt = prompt.replace("{}", text)
      
    messages = [
        {"role": "system", "content": (system_prompt_zeroshot if prompting == 0 else system_prompt_fewshot)},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(device)

    if 'attention_mask' not in inputs:
        inputs['attention_mask'] = (inputs['input_ids'] != tokenizer.pad_token_id).long()

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=inputs['attention_mask']
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    predicted_label = extract_label_from_output(response, allowed_labels)

    return predicted_label

# Main
if __name__ == "__main__":
    for prompting in range(2):
        prompt_name = "fewshot" if prompting == 1 else "zeroshot"
        results = {}
        
        for prompt in prompts:
            start_time = time.time()

            correct = 0
            posCorrect = posNumber = 0
            neuCorrect = neuNumber = 0
            negCorrect = negNumber = 0
            
            pred = []
            true = []

            for i, text in enumerate(test_text[dataset]):
                result = analyze_sentiment(text, prompt, prompting)
                true_label = test_labels[dataset][i]
                
                pred.append(result)
                true.append(true_label)
                print

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
        plt.savefig(f"confusion_matrix_{model_name}_{prompt_name}.jpg")
        print(f"\nSaved confusion matrix as: confusion_matrix_{model_name}_{prompt_name}.jpg\n")