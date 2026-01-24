# Generative Sentiment Analysis

A comparative study of sentiment analysis using generative language models with two different decoding strategies: **logit constraints** and **output post-processing**.

## Overview

This project explores sentiment analysis as a text classification task using small-scale generative language models (Llama 3.2 1B and Qwen 1.5 1.8B). Rather than traditional classification approaches, we leverage the generative capabilities of LLMs with two distinct strategies to constrain outputs to sentiment labels (Positive, Negative, Neutral).

## Project Structure

```
Generative-Sentiment-Analysis/
│
├── logit-constraints/
│   ├── logit-constraints-chat.py      # Qwen Chat model with logit constraints
│   ├── logit-constraints-instruct.py  # Llama Instruct model with logit constraints
│   ├── qwenResults.txt                # Results for Qwen model
│   └── llamaResults.txt               # Results for Llama model
│
├── output-postprocessing/
│   ├── output-postprocessing-chat.py      # Qwen Chat model with output post-processing
│   ├── output-postprocessing-instruct.py  # Llama Instruct model with output post-processing
│   ├── qwenResults.txt                    # Results for Qwen model
│   └── llamaResults.txt                   # Results for Llama model
│
└── README.md
```

## Approaches

### 1. Logit Constraints
Located in `logit-constraints/`

This approach restricts the model's output space at the token generation level by applying a logits processor that masks all tokens except the allowed sentiment labels during the first token generation.

### 2. Output Post-processing
Located in `output-postprocessing/`

This approach allows the model to generate a free-form response (up to 10-20 tokens) and then extracts the sentiment label from the generated text using pattern matching.

## Models

### Qwen 1.5 1.8B Chat
- Model: `Qwen/Qwen1.5-1.8B-Chat`
- Uses chat template with system and user messages
- Supports both zero-shot and few-shot prompting

### Llama 3.2 1B Instruct
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Uses instruct-style prompting
- Supports both zero-shot and few-shot prompting

## Dataset

**TweetEval Sentiment Dataset**
- Source: [CardiffNLP TweetEval](https://github.com/cardiffnlp/tweeteval)
- Sentiment Analysis subfolder
- Test set with sentiment labels: Negative (0), Neutral (1), Positive (2)

## Requirements

```bash
pip install -r requirements.txt
```

## Configuration

Each script contains configurable parameters at the top:

```python
prompting = 0 or 1           # 0: few-shot, 1: zero-shot

# For output-postprocessing only:
max_new_tokens = 10-20       # Number of tokens to generate
temperature = 0.7            # Sampling temperature
top_k = 5                    # Top-k sampling
top_p = 0.9                  # Top-p sampling
```

## Usage

### Running Logit Constraints Approach

**For Qwen Chat Model:**
```bash
python logit-constraints/logit-constraints-chat.py
```

**For Llama Instruct Model:**
```bash
python logit-constraints/logit-constraints-instruct.py
```

### Running Output Post-processing Approach

**For Qwen Chat Model:**
```bash
python output-postprocessing/output-postprocessing-chat.py
```

**For Llama Instruct Model:**
```bash
python output-postprocessing/output-postprocessing-instruct.py
```

## Output

Each script produces the following results:

1. **Execution Time**: Total time to process the test set
2. **Overall Accuracy**: Total correct predictions / total samples
3. **Per-Class Accuracy**: Accuracy for Positive, Neutral, and Negative classes
4. **Confusion Matrix**: Visual representation of prediction performance

### Example Output:
```
--- Results ---
Prompt: Text: {}...
Time: 245.32 s
Overall Accuracy: 1234/2567 (0.4807)
Positive Accuracy: 456/789 (0.5779)
Neutral Accuracy: 234/567 (0.4127)
Negative Accuracy: 544/1211 (0.4493)

Confusion Matrix:
[Visual matplotlib plot displayed]
```

## Results Analysis

Results are saved in `.txt` files within each approach directory:
- `qwenResults.txt`: Results for Qwen 1.5 1.8B Chat
- `llamaResults.txt`: Results for Llama 3.2 1B Instruct


## Citation

If you use the TweetEval dataset, please cite:
```
@inproceedings{barbieri2020tweeteval,
  title={TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification},
  author={Barbieri, Francesco and Camacho-Collados, Jose and Espinosa-Anke, Luis and Neves, Leonardo},
  booktitle={Findings of EMNLP},
  year={2020}
}
```
