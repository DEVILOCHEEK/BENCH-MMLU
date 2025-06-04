import argparse
import os
import numpy as np
import pandas as pd
import time
import requests

from crop import crop  # твоя функція обрізки промпта

OPENROUTER_API_KEY = "INSERT_YOUR_OPENROUTER_KEY_HERE"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

choices = ["A", "B", "C", "D"]

def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator / denominator

def format_subject(subject):
    return " ".join(subject.split("_"))

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += f"\n{choices[j]}. {df.iloc[idx, j+1]}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {df.iloc[idx, k + 1]}\n\n"
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def query_openrouter(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": 100,
        "echo": True
    }
    while True:
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API error: {e}. Retrying in 1 sec...")
            time.sleep(1)

def eval(args, subject, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]

    for i in range(test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]

        resp = query_openrouter(prompt)
        top_logprobs = resp["choices"][0]["logprobs"]["top_logprobs"][-1]

        lprobs = []
        for ans in answers:
            key = f" {ans}"  # note leading space like в оригіналі
            lprobs.append(top_logprobs.get(key, -100))

        pred = answers[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    print(f"Average accuracy {acc:.3f} - {subject}")

    return np.array(cors), acc, np.array(all_probs)

def main(args):
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print("Subjects:", subjects)
    print("Arguments:", args)

    all_cors = []

    for subject in subjects:
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", f"{subject}_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", f"{subject}_test.csv"), header=None)

        cors, acc, probs = eval(args, subject, dev_df, test_df)
        all_cors.append(cors)

        test_df[f"gpt4o_correct"] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df[f"gpt4o_choice{choice}_probs"] = probs[:, j]
        test_df.to_csv(os.path.join(args.save_dir, f"{subject}_results_gpt4o.csv"), index=False)

    weighted_acc = np.mean(np.concatenate(all_cors))
    print(f"Average accuracy across subjects: {weighted_acc:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    args = parser.parse_args()
    main(args)
