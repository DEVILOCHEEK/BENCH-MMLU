import os
import json
import argparse
import pandas as pd
import requests
from tqdm import tqdm

def format_example(question, options, answer=None):
    prompt = f"Question: {question}\n"
    for i, option in enumerate(options):
        prompt += f"{chr(ord('A') + i)}. {option}\n"
    prompt += "Answer:"
    if answer:
        prompt += f" {answer}"
    return prompt

def gen_prompt(dev_df, test_row):
    prompt = "The following are multiple choice questions (with answers).\n\n"
    for _, row in dev_df.iterrows():
        prompt += format_example(row[0], row[1:5].tolist(), row[5]) + "\n\n"
    prompt += format_example(test_row[0], test_row[1:5].tolist())
    return prompt

def chat_completion(prompt, model, api_key):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://yourdomain.com",  # заміни своїм доменом
        "X-Title": "llm-benchmarking",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 5
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code} {response.text}")
    return response.json()

def eval(args, subject, api_key, dev_df, test_df):
    cors = []
    all_probs = []

    for i in tqdm(range(len(test_df)), desc=f"Evaluating {subject}"):
        prompt = gen_prompt(dev_df, test_df.iloc[i])
        gt = test_df.iloc[i][5]

        try:
            response = chat_completion(prompt, args.model, api_key)
            choice = response["choices"][0]
            pred = choice["message"]["content"].strip()[0]

            cors.append(pred == gt)

            logprobs = choice.get("logprobs")
            if logprobs and "top_logprobs" in logprobs:
                last_probs = logprobs["top_logprobs"][-1]
                cleaned = {k.strip(): v for k, v in last_probs.items()}
            else:
                cleaned = {}

            all_probs.append(cleaned)

        except Exception as e:
            print(f"[!] Error at index {i}: {e}")
            cors.append(False)
            all_probs.append({})

    acc = sum(cors) / len(cors)
    return cors, acc, all_probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--model", type=str, default="openai/gpt-4o")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENROUTER_API_KEY", ""))
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("API key must be provided via --api_key or OPENROUTER_API_KEY env var")

    os.makedirs(args.save_dir, exist_ok=True)

    test_dir = os.path.join(args.data_dir, "test")
    dev_dir = os.path.join(args.data_dir, "dev")

    subjects = sorted([
        f.replace("_test.csv", "")
        for f in os.listdir(test_dir) if f.endswith("_test.csv")
    ])

    for subject in subjects:
        dev_df = pd.read_csv(os.path.join(dev_dir, subject + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(test_dir, subject + "_test.csv"), header=None)

        cors, acc, probs = eval(args, subject, args.api_key, dev_df, test_df)

        output = {
            "subject": subject,
            "accuracy": acc,
            "correctness": cors,
            "logprobs": probs,
        }

        out_path = os.path.join(args.save_dir, f"{subject}.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"[✔] {subject} — accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
