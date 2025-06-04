import argparse
import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from openai import OpenAI

# Отримуємо ключ з середовища
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Set environment variable OPENROUTER_API_KEY with your OpenRouter API key.")

# Ініціалізуємо клієнта OpenAI з передачею ключа
client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

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

def crop(prompt, max_tokens=4096):
    # Проста обрізка за довжиною (символи) — можна покращити під токени
    max_len = max_tokens * 4  # приблизно 4 символи на токен
    if len(prompt) > max_len:
        return prompt[-max_len:]
    return prompt

for i in tqdm(range(test_df.shape[0]), desc="Evaluating sample"):
    example_text = test_df.iloc[i, 0]
    correct_letter = test_df.iloc[i, -1]

    # Знайдемо індекс правильного варіанту (0-based) серед A, B, C, D
    correct_index = choices.index(correct_letter)
    correct_text = test_df.iloc[i, 1 + correct_index]  # варіант тексту відповіді

    tqdm.write(f"Приклад {i+1}: {example_text}")
    tqdm.write(f"Правильна відповідь: {correct_text}")

    k = args.ntrain
    prompt_end = format_example(test_df, i, include_answer=False)
    train_prompt = gen_prompt(dev_df, subject, k)
    prompt = train_prompt + prompt_end

    # Обрізка, якщо дуже довго
    while crop(prompt) != prompt:
        k -= 1
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        if k == 0:
            break

    label = correct_letter

    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            break
        except Exception as e:
            print(f"API error: {e}. Retrying in 1 second...")
            time.sleep(1)

    answer_text = response.choices[0].message.content.strip()
    tqdm.write(f"Відповідь моделі: {answer_text}")

    pred = None
    for ans in answers:
        if ans in answer_text:
            pred = ans
            break

    if pred is None:
        print(f"Warning: model answer '{answer_text}' не співпадає з варіантами {answers}. Встановлюємо дефолтний варіант '{answers[0]}'.")
        pred = answers[0]

    pred_index = choices.index(pred)
    pred_text = test_df.iloc[i, 1 + pred_index]

    tqdm.write(f"Інтерпретація відповіді моделі: {pred_text}")

    cors.append(pred == label)

    # Тимчасово задаємо рівномірні ймовірності по варіантах
    all_probs.append([1/num_choices] * num_choices)

    acc = np.mean(cors)
    print(f"Average accuracy for {subject}: {acc:.3f}")

    return np.array(cors), acc, np.array(all_probs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    args = parser.parse_args()

    subjects_path = os.path.join(args.data_dir, "test")
    if not os.path.exists(subjects_path):
        raise FileNotFoundError(f"Test directory not found: {subjects_path}")

    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(subjects_path) if f.endswith("_test.csv")])
    print(f"Subjects found: {subjects}")

    os.makedirs(args.save_dir, exist_ok=True)

    for subject in subjects:
        print(f"Processing subject: {subject}")
        dev_path = os.path.join(args.data_dir, "dev", f"{subject}_dev.csv")
        test_path = os.path.join(args.data_dir, "test", f"{subject}_test.csv")

        if not os.path.isfile(dev_path) or not os.path.isfile(test_path):
            print(f"Skipping {subject} due to missing dev or test files.")
            continue

        dev_df = pd.read_csv(dev_path, header=None)[:args.ntrain]
        test_df = pd.read_csv(test_path, header=None)

        cors, acc, probs = eval(args, subject, dev_df, test_df)

        test_df[f"gpt4o_correct"] = cors
        for j, choice in enumerate(choices):
            test_df[f"gpt4o_choice{choice}_probs"] = probs[:, j]

        save_path = os.path.join(args.save_dir, f"{subject}.csv")
        test_df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()
