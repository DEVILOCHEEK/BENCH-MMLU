import tiktoken

# Отримуємо енкодер для моделі GPT-4o
enc = tiktoken.encoding_for_model("gpt-4o")

def encode(text: str) -> list[int]:
    """Кодує текст у список токенів (індексів)."""
    return enc.encode(text)

def decode(tokens: list[int]) -> str:
    """Декодує список токенів назад у текст."""
    return enc.decode(tokens)

def crop_prompt(prompt: str, max_tokens: int = 2048) -> str:
    """Обрізає промпт до максимальної довжини у токенах і повертає текст."""
    tokens = encode(prompt)
    cropped_tokens = tokens[:max_tokens]
    return decode(cropped_tokens)

# Приклад використання:
text = "Привіт, це тестовий текст для токенізації GPT-4o!"
tokens = encode(text)
print("Токени:", tokens)

restored_text = decode(tokens)
print("Відновлений текст:", restored_text)

cropped_text = crop_prompt(text, max_tokens=5)
print("Обрізаний текст:", cropped_text)
