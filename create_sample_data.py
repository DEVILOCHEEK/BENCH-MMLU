import os
import pandas as pd

def save_examples():
    dev_examples = [
        ["What is 1+1?", "1", "2", "3", "4", "B"],
        ["What color is the sky?", "Blue", "Green", "Red", "Yellow", "A"],
        ["Who wrote \"Romeo and Juliet\"?", "Shakespeare", "Dickens", "Hemingway", "Tolkien", "A"],
        ["Which planet is known as the Red Planet?", "Earth", "Mars", "Jupiter", "Venus", "B"],
        ["What is the boiling point of water (in Celsius)?", "90", "95", "100", "105", "C"]
    ]

    test_examples = [
        ["What is 2+2?", "3", "4", "5", "6", "B"],
        ["What is the capital of France?", "Berlin", "London", "Paris", "Rome", "C"],
        ["Which gas do plants absorb from the atmosphere?", "Oxygen", "Carbon Dioxide", "Nitrogen", "Helium", "B"],
        ["What is the largest mammal?", "Elephant", "Blue Whale", "Giraffe", "Hippopotamus", "B"],
        ["In which continent is Egypt located?", "Asia", "Europe", "Africa", "Australia", "C"]
    ]

    os.makedirs("data/dev", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)

    dev_df = pd.DataFrame(dev_examples)
    test_df = pd.DataFrame(test_examples)

    dev_df.to_csv("data/dev/sample_dev.csv", header=False, index=False)
    test_df.to_csv("data/test/sample_test.csv", header=False, index=False)

if __name__ == "__main__":
    save_examples()
