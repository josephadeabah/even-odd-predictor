import pandas as pd
import numpy as np

def generate_dataset(size=200):
    numbers = np.random.randint(1, 1000, size)
    labels = [1 if n % 2 == 0 else 0 for n in numbers]

    df = pd.DataFrame({
        "number": numbers,
        "label": labels
    })

    df.to_csv("dataset.csv", index=False)
    print("Dataset generated")

if __name__ == "__main__":
    generate_dataset()