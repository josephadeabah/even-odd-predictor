import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_model():
    df = pd.read_csv("dataset.csv")

    # Feature engineering
    df["mod2"] = df["number"] % 2

    X = df[["mod2"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Model Accuracy:", accuracy)

    joblib.dump(model, "even_odd_model.pkl")
    print("Model saved!")

if __name__ == "__main__":
    train_model()