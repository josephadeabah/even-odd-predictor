import joblib
import numpy as np
import pandas as pd

model = joblib.load("even_odd_model.pkl")

while True:
    user_input = input("Enter a number (q to quit): ")

    if user_input.lower() == "q":
        break

    try:
        number = int(user_input)
        mod2 = number % 2

        data = pd.DataFrame([[mod2]], columns=["mod2"])
        prediction = model.predict(data)

        if prediction[0] == 1:
            print("Prediction: Even")
        else:
            print("Prediction: Odd")

    except ValueError:
        print("Invalid number")