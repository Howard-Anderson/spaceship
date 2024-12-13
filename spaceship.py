"""
                Spaceship Titanic Prediction:

    Author: Howard Anderson.

    Date: 13/12/2024.

    Description: Classification of Survivors.

    Filename: spaceship.py
"""

# General Imports:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-Learn Imports:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision
from sklearn.metrics import f1_score


def main(filename: str) -> None:
    # Loading the Data:
    df = pd.read_csv(filename)

    # Performing EDA:
    print(f"\nInfo: {df.info()}")
    print(f"\nDesciption: {df.describe()}")
    print(f"\nHead: {df.head()}")




if __name__ == "__main__":
    main("train.csv")
