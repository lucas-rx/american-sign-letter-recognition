"""
Sources
https://keras.io/guides/sequential_model/#getting-started-with-the-keras-sequential-model
https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
https://www.tensorflow.org/tutorials/quickstart/beginner?hl=fr
https://www.tensorflow.org/tutorials/images/classification?hl=fr
"""

import numpy as np
import pandas as pd
import time

def load_data() -> None:

    beginning = time.time()

    # Prétraitement des données

    train_dataset = pd.read_csv("sign_language_mnist/sign_mnist_train.csv")
    test_dataset = pd.read_csv("sign_language_mnist/sign_mnist_test.csv")

    # a) Entraînement

    X_train_df = train_dataset.drop(columns="label")
    X_train = []

    for _, row in X_train_df.iterrows():
        image = row.to_numpy().reshape((28, 28))
        X_train.append(image)

    X_train = np.array(X_train)
    X_train = X_train / 255

    y_train = train_dataset["label"].values

    # b) Validation

    X_test_df = test_dataset.drop(columns="label")
    X_test = []

    for _, row in X_test_df.iterrows():
        image = row.to_numpy().reshape((28, 28))
        X_test.append(image)

    X_test = np.array(X_test)
    X_test = X_test / 255

    y_test = test_dataset["label"].values

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    print(f"Data loaded in {round(time.time() - beginning, 2)} seconds")

    return (X_train, y_train, X_test, y_test)
