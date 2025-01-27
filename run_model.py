import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys
import tensorflow as tf
import time

from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from load_data import load_data
from sklearn.metrics import confusion_matrix

STDOUT = sys.stdout
CLASS_NAMES = [chr(i) for i in range(65, 65 + 26)] # Lettres de A à Z, sans J in Z
CLASS_NAMES.remove("J")
CLASS_NAMES.remove("Z")

def ask_for_model_name():

    already_chosen_reports_names = os.listdir("./reports")
    already_chosen_reports_names = [name.replace(".txt", "") for name in already_chosen_reports_names if name not in [".", ".."]]
    already_chosen_reports_names.sort()
    print("Already chosen names :\n")
    for name in already_chosen_reports_names:
        print(name)

    print()
    report_name = ""
    while ((report_name in already_chosen_reports_names) or (report_name == "")):
        report_name = input("Choose a report name :\n")

    return report_name

X_train, y_train, X_test, y_test = load_data()

beginning = time.time()

# Hyperparamètres (et paramètres)

batch_size = 64
epochs = 10
validation_split = 0.2

# Modèle

def dummy_model(): # Modèle très simple pour tester le code

    model = Sequential(name="dummy")
    
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(26, activation="softmax"))

    model.build()

    return model

def perceptron_model():

    model = Sequential(name="perceptron")

    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(26, activation="softmax"))

    model.build()

    return model
    

def conv_model():

    model = Sequential(name="convolutional")

    model.add(Conv2D(48, (5, 5), activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(72, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.55))
    
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.7))

    model.add(Dense(26, activation="softmax"))

    model.build()

    return model

def do_multiple_simulations(nb_simulations, type_model):

    chrono = time.time()

    print(f"Type model : {type_model}")

    test_losses = []
    test_accuracies = []
    timestamps = []

    for i in range(nb_simulations):
        
        if type_model == "p":
            model = perceptron_model()
        elif type_model == "c":
            model = conv_model()
        elif type_model == "d":
            model = dummy_model()
        else:
            print("Wrong argument :\n'p' for perceptron\n'c' for convolutional (CNN)\n'd' for dummy (tests)")
            return
        
        if i == 0:
            model.summary()

        model.compile(
            optimizer="adam",
            loss=SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            verbose=0
        )

        test_loss, test_acc = model.evaluate(X_test,  y_test, batch_size=batch_size, verbose=2)

        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        timestamps.append(time.time() - chrono)
        chrono = time.time()
        print(f"[{i + 1} / {nb_simulations}] - Timestamp : {timestamps[i]}")

    print(f"--- Losses ---\nMean : {np.mean(test_losses)}\nStd : {np.std(test_losses)}\n\n--- Accuracies ---\nMean : {np.mean(test_accuracies)}\nStd : {np.std(test_accuracies)}\n\n")
    print(f"--- Timestamps ---\nMean : {np.mean(timestamps)}\nStd : {np.std(timestamps)}\n")
    print(f"All losses : {test_losses}\nAll accuracies : {test_accuracies}\nAll timestamps : {timestamps}\n")
    print(f"Max accuracy : {np.max(test_accuracies)}\n")
    print(f"Full time : {sum(timestamps)}\n")

def main(type_model):

    print(f"Type model : {type_model}")
    
    if type_model == "p":
        model = perceptron_model()
    elif type_model == "c":
        model = conv_model()
    elif type_model == "d":
        model = dummy_model()
    else:
        print("Wrong argument :\n'p' for perceptron\n'c' for convolutional (CNN)\n'd' for dummy (tests)")
        return
    
    report_name = ask_for_model_name()

    model.summary()

    model.compile(
        optimizer="adam",
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    test_loss, test_acc = model.evaluate(X_test,  y_test, batch_size=batch_size, verbose=2)
    print(f"Test loss : {test_loss}\nTest accuracy : {test_acc}")
    print(f"Time : {time.time() - beginning}")
    
    model.save(f"./models/{report_name}.h5")

    # Matrice de confusion

    predictions = model.predict(X_test)
    y_pred = np.zeros(len(predictions), dtype="int64")

    for i in range(len(predictions)):
        y_pred[i] = np.argmax(predictions[i])

    print(f"y_test : {y_test.shape}, {y_test.dtype}")
    print(f"y_pred : {y_pred.shape}, {y_pred.dtype}")

    cm = confusion_matrix(y_test, y_pred)
    cm_sum = cm.sum()
    print(cm_sum, cm.shape)
    cm_ratio = cm / cm_sum

    _, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, square=True, annot_kws={"fontsize": 6.5})
    # Si ratio : fmt='.2%'

    ax.set_xlabel("Prédictions")
    ax.set_ylabel("Vraies étiquettes")

    plt.title(f"Modèle {report_name} : matrice de confusion")
    plt.savefig(f"./matconfs/{report_name}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Rapport

    with open(f"./reports/{report_name}.txt", "w") as file:
        file.write(f"Name : {report_name}\n")
        file.write(f"Type : {type_model}\n\n")
        file.write(f"Batch size : {batch_size}\n")
        file.write(f"Epochs : {epochs}\n")
        file.write(f"Validation split : {validation_split}\n\n")

        file.write(f"Test accuracy : {test_acc}\n")
        file.write(f"Test loss : {test_loss}\n\n")
        
        file.write(f"Summary :\n\n")
        sys.stdout = file
        model.summary()
        sys.stdout = STDOUT

        file.write(f"\nAccuracy : {acc}\n\n")
        file.write(f"Loss : {loss}\n\n")
        file.write(f"Validation accuracy : {val_acc}\n\n")
        file.write(f"Validation loss : {val_loss}\n\n")

        """file.write(f"")
        file.write(f"")
        file.write(f"")"""

    # Graphiques

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.suptitle(f"Name = {report_name}, type = {type_model}, batch_size = {batch_size}, epochs = {epochs}, val_split = {validation_split}")

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.grid(True)

    plt.savefig(f"./results/{report_name}.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", type=str, default="p", help="Model type")
    parser.add_argument("-m", "--multiple", type=int, default=None, help="Multiple simulations")
    args = parser.parse_args()

    if args.multiple != None and args.type in ["p", "c", "d"]:
        print("Multiple simulations")
        do_multiple_simulations(args.multiple, args.type)
    elif args.type in ["p", "c", "d"]:
        print("One simulation, with graph / report / conf. matrix")
        main(args.type)
    else:
        print("Wrong input.")


