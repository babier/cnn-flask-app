import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import os

# ======================
# LOAD DATASET CIFAR-10
# ======================
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalisasi
x_train = x_train / 255.0
x_test = x_test / 255.0

# ======================
# BUAT FOLDER MODEL
# ======================
os.makedirs("model", exist_ok=True)

# ======================
# FUNCTION MODEL
# ======================
def create_model(optimizer='adam'):
    model = models.Sequential()

    # Conv Block 1
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    # Conv Block 2
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    # Fully Connected
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    # Output
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ======================
# HYPERPARAMETER TUNING
# ======================
optimizers = ['adam', 'rmsprop']
epochs_list = [10, 20]

best_acc = 0
best_model = None

for opt in optimizers:
    for ep in epochs_list:
        print(f"\nTraining dengan optimizer={opt}, epoch={ep}")

        model = create_model(optimizer=opt)

        history = model.fit(
            x_train, y_train,
            epochs=ep,
            batch_size=64,
            validation_split=0.2,
            verbose=1
        )

        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"Test Accuracy: {acc}")

        if acc > best_acc:
            best_acc = acc
            best_model = model

# ======================
# SIMPAN MODEL TERBAIK
# ======================
best_model.save("model/model.h5")

print("\n=================================")
print(f"Model terbaik disimpan!")
print(f"Akurasi terbaik: {best_acc}")
print("=================================")