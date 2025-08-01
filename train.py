import argparse
from typing import Tuple

import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_loader import load_data


def build_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train(data_folder: str = './data', model_path: str = 'model.h5', epochs: int = 5) -> tf.keras.Model:
    x_train, y_train, _ = load_data(data_folder)
    X_tr, X_val, y_tr, y_val = train_test_split(x_train.values, y_train.values,
                                                test_size=0.2, random_state=42)
    model = build_model(X_tr.shape[1], y_tr.shape[1])
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=32)
    model.save(model_path)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a baseline model on ChallengeData')
    parser.add_argument('--data-folder', default='./data', help='Folder containing training CSV files')
    parser.add_argument('--model-path', default='model.h5', help='File to save the trained model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    args = parser.parse_args()

    train(args.data_folder, args.model_path, args.epochs)


if __name__ == '__main__':
    main()
